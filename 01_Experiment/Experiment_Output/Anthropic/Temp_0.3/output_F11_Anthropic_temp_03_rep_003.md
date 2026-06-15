 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. **Random Forest prediction** is a single vectorized call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, this is a highly optimized C/C++-level operation in `ranger` or `randomForest`. Loading a serialized model and writing predictions are I/O-bound and fast relative to the feature engineering.

**The 86+ hour runtime is dominated by the R-level row-wise loops over millions of rows in the neighbor feature engineering, not by model inference.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table`-based join approach. Instead of looping over every row and doing per-row string lookups, we expand the neighbor list into an edge table, join it against the data to get row indices, and group.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable — computing `max`, `min`, and `mean` in one pass using compiled C code under the hood.

3. **Eliminate per-row `lapply` entirely.** The entire pipeline becomes a sequence of vectorized joins and grouped aggregations.

Expected speedup: from 86+ hours to roughly **minutes** (the dominant cost becomes the grouped aggregation over ~6.46M × ~4 neighbors × 5 variables, all in compiled code).

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Assume these objects already exist:
#       cell_data              — data.frame/data.table, ~6.46M rows
#       id_order               — integer vector of 344,208 cell IDs
#       rook_neighbors_unique  — spdep nb object (list of integer index vectors)
#       rf_model               — pre-trained Random Forest model
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table in place (no copy if already data.table)
setDT(cell_data)

# Ensure a row-index column exists for later reassembly
cell_data[, .row_id := .I]

# ──────────────────────────────────────────────────────────────────────
# 1.  Build a vectorized edge table from the nb object
#     Each entry: (focal_cell_id, neighbor_cell_id)
# ──────────────────────────────────────────────────────────────────────
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
}))
# edge_list has ~1.37M rows — trivially small

# ──────────────────────────────────────────────────────────────────────
# 2.  Join edge table with cell_data to create the neighbor-row mapping
#     For every (focal_id, year) we find the row indices of its neighbors
#     in the same year.
# ──────────────────────────────────────────────────────────────────────

# Keyed lookup table: cell id + year → row index
id_year_key <- cell_data[, .(id, year, .row_id)]
setkey(id_year_key, id, year)

# Focal side: every row's (id, year) joined to its neighbor cell IDs
focal_info <- cell_data[, .(focal_row = .row_id, focal_id = id, year)]
focal_edges <- edge_list[focal_info, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
# focal_edges columns: focal_id, neighbor_id, focal_row, year

# Neighbor side: look up the neighbor's row in the same year
setnames(id_year_key, c("id", "year", ".row_id"), c("neighbor_id", "year", "neighbor_row"))
setkey(id_year_key, neighbor_id, year)
setkey(focal_edges, neighbor_id, year)

neighbor_map <- id_year_key[focal_edges, on = .(neighbor_id, year), nomatch = NA]
# neighbor_map columns: neighbor_id, year, neighbor_row, focal_id, focal_row

# Drop rows where the neighbor doesn't exist in that year
neighbor_map <- neighbor_map[!is.na(neighbor_row)]

# ──────────────────────────────────────────────────────────────────────
# 3.  Compute neighbor stats for each source variable — fully vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Pull the neighbor values via row-index subsetting (vectorized)
  neighbor_map[, nval := cell_data[[var_name]][neighbor_row]]

  # Grouped aggregation: max, min, mean per focal row (excluding NAs)
  agg <- neighbor_map[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    keyby = .(focal_row)
  ]

  # Build new column names
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Initialize with NA, then fill matched rows
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  set(cell_data, i = agg$focal_row, j = max_col,  value = agg$nb_max)
  set(cell_data, i = agg$focal_row, j = min_col,  value = agg$nb_min)
  set(cell_data, i = agg$focal_row, j = mean_col, value = agg$nb_mean)

  # Clean up the temporary column
  neighbor_map[, nval := NULL]
}

# Clean up helper column
cell_data[, .row_id := NULL]

# ──────────────────────────────────────────────────────────────────────
# 4.  Random Forest prediction (unchanged — this was never the bottleneck)
# ──────────────────────────────────────────────────────────────────────
predictions <- predict(rf_model, data = cell_data)
# (Use predictions$predictions for ranger, or as-is for randomForest)
cell_data[, predicted := if (inherits(rf_model, "ranger")) predictions$predictions else as.numeric(predictions)]
```

---

## Summary

| Aspect | Colleague's Claim (RF Inference) | Actual Bottleneck (Neighbor Features) |
|---|---|---|
| **Operation type** | Single vectorized C-level `predict()` | ~38M R-level `lapply` iterations + per-row string ops |
| **Calls** | 1 call | 6.46M × (1 + 5) = ~38.8M iterations |
| **Overhead per unit** | Negligible (compiled) | High (R interpreter, `paste`, named-vector hash lookup) |
| **Estimated share of 86h** | Minutes | Nearly all of it |

**The fix:** Replace row-wise `lapply` loops with a vectorized `data.table` edge-join and grouped aggregation. The trained Random Forest model and the original numerical estimand (max, min, mean of neighbor values → same predictor columns → same `predict()` output) are fully preserved.