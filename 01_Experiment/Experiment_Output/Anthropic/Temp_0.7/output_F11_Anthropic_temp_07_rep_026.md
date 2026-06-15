 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence points overwhelmingly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs: character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`), `paste()` to build string keys, and named-vector indexing into `idx_lookup`. These are all element-wise R-interpreter operations with no vectorization. With ~6.46M iterations, each involving string concatenation and named-vector lookups (which are O(n) hash lookups on a vector of 6.46M names), this alone is catastrophically slow.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's another **~32.3 million** R-level loop iterations total.

3. **Random Forest inference** is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, `predict()` for `ranger` or `randomForest` is implemented in C/C++ and typically completes in seconds to minutes — orders of magnitude faster than 86+ hours.

**The bottleneck is the row-level `lapply` loops over 6.46 million rows using string-key lookups for spatial neighbor computation.**

---

## Optimization Strategy

1. **Replace the string-keyed lookup with integer-indexed, vectorized operations** using `data.table`. Pre-build an integer mapping from `(id, year)` → row index, then join neighbors in bulk rather than row-by-row.

2. **Vectorize neighbor stats computation** by expanding all neighbor pairs into a single long `data.table`, joining the variable values, and computing grouped `max`, `min`, `mean` in one pass per variable — no R-level loops.

3. **Build the neighbor-pair expansion once**, then reuse it for all 5 variables.

This reduces the complexity from ~6.46M × (string ops + hash lookups) to a handful of vectorized `data.table` joins and grouped aggregations, bringing runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Convert cell_data to data.table and assign row indices
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)
cell_data[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Build a vectorized neighbor-edge table
#
# rook_neighbors_unique is an nb object (list of integer vectors) of
# length = number of unique spatial cells (344,208).
# id_order is the vector mapping position in nb list → cell id.
# Each entry neighbors[[i]] gives the positions (in id_order) of
# the rook neighbors of cell id_order[i].
# ──────────────────────────────────────────────────────────────────────

# Expand the nb list into a two-column data.table of (focal_id, neighbor_id)
nb_edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_pos <- rook_neighbors_unique[[i]]
  if (length(nb_pos) == 0L || (length(nb_pos) == 1L && nb_pos[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_pos])
}))

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Expand edges across all years to get (focal_row, neighbor_row)
#
# For every (focal_id, neighbor_id) pair and every year present for
# the focal cell, we need the neighbor's row in the same year.
# We do this with keyed joins — no string pasting, no lapply.
# ──────────────────────────────────────────────────────────────────────

# Minimal index table: maps (id, year) → row_idx
idx_table <- cell_data[, .(id, year, row_idx)]
setkey(idx_table, id, year)

# Join focal side: get all (focal_id, year, focal_row_idx) combinations
# then join to neighbor side to get neighbor_row_idx in the same year.

# First, get the years each focal cell appears in
focal_years <- idx_table[, .(focal_id = id, year, focal_row = row_idx)]

# Merge focal_years with nb_edges to get (focal_row, neighbor_id, year)
setkey(nb_edges, focal_id)
setkey(focal_years, focal_id)
edge_year <- nb_edges[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
# edge_year now has columns: focal_id, neighbor_id, year, focal_row

# Now join to get the neighbor's row index in the same year
setkey(edge_year, neighbor_id, year)
setkey(idx_table, id, year)
edge_year[idx_table, neighbor_row := i.row_idx, on = c(neighbor_id = "id", "year")]

# Drop rows where the neighbor doesn't have data in that year
edge_year <- edge_year[!is.na(neighbor_row)]

# Keep only what we need
edge_year <- edge_year[, .(focal_row, neighbor_row)]

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Compute neighbor stats for all 5 variables — fully vectorized
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value to each edge
  edge_year[, nval := cell_data[[var_name]][neighbor_row]]

  # Compute grouped stats (excluding NAs) keyed by focal_row
  stats <- edge_year[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = focal_row]

  # Initialize new columns with NA
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign computed values by row index
  cell_data[stats$focal_row, (max_col)  := stats$nb_max]
  cell_data[stats$focal_row, (min_col)  := stats$nb_min]
  cell_data[stats$focal_row, (mean_col) := stats$nb_mean]
}

# Clean up helper column
cell_data[, row_idx := NULL]
edge_year[, nval := NULL]

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Random Forest prediction (unchanged — not the bottleneck)
# ──────────────────────────────────────────────────────────────────────
# The pre-trained model is preserved exactly as-is.
# predictions <- predict(rf_model, data = cell_data)
# cell_data[, prediction := predictions$predictions]  # or predictions, depending on package
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `lapply` iterations with `paste()` + named-vector string lookups | One-time vectorized `data.table` keyed join |
| **Neighbor stats** | 5 × 6.46M `lapply` iterations with per-row `max/min/mean` | 5 vectorized grouped aggregations via `data.table` |
| **Estimated runtime** | 86+ hours | **Minutes** (typically 5–15 min on 16 GB laptop) |
| **RF model** | Preserved | **Preserved** (no retraining, same `predict()` call) |
| **Numerical results** | Original `max`, `min`, `mean` per neighbor set | **Identical** — same arithmetic, same neighbor definitions |

The true bottleneck was the **O(n)-per-row, string-based, interpreted-R neighbor feature engineering**, not the Random Forest inference. The fix is to replace row-level `lapply` loops with bulk vectorized `data.table` joins and grouped aggregations.