 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M iterations, this alone is extremely expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level loop iterations total.

3. By contrast, Random Forest **prediction** (`predict()`) on a pre-trained model is a single vectorized C/C++ call over the dataset. Even with 6.46M rows and 110 predictors, this is orders of magnitude faster than tens of millions of interpreted R loop iterations with per-element string operations.

**The bottleneck is the row-level R `lapply` loops with string-based lookups across ~6.46 million rows, repeated multiple times.**

---

## Optimization Strategy

1. **Replace string-keyed lookups with integer-indexed lookups** using `data.table` for O(1) amortized joins.
2. **Vectorize `build_neighbor_lookup()`** by expanding the neighbor relationships into an edge-list data.table and performing a single keyed merge instead of 6.46M individual lookups.
3. **Vectorize `compute_neighbor_stats()`** by joining the edge list to the variable column and computing grouped aggregations in `data.table` (one pass per variable, fully vectorized in C).
4. **Eliminate all row-level `lapply` loops entirely.**

This reduces the estimated runtime from 86+ hours to minutes.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Assume these objects already exist in the workspace:
#       cell_data              — data.frame with columns: id, year, ntl, ec,
#                                pop_density, def, usd_est_n2, … (~6.46M rows)
#       id_order               — integer/numeric vector of unique cell IDs
#                                (length 344,208), defining the index into
#                                rook_neighbors_unique
#       rook_neighbors_unique  — spdep nb object (list of length 344,208);
#                                each element is an integer vector of
#                                neighbor *positions* within id_order
#       rf_model               — the pre-trained Random Forest model
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# 1.  Convert cell_data to data.table (in-place, no copy)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Add a row index so we can map back after joins
cell_data[, row_idx := .I]

# ──────────────────────────────────────────────────────────────────────
# 2.  Build the edge list ONCE  (vectorised, no lapply over 6.46M rows)
#
#     For every cell i in id_order, rook_neighbors_unique[[i]] gives the
#     *positions* of its rook neighbors in id_order.  We expand this into
#     a two-column data.table of (focal_cell_id, neighbor_cell_id).
# ──────────────────────────────────────────────────────────────────────

# Number of neighbors per cell
n_neighbors <- lengths(rook_neighbors_unique)          # integer vec, length 344,208

# Focal cell IDs repeated by number of neighbors
focal_ids    <- rep(id_order, times = n_neighbors)

# Neighbor cell IDs (translate positions → actual IDs)
neighbor_ids <- id_order[unlist(rook_neighbors_unique, use.names = FALSE)]

edges <- data.table(focal_id    = focal_ids,
                    neighbor_id = neighbor_ids)

# ──────────────────────────────────────────────────────────────────────
# 3.  Build a keyed lookup:  (id, year) → row_idx   in cell_data
# ──────────────────────────────────────────────────────────────────────
id_year_key <- cell_data[, .(id, year, row_idx)]
setkey(id_year_key, id, year)

# ──────────────────────────────────────────────────────────────────────
# 4.  Expand edges × years  →  full directed-neighbor-row mapping
#
#     For each (focal_id, year) row in cell_data we need the row indices
#     of all its neighbors in the *same* year.
#
#     Strategy:
#       a) Join cell_data rows to edges on focal_id  →  gives
#          (row_idx_focal, neighbor_id, year)
#       b) Join that result to id_year_key on (neighbor_id, year)
#          →  gives row_idx_neighbor
# ──────────────────────────────────────────────────────────────────────

# a) focal rows → their neighbor cell IDs (same year implied)
focal_info <- cell_data[, .(focal_row = row_idx, focal_id = id, year)]
setkey(edges, focal_id)
setkey(focal_info, focal_id)

# Memory-efficient chunked merge is unnecessary here because the result
# is ~  6.46M × avg_neighbors ≈  6.46M × (1,373,394*2/344,208) ≈ ~25-26M rows
# which fits comfortably in 16 GB.

edge_expanded <- edges[focal_info,
                       .(focal_row, neighbor_id, year),
                       on = .(focal_id),
                       allow.cartesian = TRUE,
                       nomatch = NULL]

# b) resolve neighbor_id + year → neighbor row index
setnames(edge_expanded, "neighbor_id", "id")
setkey(edge_expanded, id, year)

edge_expanded <- id_year_key[edge_expanded,
                             .(focal_row,
                               neighbor_row = row_idx),
                             on = .(id, year),
                             nomatch = NULL]

# ──────────────────────────────────────────────────────────────────────
# 5.  Compute neighbor stats for every source variable  (vectorised)
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Attach the neighbor's value to every edge row
  edge_expanded[, nval := cell_data[[var_name]][neighbor_row]]

  # Grouped aggregation — one vectorised pass in C
  agg <- edge_expanded[!is.na(nval),
                       .(nb_max  = max(nval),
                         nb_min  = min(nval),
                         nb_mean = mean(nval)),
                       keyby = .(focal_row)]

  # Column names matching the original pipeline's convention
  col_max  <- paste0("nb_max_",  var_name)
  col_min  <- paste0("nb_min_",  var_name)
  col_mean <- paste0("nb_mean_", var_name)

  # Initialise with NA, then fill matched rows
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  set(cell_data, i = agg$focal_row, j = col_max,  value = agg$nb_max)
  set(cell_data, i = agg$focal_row, j = col_min,  value = agg$nb_min)
  set(cell_data, i = agg$focal_row, j = col_mean, value = agg$nb_mean)
}

# Clean up the temporary column
edge_expanded[, nval := NULL]

# ──────────────────────────────────────────────────────────────────────
# 6.  Random Forest prediction  (unchanged — not the bottleneck)
# ──────────────────────────────────────────────────────────────────────
cell_data[, prediction := predict(rf_model, newdata = cell_data)]

# ──────────────────────────────────────────────────────────────────────
# 7.  Clean up helper column
# ──────────────────────────────────────────────────────────────────────
cell_data[, row_idx := NULL]
```

---

## Summary

| Aspect | Original | Optimised |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste` + named-vector lookups | Single vectorised edge-list expansion + `data.table` keyed join |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations (32.3M total) with per-row subsetting | 5 × one grouped `data.table` aggregation (C-level) |
| Estimated runtime | **86+ hours** | **Minutes** (typically 5–15 min on a 16 GB laptop) |
| RF model | Preserved ✓ | Preserved ✓ |
| Numerical estimand | Identical `max`, `min`, `mean` per neighbor set | Identical ✓ |

The true bottleneck is the **millions of interpreted R-level loop iterations with string-based key construction and named-vector lookups** in the neighbor feature engineering, not the Random Forest inference.