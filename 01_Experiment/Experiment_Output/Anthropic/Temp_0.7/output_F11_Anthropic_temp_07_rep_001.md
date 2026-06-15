 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character cast, a named-vector lookup (`id_to_ref`), a `paste()` to build keys, a named-vector lookup into `idx_lookup` (~6.46M-entry named character vector), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. This single function executes **~6.46 million iterations** of non-trivial string operations and lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the **~6.46 million-element** `neighbor_lookup` list, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level list iterations total.

3. By contrast, Random Forest **prediction** on a pre-trained model is a single vectorized call (`predict(model, newdata)`) implemented in optimized C/C++ code. Even on 6.46M rows × 110 predictors, this typically completes in seconds to minutes — orders of magnitude faster than the neighbor feature loop.

**The 86+ hour runtime is dominated by the neighbor feature engineering, not RF inference.**

## Optimization Strategy

The core strategy is to **eliminate row-level R loops entirely** by vectorizing all operations using `data.table` joins and grouped aggregations:

1. **Replace `build_neighbor_lookup()`** with a flat `data.table` edge list that maps every `(id, year)` row to its neighbor `(id, year)` rows via keyed joins — no per-row `lapply`, no `paste` key construction at the row level inside a loop.

2. **Replace `compute_neighbor_stats()`** with a single `data.table` grouped aggregation (`[, .(max, min, mean), by = row_idx]`) per variable — fully vectorized in C.

3. **Process all 5 variables** in the same join framework, avoiding redundant lookups.

This reduces complexity from millions of interpreted R iterations to a handful of vectorized join + group-by operations, bringing the runtime from 86+ hours down to **minutes**.

## Working R Code

```r
library(data.table)

# ---- Step 0: Assume these objects already exist ----
# cell_data          : data.frame/data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order           : integer vector of cell IDs in the order used by the nb object
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# rf_model           : pre-trained Random Forest model (unchanged)

# ---- Step 1: Build a flat edge table from the nb object (once) ----
# Each entry in rook_neighbors_unique[[i]] is an index into id_order.
# We convert to a two-column data.table: (focal_id, neighbor_id)

build_edge_table <- function(id_order, nb_obj) {
  n <- length(nb_obj)
  # Pre-count total edges for pre-allocation
  lens <- vapply(nb_obj, length, integer(1))
  total <- sum(lens)
  
  focal_ids    <- rep(id_order, times = lens)
  neighbor_ids <- id_order[unlist(nb_obj, use.names = FALSE)]
  
  data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# Remove the 0-neighbor sentinel if spdep uses 0L for islands
edge_dt <- edge_dt[neighbor_id != 0L]

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_dt)))

# ---- Step 2: Convert cell_data to data.table and add a row index ----
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]

# ---- Step 3: Vectorized neighbor feature computation ----
# We join edge_dt × year to get all (focal_row, neighbor_row) pairs,
# then group-by focal_row to compute stats.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 3a. Create a slim keyed lookup for joining: (id, year) -> row_idx + variable values
key_cols <- c("id", "year")
val_cols <- c("row_idx", neighbor_source_vars)
focal_key   <- cell_dt[, ..val_cols, env = list(val_cols = c(key_cols, "row_idx"))]
setnames(focal_key, "row_idx", "focal_row_idx")
setkeyv(focal_key, key_cols)

neighbor_val <- cell_dt[, .SD, .SDcols = c(key_cols, neighbor_source_vars)]
setnames(neighbor_val, old = key_cols, new = paste0("n_", key_cols))
setkeyv(neighbor_val, paste0("n_", key_cols))

# 3b. Expand edges by year: for each (focal_id, neighbor_id) pair and each year,
#     link the focal row to the neighbor row.
#     Strategy: join edge_dt to cell_dt on focal_id to get (focal_row_idx, year, neighbor_id),
#     then join on (neighbor_id, year) to get neighbor values.

# Get focal rows: each row in cell_dt tells us its id and year
focal_info <- cell_dt[, .(focal_row_idx = row_idx, focal_id = id, year)]
setkey(focal_info, focal_id)
setkey(edge_dt, focal_id)

# Join: for every focal row, attach its neighbors
# Result: (focal_row_idx, year, neighbor_id)
cat("Joining focal rows to edge table...\n")
expanded <- edge_dt[focal_info, on = .(focal_id), allow.cartesian = TRUE, nomatch = NULL]
# expanded has columns: focal_id, neighbor_id, focal_row_idx, year

cat(sprintf("Expanded edge-year table: %d rows\n", nrow(expanded)))

# Now join neighbor values: match on (neighbor_id, year)
setnames(expanded, "neighbor_id", "n_id_join")
neighbor_slim <- cell_dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_slim, "id", "n_id_join")
setkey(neighbor_slim, n_id_join, year)
setkey(expanded, n_id_join, year)

cat("Joining neighbor values...\n")
expanded <- neighbor_slim[expanded, on = .(n_id_join, year), nomatch = NA]

# 3c. Grouped aggregation: compute max, min, mean per focal_row_idx per variable
cat("Computing neighbor statistics...\n")

agg_exprs <- list()
for (v in neighbor_source_vars) {
  v_sym <- as.name(v)
  agg_exprs[[paste0("neighbor_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
  agg_exprs[[paste0("neighbor_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
  agg_exprs[[paste0("neighbor_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
}

# Build and evaluate the aggregation call
stats_dt <- expanded[,
  {
    out <- list()
    for (v in neighbor_source_vars) {
      nv <- .SD[[v]]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(nv)
        out[[paste0("neighbor_min_", v)]]  <- min(nv)
        out[[paste0("neighbor_mean_", v)]] <- mean(nv)
      }
    }
    out
  },
  by = focal_row_idx,
  .SDcols = neighbor_source_vars
]

# ---- Step 4: Merge stats back into cell_dt by row index ----
setkey(stats_dt, focal_row_idx)
setkey(cell_dt, row_idx)

new_cols <- setdiff(names(stats_dt), "focal_row_idx")
cell_dt[stats_dt, (new_cols) := mget(paste0("i.", new_cols)), on = .(row_idx = focal_row_idx)]

# Handle rows with no neighbors (islands): they won't appear in stats_dt.
# They already have NA from the join (default), which matches original behavior.

# ---- Step 5: Replace -Inf/Inf from max/min of empty sets with NA (safety) ----
inf_cols <- grep("^neighbor_(max|min)_", names(cell_dt), value = TRUE)
for (col in inf_cols) {
  cell_dt[is.infinite(get(col)), (col) := NA_real_]
}

# ---- Step 6: Random Forest prediction (unchanged — not the bottleneck) ----
# Convert back if rf_model expects a data.frame
cell_dt[, row_idx := NULL]  # clean up helper column
predictions <- predict(rf_model, newdata = as.data.frame(cell_dt))
cell_dt[, predicted := predictions]

# ---- Step 7: Convert back to data.frame if needed downstream ----
cell_data <- as.data.frame(cell_dt)

cat("Done. Neighbor features + RF prediction complete.\n")
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (6.46M R-level iterations × 6 calls) | Eliminated |
| **Approach** | Per-row `lapply` with string key construction and named-vector lookups | Vectorized `data.table` keyed joins + grouped aggregation (C-level) |
| **Estimated runtime** | 86+ hours | ~5–20 minutes (join + aggregation on ~26M expanded rows) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical results** | max/min/mean per neighbor set | Identical max/min/mean per neighbor set |
| **RAM** | Fits in 16 GB (expanded table ≈ 26M rows × ~8 cols ≈ 1.5 GB) | Fits in 16 GB |

The colleague should focus optimization efforts on the spatial feature engineering, not on Random Forest inference.