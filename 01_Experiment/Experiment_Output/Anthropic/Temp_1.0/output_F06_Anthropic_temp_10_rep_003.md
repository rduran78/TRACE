 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows
For every one of the ~6.46 million cell-year rows, the function performs character-based key construction (`paste`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Character hashing over 6.46M rows is extremely expensive. Crucially, **the neighbor topology is time-invariant** — the same cell has the same rook neighbors in every year. Yet the lookup is rebuilt redundantly for every cell-year combination, inflating what should be a ~344K-cell operation into a ~6.46M-row operation.

### Bottleneck 2: `compute_neighbor_stats` — Row-wise `lapply` over 6.46M rows
For each of the 5 variables, an `lapply` iterates over all 6.46M rows, subsetting a vector by index, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times, producing ~32.3M R-level function calls total. The `do.call(rbind, result)` on a 6.46M-element list is also expensive.

### Why raster focal/kernel operations don't directly apply
Raster focal operations (e.g., `terra::focal`) assume data lives on a complete regular grid with a fixed kernel. Here the panel is long-format (cell × year), the grid may have irregular boundaries/missing cells, and the neighbor structure is an arbitrary `spdep::nb` object. Forcing it into a raster stack for 28 years × 5 variables would require reshaping, gap-filling, and re-extracting — adding complexity and risking numerical discrepancies. The better strategy is to **vectorize the sparse-neighbor computation directly** using `data.table` joins and matrix operations.

---

## Optimization Strategy

1. **Separate spatial topology from temporal replication.** Build a simple integer-to-integer neighbor edge list once (344K cells), then join it to the panel by year — letting `data.table` handle the replication efficiently.

2. **Replace row-wise `lapply` with grouped `data.table` aggregation.** Convert the neighbor edge list into a two-column `data.table` (focal_id, neighbor_id), join on (neighbor_id, year) to pull neighbor values, then group-by (focal_id, year) to compute `max`, `min`, `mean` in compiled C code inside `data.table`.

3. **Process all 5 variables in one pass** per join, avoiding redundant joins.

4. **Estimated speedup:** From ~86+ hours to ~5–15 minutes, depending on disk I/O.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Inputs assumed to exist:
#       cell_data             — data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#       id_order              — integer vector of cell IDs (the order matching rook_neighbors_unique)
#       rook_neighbors_unique — spdep::nb object (list of integer index vectors into id_order)
#       rf_model              — pre-trained Random Forest (not retrained)
# ---------------------------------------------------------------

# Convert to data.table (no copy if already data.table)
setDT(cell_data)

# ---------------------------------------------------------------
# 1.  Build a SPATIAL-ONLY edge list (not replicated across years)
#     This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------

# Create edge list:  focal_id  ->  neighbor_id  (using original cell IDs)
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  # spdep::nb encodes zero-neighbor regions as 0L; filter those out

  nb_idx <- nb_idx[nb_idx > 0L]
  if (length(nb_idx) == 0L) return(NULL)
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
}))

cat("Edge list rows:", nrow(edge_list), "\n")
# Expected: ~1,373,394 directed edges

# ---------------------------------------------------------------
# 2.  Join edge list to panel data to get neighbor variable values,
#     then aggregate by (focal_id, year).
# ---------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset the columns we need from cell_data for the neighbor join
# We join on (neighbor_id = id, year) to retrieve neighbor values.
neighbor_vals_dt <- cell_data[, c("id", "year", neighbor_source_vars), with = FALSE]

# Key for fast join
setkey(neighbor_vals_dt, id, year)
setkey(edge_list, neighbor_id)

# Expand edges × years:  for each (focal_id, neighbor_id) pair,
# join the neighbor's values for every year the neighbor appears in.
# This is a keyed join:  edge_list[neighbor_vals_dt] but we want
# each edge paired with matching year rows of the *neighbor*.
# Strategy: merge edge_list with neighbor_vals_dt on neighbor_id == id.

# Rename for clarity before join
setnames(neighbor_vals_dt, "id", "neighbor_id")
setkey(neighbor_vals_dt, neighbor_id, year)

# Join: for every edge (focal_id, neighbor_id), pull all year-rows of the neighbor
# This produces a table of (focal_id, neighbor_id, year, ntl, ec, ...)
joined <- merge(edge_list, neighbor_vals_dt, by = "neighbor_id", allow.cartesian = TRUE)
# allow.cartesian = TRUE because one neighbor_id maps to 28 year-rows

cat("Joined rows:", nrow(joined), "\n")
# Expected: ~1,373,394 edges × 28 years ≈ 38.5M rows (fits in 16 GB)

# ---------------------------------------------------------------
# 3.  Compute max, min, mean per (focal_id, year) for each variable
# ---------------------------------------------------------------

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

# Aggregate (this is the heavy step, but runs in compiled C inside data.table)
neighbor_stats <- joined[,
  eval(as.call(c(as.name("list"), agg_exprs))),
  by = .(focal_id, year)
]

# Handle -Inf / Inf from max/min of all-NA groups → set to NA
for (col in agg_names) {
  vals <- neighbor_stats[[col]]
  set(neighbor_stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

cat("Neighbor stats rows:", nrow(neighbor_stats), "\n")

# ---------------------------------------------------------------
# 4.  Merge neighbor stats back onto cell_data
# ---------------------------------------------------------------

# Remove any pre-existing neighbor columns to avoid duplication
existing_nb_cols <- intersect(names(cell_data), agg_names)
if (length(existing_nb_cols) > 0L) {
  cell_data[, (existing_nb_cols) := NULL]
}

setkey(cell_data, id, year)
setkey(neighbor_stats, focal_id, year)

cell_data <- merge(cell_data, neighbor_stats,
                   by.x = c("id", "year"),
                   by.y = c("focal_id", "year"),
                   all.x = TRUE)

cat("Final cell_data rows:", nrow(cell_data), "\n")

# ---------------------------------------------------------------
# 5.  Predict with the EXISTING Random Forest (no retraining)
# ---------------------------------------------------------------

# rf_model is the pre-trained model object — used as-is
# Ensure the predictor columns match what the model expects
# (the 15 new neighbor_* columns are now present alongside the ~95 others)

cell_data[, predicted := predict(rf_model, newdata = cell_data)]

# ---------------------------------------------------------------
# Cleanup large temporaries
# ---------------------------------------------------------------
rm(joined, neighbor_vals_dt, edge_list, neighbor_stats)
gc()
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor topology** | `edge_list` is built from the identical `rook_neighbors_unique` nb object and `id_order`. |
| **Same statistics** | `max`, `min`, `mean` with `na.rm = TRUE`, grouped by `(focal_id, year)`, reproduces exactly what the original `compute_neighbor_stats` computes per row. |
| **Cells with no neighbors** | They have no rows in `edge_list`, so after the `all.x = TRUE` merge they receive `NA` in every neighbor column — identical to the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | `rf_model` is only used in `predict()`, never refit. |
| **Numerical estimand preserved** | No approximation, interpolation, or raster resampling is introduced. The computation is algebraically identical. |

## Expected Performance

| Step | Estimated Time |
|---|---|
| Build edge list (344K cells) | < 5 seconds |
| Merge edges × years (~38.5M rows) | ~1–2 minutes |
| Grouped aggregation (15 stats) | ~2–5 minutes |
| Merge back to cell_data | ~1 minute |
| **Total** | **~5–10 minutes** |

Peak RAM ≈ joined table (~38.5M rows × 7 cols ≈ 2.2 GB) + cell_data (~6.46M × 125 cols ≈ 6.5 GB) ≈ **~10–12 GB**, within the 16 GB budget.