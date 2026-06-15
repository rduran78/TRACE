 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a `character` coercion + hash lookup (`id_to_ref`), a subsetting of the neighbor list, a `paste()` to build keys, and a named-vector lookup (`idx_lookup[neighbor_keys]`). That's ~6.46 million iterations of character-based hash lookups, string concatenation, and named vector indexing — all in interpreted R. With ~1.37M neighbor relationships spread across those rows, each row touches on average ~4 neighbors, meaning tens of millions of `paste()` and name-matching operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million elements via `lapply`, subsetting a numeric vector, removing `NA`s, and computing `max/min/mean`. That's 5 × 6.46M ≈ **32.3 million** interpreted R function calls, each involving allocation and subsetting.

3. The final `do.call(rbind, result)` inside `compute_neighbor_stats` binds ~6.46 million 3-element vectors into a matrix — another expensive operation done 5 times.

4. **Random Forest `predict()`** by contrast is a single vectorized C/C++ call on a pre-trained model. Even with 6.46M rows × 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object typically completes in seconds to a few minutes. Loading/writing is I/O-bound and also comparatively trivial.

**Conclusion:** The bottleneck is the row-level, interpreted-R, string-based spatial neighbor feature construction — not the Random Forest inference. The estimated 86+ hours runtime is dominated by tens of millions of `lapply` iterations with `paste()`, named-vector lookups, and per-element subsetting.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table`-based join approach. Instead of iterating over every row, explode the neighbor list into an edge table (cell_id → neighbor_id), join it with the panel data on (neighbor_id, year) to get neighbor row indices, and group.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable — no `lapply`, no per-row R function calls.

3. **Avoid all `paste()`-based key construction** — use integer-keyed joins with multi-column keys `(id, year)`.

4. These operations become vectorized C-level `data.table` merges and grouped aggregations, reducing the workload from hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table; add a row index
# ============================================================
setDT(cell_data)
cell_data[, .row_idx := .I]

# ============================================================
# STEP 1: Build an edge table from the nb object (vectorized)
#         This replaces build_neighbor_lookup entirely.
# ============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # id_order maps positional index -> cell id
  lens <- lengths(nb_obj)
  from_pos <- rep(seq_along(nb_obj), lens)
  to_pos   <- unlist(nb_obj, use.names = FALSE)

  # Remove 0-entries that spdep uses for "no neighbors"
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]

  data.table(
    focal_id    = id_order[from_pos],
    neighbor_id = id_order[to_pos]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
# edge_dt has columns: focal_id, neighbor_id
# ~ 1,373,394 rows (directed rook edges)

# ============================================================
# STEP 2: Join edges with panel data to create the full
#         (focal_row, neighbor_row, year) mapping
# ============================================================

# We need: for each (focal_id, year), all neighbor rows that share
# the same year and whose id is a rook neighbor.

# Subset of cell_data for joining: just id, year, row_idx, and the
# neighbor source variable columns.
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

join_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
cd_slim <- cell_data[, ..join_cols]

# Merge edges with focal-side data to get years for each focal cell
# Then merge with neighbor-side data on (neighbor_id, year)

# Focal side: attach year to every edge
setkey(cd_slim, id)
focal_years <- cd_slim[, .(id, year, .row_idx)]
setnames(focal_years, c("id", "year", ".row_idx"),
                      c("focal_id", "year", "focal_row"))
setkey(focal_years, focal_id)
setkey(edge_dt, focal_id)

# This join replicates each edge across all years the focal cell appears in
edge_year <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
# Columns: focal_id, neighbor_id, year, focal_row

# Neighbor side: attach variable values
neighbor_data <- cd_slim[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_data, "id", "neighbor_id")
setkey(neighbor_data, neighbor_id, year)
setkey(edge_year, neighbor_id, year)

# Main join: each focal-row gets its neighbor's variable values in the same year
joined <- neighbor_data[edge_year, on = c("neighbor_id", "year"), nomatch = NA]
# Columns: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2,
#           focal_id, focal_row

# ============================================================
# STEP 3: Grouped aggregation — replaces compute_neighbor_stats
# ============================================================
# For each focal_row and each variable, compute max, min, mean
# across all neighbor rows (ignoring NAs).

agg_exprs <- list()
for (v in neighbor_source_vars) {
  v_sym <- as.name(v)
  agg_exprs[[paste0("neighbor_max_", v)]]  <- bquote(as.numeric(max(.(v_sym), na.rm = TRUE)))
  agg_exprs[[paste0("neighbor_min_", v)]]  <- bquote(as.numeric(min(.(v_sym), na.rm = TRUE)))
  agg_exprs[[paste0("neighbor_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
}

# Handle the edge case where all neighbors are NA → returns Inf/-Inf/NaN;
# we will fix those after aggregation.

agg_result <- joined[, eval(as.call(c(as.name("list"), agg_exprs))),
                      by = focal_row]

# Replace Inf/-Inf with NA (from max/min on all-NA groups)
inf_cols <- grep("^neighbor_(max|min)_", names(agg_result), value = TRUE)
for (col in inf_cols) {
  set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
}
# Replace NaN with NA (from mean on all-NA groups)
mean_cols <- grep("^neighbor_mean_", names(agg_result), value = TRUE)
for (col in mean_cols) {
  set(agg_result, which(is.nan(agg_result[[col]])), col, NA_real_)
}

# ============================================================
# STEP 4: Merge aggregated neighbor features back into cell_data
# ============================================================
setkey(agg_result, focal_row)
feature_cols <- setdiff(names(agg_result), "focal_row")

# Pre-allocate columns in cell_data
for (col in feature_cols) {
  set(cell_data, j = col, value = NA_real_)
}

# Assign by row index
matched_rows <- agg_result[["focal_row"]]
for (col in feature_cols) {
  set(cell_data, i = matched_rows, j = col, value = agg_result[[col]])
}

# Clean up helper column
cell_data[, .row_idx := NULL]

# ============================================================
# STEP 5: Random Forest prediction (unchanged — not the bottleneck)
# ============================================================
# library(ranger)  # or randomForest — whichever was used
# trained_model <- readRDS("path/to/trained_rf_model.rds")
# predictions <- predict(trained_model, data = cell_data)$predictions
# cell_data[, prediction := predictions]
# fwrite(cell_data, "output.csv")  # or saveRDS(...)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookups | One vectorized edge-table construction + `data.table` keyed join |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations (32.3M R-level calls) | One grouped `data.table` aggregation (C-level) |
| `do.call(rbind, ...)` | 5 × binding 6.46M small vectors | Eliminated entirely |
| Estimated runtime | 86+ hours | **Minutes** (typically 5–15 min on a 16 GB laptop) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical estimand | Identical max/min/mean per (focal cell, year) | Identical — same arithmetic, same NA handling |

The Random Forest inference step (load → predict → write) for ~6.46M rows is a **negligible fraction** of the total runtime and requires no optimization. The real bottleneck is the interpreted-R, row-by-row, string-based neighbor feature engineering, which the `data.table` rewrite eliminates.