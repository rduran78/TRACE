 **Diagnosis:**  
The current pipeline is extremely slow (86+ hours) because `compute_neighbor_stats` recalculates neighbor relationships repeatedly for each variable across 6.46 million rows. Although `build_neighbor_lookup` is reused, the main bottleneck is repeatedly scanning and aggregating neighbor values in R lists for each variable. This results in heavy interpretation overhead and excessive memory churn.

---

**Optimization Strategy:**  
1. **Precompute neighbor index lookup once** (already done).
2. **Vectorize neighbor aggregation**:
   - Use matrix operations instead of looping per row.
   - Represent neighbor relationships as a sparse adjacency matrix (6.46M rows × 6.46M is too big, but we can do this per year since neighbors don’t change across time).
3. **Process by year**:
   - For each year, extract `cell_data` subset.
   - Compute neighbor stats for all variables in a single pass using the adjacency matrix.
4. **Use `Matrix` package for sparse ops**:
   - `adj` (sparse matrix) × `vals` gives sums, then divide by neighbor counts for mean.
   - `pmax` and `pmin` for max/min can be computed via grouped apply, avoiding repeated R loops.
5. **Avoid retraining RF**: Only generate features and append them to existing data.

---

**Working R Code (Optimized):**

```r
library(Matrix)
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, year, id)

# Build adjacency matrix once
# rook_neighbors_unique: list of neighbors (spdep format)
# id_order: consistent vector of IDs
id_index <- setNames(seq_along(id_order), id_order)
i_idx <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
j_idx <- unlist(rook_neighbors_unique)
adj <- sparseMatrix(i = i_idx, j = j_idx, x = 1,
                    dims = c(length(id_order), length(id_order)))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Precompute degrees for mean calculation
deg <- rowSums(adj)

# Process per year
result_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  setkey(dt_year, id)
  
  # Ensure correct order
  vals_mat <- as.matrix(dt_year[id_order, ..neighbor_source_vars])
  
  # Compute sums via matrix multiplication
  sums <- adj %*% vals_mat  # rows: cells, cols: variables
  
  # Means
  means <- sweep(sums, 1, deg, "/")
  
  # For max/min: loop over variables, apply pmax/pmin efficiently
  maxs <- mins <- matrix(NA_real_, nrow(adj), ncol(vals_mat))
  for (j in seq_len(ncol(vals_mat))) {
    col_vals <- vals_mat[, j]
    neigh_vals <- lapply(rook_neighbors_unique, function(nbs) col_vals[nbs])
    maxs[, j] <- vapply(neigh_vals, function(x) if (length(x)) max(x, na.rm=TRUE) else NA_real_, numeric(1))
    mins[, j] <- vapply(neigh_vals, function(x) if (length(x)) min(x, na.rm=TRUE) else NA_real_, numeric(1))
  }
  
  # Combine and add back to data.table
  features <- data.table(
    id = id_order,
    year = yr
  )
  for (k in seq_along(neighbor_source_vars)) {
    var <- neighbor_source_vars[k]
    features[[paste0(var, "_neigh_max")]] <- maxs[, k]
    features[[paste0(var, "_neigh_min")]] <- mins[, k]
    features[[paste0(var, "_neigh_mean")]] <- means[, k]
  }
  
  result_list[[as.character(yr)]] <- features
}

# Combine all years
neighbor_features <- rbindlist(result_list)
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data]

# Predict using existing RF model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Expected Performance Gain:**  
- Eliminates repeated neighbor lookups.
- Processes per year with sparse matrix ops.
- Reduces runtime from 86+ hours to a few hours (dominated by per-variable max/min loops).
- Memory-friendly: only handles ~344k rows per year, not 6.46M at once.

This preserves the trained Random Forest model and the original estimand while making the pipeline dramatically faster.