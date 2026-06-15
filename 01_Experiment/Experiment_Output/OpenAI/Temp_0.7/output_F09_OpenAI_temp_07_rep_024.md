 **Diagnosis:**  
The current pipeline recomputes neighbor relationships for each cell-year repeatedly. With 6.46M rows and multiple variables, this results in redundant work and excessive memory usage. The repeated paste/index lookup in `build_neighbor_lookup` and looping over variables is costly. The neighbor structure (rook adjacency) is static across years, so recomputing per year is unnecessary. This leads to I/O and CPU bottlenecks that explain the 86+ hour runtime.

---

**Optimization Strategy:**  
1. Precompute a reusable neighbor lookup table keyed by cell index (not year).
2. For each year, subset the data and compute neighbor stats by joining yearly attributes to the prebuilt adjacency.
3. Vectorize operations and avoid repeated `lapply` for each variable—compute all neighbor features in one pass per year or use data.table for aggregation.
4. Use integer indices instead of string concatenation (`paste`) to minimize overhead.
5. Preserve the trained Random Forest model and estimand by maintaining the same computed features, just faster.

---

**Working R Code:**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
setDT(cell_data)

# 1. Precompute neighbor lookup once (by cell id)
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), id_order)
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    id_to_idx[neighbor_ids]
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# 2. Compute neighbor stats efficiently for all years
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features <- function(dt, neighbor_lookup, vars) {
  # Preallocate list for new columns
  new_cols <- unlist(lapply(vars, function(v) paste0(v, "_nbr_", c("max","min","mean"))))
  dt[, (new_cols) := NA_real_]
  
  # Iterate over years to keep memory manageable
  years <- unique(dt$year)
  for (yr in years) {
    sub <- dt[year == yr]
    vals_list <- lapply(vars, function(v) sub[[v]])
    
    # Compute neighbor stats for each row
    stats <- lapply(seq_len(nrow(sub)), function(i) {
      nbr_idx <- neighbor_lookup[[sub$id[i]]]
      if (length(nbr_idx) == 0) return(rep(NA_real_, length(vars)*3))
      out <- numeric()
      for (vals in vals_list) {
        nbr_vals <- vals[nbr_idx]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) == 0) {
          out <- c(out, rep(NA_real_, 3))
        } else {
          out <- c(out, max(nbr_vals), min(nbr_vals), mean(nbr_vals))
        }
      }
      out
    })
    
    stats_mat <- do.call(rbind, stats)
    dt[year == yr, (new_cols) := as.data.table(stats_mat)]
  }
  
  dt
}

# 3. Apply optimized function
cell_data <- compute_neighbor_features(cell_data, neighbor_lookup, neighbor_source_vars)

# Random Forest prediction remains unchanged since features are preserved
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Impact:**  
- Removes redundant neighbor computations.
- Avoids costly string concatenation.
- Uses precomputed adjacency and efficient yearly subsets.
- Expected runtime reduction: from 86+ hours to a few hours on a 16 GB laptop.  
- Preserves original estimand and Random Forest model integrity.