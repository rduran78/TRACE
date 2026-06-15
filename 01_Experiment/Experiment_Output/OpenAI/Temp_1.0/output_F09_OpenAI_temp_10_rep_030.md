 **Diagnosis:**  
The current pipeline recomputes neighbor relationships for every iteration of the outer loop (`compute_and_add_neighbor_features`), which is costly given 6.46M rows × 5 variables. This repeated overhead drastically inflates runtime. Furthermore, much work is redundant across years because the neighbor topology is static — only attribute values vary yearly. Building neighbor index lookups once, then applying stats calculation for each variable, is the optimal approach. Memory-wise, 16 GB is tight but feasible if we avoid constructing large intermediate duplicated objects.

---

**Optimization Strategy:**  
- Compute a reusable `neighbor_lookup` *once* across all cell-years.  
- Each element caches indices for a cell-year’s neighbors in the data frame.  
- For each variable, vectorized or batched computation applies neighbor functions without repeated recomputation.  
- Process in chunks (optional) to avoid RAM overflow.  
- Preserve model and estimand by only changing feature computation backend, not modeling steps.  

---

**Working R Code:**

```r
# Build neighbor lookup ONCE
build_neighbor_lookup <- function(data, id_order, neighbors) {
  # Map cell IDs to positional reference
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Lookup: "id_year" -> row index
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  # Precompute rows
  row_ids <- seq_len(nrow(data))
  
  # Build lookup list: for each row, find indices of neighbors in same year
  lapply(row_ids, function(i) {
    ref_idx           <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys     <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result            <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}

# Compute per-variable neighbor stats
compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- lapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}

# Driver: Single lookup, multiple vars
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}

# Save cell_data and run Random Forest predict
# rf_model <- readRDS("trained_rf_model.rds")
# preds <- predict(rf_model, newdata = cell_data)
```

---

**Expected Outcome:**  
- Eliminates repeated neighbor mapping for each variable → major speed up.  
- Lookup built once for 6.46M rows; only 5 matrix calculations after.  
- Runtime likely drops from ~86 hours to a few hours (I/O bound) depending on disk and RAM.  
- No retraining; estimands preserved.  
- Expand with parallel `mclapply` if needed for extra gains under your RAM constraints.