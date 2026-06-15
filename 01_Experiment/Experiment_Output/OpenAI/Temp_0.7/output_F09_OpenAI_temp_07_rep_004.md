 **Diagnosis:**  
The current pipeline repeatedly computes neighbor relationships for every variable across 6.46 million rows, causing redundant operations and high memory overhead. The `build_neighbor_lookup` function is efficient, but computing neighbor stats inside the loop for each variable involves multiple full scans of the dataset. This results in excessive time complexity (~O(n × v × m), where n = rows, v = variables, m = neighbor count). Given 86+ hours runtime, the bottleneck is repeated neighbor processing.

---

**Optimization Strategy:**  
- **Precompute and reuse adjacency:** Build the neighbor index table once (`neighbor_lookup`), then compute all neighbor statistics in one pass per variable using vectorization or parallelization.
- **Use `data.table` for efficient joins and column updates.**
- **Avoid repeated `lapply` for each variable:** Instead, compute stats for all variables in a single loop with minimal overhead.
- Ensure memory efficiency by working in chunks if needed (but likely feasible with 16 GB RAM using `data.table`).
- Preserve the trained Random Forest model and estimand by maintaining original variable names and structure.

---

**Working R Code (Optimized):**

```r
library(data.table)
library(parallel)

# Convert to data.table for efficiency
setDT(cell_data)

# Build reusable neighbor lookup (same as before)
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  row_ids <- seq_len(nrow(data))

  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}

# Precompute neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Compute stats for all variables using parallel processing
compute_neighbor_stats_all <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  names(vals_list) <- vars

  # Parallel over rows
  mclapply(seq_along(neighbor_lookup), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(rep(NA_real_, 3 * length(vars)))
    out <- numeric(3 * length(vars))
    k <- 1
    for (v in vars) {
      neighbor_vals <- vals_list[[v]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        out[k:(k+2)] <- NA_real_
      } else {
        out[k:(k+2)] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
      k <- k + 3
    }
    out
  }, mc.cores = detectCores() - 1)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_list <- compute_neighbor_stats_all(cell_data, neighbor_lookup, neighbor_source_vars)
stats_mat <- do.call(rbind, stats_list)

# Add computed features back to data.table
col_names <- unlist(lapply(neighbor_source_vars, function(v) paste0(v, "_nbr_", c("max", "min", "mean"))))
cell_data[, (col_names) := as.data.table(stats_mat)]

# At this point, cell_data has all neighbor features added
# Proceed with Random Forest prediction using the trained model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Expected Impact:**  
- Adjacency lookup computed once.
- All neighbor features computed in a single parallelized pass.
- Eliminates repeated scans; runtime should reduce from 86+ hours to a few hours (depending on cores).
- Preserves original estimand and trained RF model.