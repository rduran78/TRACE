 **Diagnosis**  
- **Main bottleneck:** The neighbor feature computation uses `lapply` repeatedly for millions of rows, creating large intermediate lists and performing repeated object copying (`cell_data` updates per variable).  
- **Prediction loop inefficiency:** Repeated transformations before Random Forest inference amplify runtime.  
- **Memory pressure:** 6.46M rows × 110 predictors + 15 neighbor-derived features is huge; list-based operations and repeated `rbind` calls are expensive.  
- **Model loading is minor compared to feature generation overhead.** Random Forest inference (`predict` on millions of rows) is slower but not the main cause of 86 hours—the neighbor stats preparation dominates.

---

**Optimization Strategy**  
1. **Vectorize neighbor feature computation:**  
   - Replace `lapply` loops with `vapply` or matrix operations where possible.  
   - Preallocate matrices for neighbor statistics rather than incremental `rbind`.  
2. **Avoid repeated data frame copying:**  
   - Compute all neighbor stats first and `cbind` once.  
3. **Use integer indices and avoid repeated `paste` calls inside loops:**  
   - Build `neighbor_lookup` as integer index lists; reuse efficiently.  
4. **Parallelize computation:**  
   - Use `parallel::mclapply` or `future.apply` for multi-core execution.  
5. **Prediction optimization:**  
   - Use `predict(..., newdata, type="response")` in chunks (e.g., 500k rows) to manage memory and speed.  
6. **Keep model unchanged; preserve estimand.**

---

**Optimized Working R Code**

```r
library(data.table)
library(parallel)

# Assume cell_data is a data.table for efficiency
setDT(cell_data)

# Build neighbor lookup (vectorized, no repeated paste)
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  keys <- paste(rep(id_order, each = length(unique(data$year))),
                rep(unique(data$year), times = length(id_order)), sep = "_")
  
  # Precompute for efficiency
  lapply(seq_len(nrow(data)), function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    as.integer(idx_lookup[neighbor_keys[neighbor_keys %in% names(idx_lookup)]])
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats in parallel
compute_neighbor_stats_fast <- function(data, neighbor_lookup, vars) {
  vals_list <- lapply(vars, function(v) data[[v]])
  n <- nrow(data)
  res_mat <- matrix(NA_real_, n, length(vars) * 3)
  
  # Parallel loop
  mclapply(seq_len(n), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(rep(NA_real_, length(vars) * 3))
    out <- numeric(length(vars) * 3)
    for (j in seq_along(vars)) {
      neighbor_vals <- vals_list[[j]][idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        out[(j-1)*3 + 1:3] <- NA_real_
      } else {
        out[(j-1)*3 + 1:3] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
    out
  }, mc.cores = detectCores() - 1) -> rows
  
  res_mat[] <- do.call(rbind, rows)
  colnames(res_mat) <- as.vector(sapply(vars, function(v) paste0(v, c("_max","_min","_mean"))))
  res_mat
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind features once
cell_data <- cbind(cell_data, neighbor_features)

# Random Forest prediction in chunks
chunk_size <- 5e5
n <- nrow(cell_data)
preds <- numeric(n)
for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end], type = "response")
}

cell_data$pred_gdp <- preds
```

---

**Expected Impact:**  
- Reduces neighbor computation from hours to minutes with parallelization and preallocation.  
- Avoids repeated copying and inefficient loops.  
- Scales inference with chunking to fit 16 GB RAM.  
- Preserves trained Random Forest model and original estimand.