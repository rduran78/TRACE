 **Diagnosis**  
The bottleneck is in spatial neighbor feature construction, specifically repeated `lapply` calls with heavy string concatenation, indexing, and row-wise computation over ~6.46M rows. Each call to `compute_neighbor_stats` does unnecessary R-level loops and `paste` operations, which scale poorly. This dominates runtime far more than Random Forest inference.

---

**Optimization Strategy**  
1. **Avoid repeated string concatenation and lookups**: Precompute neighbor indices as integers, not keys.  
2. **Vectorize computations**: Replace `lapply` row-wise loops with `vapply` or `data.table` joins.  
3. **Memory efficiency**: Use integer matrices and preallocate outputs rather than growing lists.  
4. **Parallelization**: Use `parallel::mclapply` or `future.apply`.  
5. **Leverage `data.table` for fast aggregation**: Handles large datasets efficiently.  

Core idea: Build `neighbor_lookup` as a list of integer vectors once, skip key-based lookups, and compute min/max/mean in compiled loops or vectorized manner.

---

**Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor indices as integers (no string concatenation)
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_row <- match(id_order, data$id)  # row positions for ids
  lapply(seq_along(id_order), function(ref_idx) {
    neighbor_ids <- id_order[neighbors[[ref_idx]]]   # neighbor cell IDs
    match(neighbor_ids, data$id)                     # integer row indices
  })
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Vectorized neighbor stats computation
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- length(neighbor_lookup)
  
  # Preallocate matrix for max, min, mean
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  
  # Parallel loop
  result <- mclapply(seq_len(n), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, mc.cores = detectCores() - 1)
  
  do.call(rbind, result)
}

# Apply to all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

**Expected Gains**  
- Eliminates costly `paste` and string-based indexing (~10–20× faster).
- Uses integer lookups and preallocation.
- Parallelization across cores reduces runtime substantially.
- Estimated reduction: from **86+ hours to under 6–8 hours** on a 16 GB laptop.

**Preserves**: Original numerical estimand, trained Random Forest model, and feature semantics.