 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46 million rows, creating millions of small vectors and repeatedly allocating memory.  
- Neighbor lookups are recomputed for each variable, causing redundant work.  
- `do.call(rbind, ...)` on millions of elements is inefficient.  
- No vectorization or matrix-based operations; everything is handled in R loops.  

**Optimization Strategy**  
- Precompute `neighbor_lookup` once and store as an integer list.  
- Use `vapply` or matrix preallocation instead of `lapply` + `rbind`.  
- Compute all neighbor statistics in a single pass per variable using efficient aggregation.  
- Consider `data.table` for fast row access and assignment.  
- Avoid repeated `paste` operations; use integer indexing instead.  
- Use parallel processing if possible (`parallel::mclapply` or `future.apply`).  

**Working R Code** (optimized, memory-safe, preserves estimand and trained model):  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Convert neighbor_lookup to a list of integer vectors
# Already done by build_neighbor_lookup

compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(result) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Apply for all variables efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats_mat <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  
  # Add columns to cell_data
  cell_data[[paste0(var_name, "_nbr_max")]]  <- stats_mat[, 1]
  cell_data[[paste0(var_name, "_nbr_min")]]  <- stats_mat[, 2]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats_mat[, 3]
}
```

**Additional Enhancements**  
- Use `parallel::mclapply` for the loop inside `compute_neighbor_stats_fast` on multicore machines.  
- If memory allows, chunk processing by year to reduce working set size.  
- Avoid repeated neighbor computations across variables by reusing `neighbor_lookup`.  

**Expected Impact**  
- Eliminates repeated work, reduces overhead from millions of small objects.  
- Moves from 86+ hours to a few hours or less on a standard laptop.  
- Preserves original rook-neighbor relationships and numerical estimand.  
- Keeps the trained Random Forest model intact.