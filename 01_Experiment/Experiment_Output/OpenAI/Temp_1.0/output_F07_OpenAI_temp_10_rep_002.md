 **Diagnosis**  
The bottleneck is the `compute_neighbor_stats` loop (and `compute_and_add_neighbor_features`) operating in pure R over ~6.46M rows and millions of neighbor lookups. The nested `lapply` for every variable introduces redundant passes over the same neighbor structure and incurs huge R-level overhead. Building repeated vectors and row-bind calls on such scale explodes memory/time cost.  

Main inefficiencies:  
- Many small lists → repeated allocations.  
- Looping through `neighbor_source_vars` and recomputing stats each time.  
- Not leveraging vectorized or compiled operations (C++/data.table).  

---

**Optimization Strategy**  
1. Compute all neighbor stats in **one pass** rather than separately per variable.  
2. Use `data.table` for fast column access and binding.  
3. Pre-build neighbor index matrix for efficient access.  
4. Use `vapply` or vectorized methods over lists to reduce interpreter overhead.  
5. Optionally go to C++ via `Rcpp` for maximal speed if still too slow.  

Design goals:  
- Keep exact rook-neighbor relationships and same numerical estimand.  
- Avoid retraining model (so preserve feature names and pipeline compatibility).  
- Memory-aware: avoid creating huge intermediate lists repeatedly.  

---

**Optimized Implementation (R with data.table)**  

```r
library(data.table)

# Assume cell_data is already a data.table
setDT(cell_data)

# Build neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Convert neighbor_lookup into a list of integer indices (already returns that)

# Vectorized computation for multiple variables in one pass:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  # Preallocate a matrix: nrow = nrow(data), ncol = length(vars) * 3
  n <- nrow(data)
  k <- length(vars)
  res_mat <- matrix(NA_real_, n, k * 3)
  colnames(res_mat) <- as.vector(t(outer(vars, c("_max", "_min", "_mean"), paste0)))
  
  # For faster access, convert to list of numeric vectors
  var_lists <- lapply(vars, function(v) data[[v]])
  
  # Iterate rows (fast lapply, but do numeric operations in Rcpp for real speed)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    for (j in seq_along(vars)) {
      vals <- var_lists[[j]][idx]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0) {
        res_mat[i, (j - 1) * 3 + 1] <- max(vals)
        res_mat[i, (j - 1) * 3 + 2] <- min(vals)
        res_mat[i, (j - 1) * 3 + 3] <- mean(vals)
      }
    }
  }
  
  res_mat
}

# Apply and cbind efficiently
res <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)
cell_data <- cbind(cell_data, as.data.table(res))
```

---

**Further Speed Gains**  
- Replace inner loops with an `Rcpp` function operating on integer neighbor lists and numeric vectors (orders of magnitude faster).  
- If memory allows, store neighbor_lookup as an integer matrix (ragged list flattened with offsets).  
- Consider parallelization with `future.apply` or `parallel::mclapply` if CPU cores available.  

**Expected Improvement**: Processing 6.46M rows with combined vectorization and compiled code should drop runtime from 86+ hours to well under 1–2 hours on a 16GB laptop.  

This preserves the trained Random Forest model, keeps identical estimand, and ensures integrity of rook-neighbor topology.