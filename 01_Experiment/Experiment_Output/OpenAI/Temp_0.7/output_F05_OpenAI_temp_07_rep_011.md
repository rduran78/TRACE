 **Diagnosis:**  
The repeated `paste()` and string-key-based indexing inside `lapply` is a major inefficiency. For every row (~6.46 million), the function constructs `neighbor_keys` and performs lookups in `idx_lookup` using character keys. This is not just a local inefficiency—it is symptomatic of a larger issue: repeated string concatenation and hash lookups dominate runtime. The outer loop further multiplies cost as the same neighbor structure is recomputed for each variable.

**Optimization Strategy:**  
Reformulate the algorithm to avoid string-based indexing entirely. Use integer-based mapping and precompute the neighbor lookup once as a list of integer indices. Then reuse this list for all variables. Steps:  
1. Map `id` to integer positions up front.  
2. Use a single integer join for `(id, year)` instead of concatenated strings.  
3. Build `neighbor_lookup` efficiently without `paste()` calls.  
4. Use vectorized or preallocated structures for neighbor stats.

This reduces repeated work and memory churn. The result: one pass to build neighbor indices, then fast numeric operations for stats.

---

### **Working R Code**

```r
# Precompute mappings for fast integer joins
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Map cell id to reference position
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) -> row index using integer keys
  # Convert years to integer offset
  years <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(years), years)
  
  # Precompute row index matrix for fast lookup
  # row_key[i] = (id_ref[i]-1)*n_years + year_ref[i]
  n_years <- length(years)
  id_ref <- id_to_ref[as.character(data$id)]
  year_ref <- year_to_idx[as.character(data$year)]
  row_key <- (id_ref - 1L) * n_years + year_ref
  
  # Map key -> row index
  idx_lookup <- seq_len(nrow(data))
  names(idx_lookup) <- row_key
  
  # Build neighbor list as integer indices
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_ref[i]
    neighbor_refs <- neighbors[[ref_idx]]
    if (length(neighbor_refs) == 0) return(integer(0))
    neighbor_keys <- (neighbor_refs - 1L) * n_years + year_ref[i]
    as.integer(idx_lookup[as.character(neighbor_keys)])
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      v <- vals[idx]
      v <- v[!is.na(v)]
      if (length(v) > 0) {
        result[i, ] <- c(max(v), min(v), mean(v))
      }
    }
  }
  result
}

# Pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

**Why faster?**  
- Eliminates `paste()` and string-based lookups inside inner loops.
- Integer keys allow O(1) array indexing.
- `neighbor_lookup` computed once and reused.
- Memory footprint reduced (no giant character vectors).

**Expected improvement:**  
From 86+ hours down to a few hours (or less), depending on disk I/O and machine constraints.