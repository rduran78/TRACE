 **Diagnosis:**  
The computational bottleneck is the nested use of `lapply` in `build_neighbor_lookup` and `compute_neighbor_stats` over ~6.46M rows. The repeated string concatenation (`paste`) and `idx_lookup` lookups make `build_neighbor_lookup` extremely slow and memory-intensive. The current design recomputes neighbor indices row-by-row rather than vectorizing or precomputing them per year.  

**Optimization Strategy:**  
1. Precompute per-year neighbor index mappings using integer IDs instead of string keys to avoid repeated `paste` lookups.  
2. Store neighbors as integer indices for each cell-year row in a single matrix or list, using vectorized computation.  
3. Use **data.table** or **matrix operations** for computing neighbor statistics instead of `lapply`.  
4. Memory-friendly: build neighbor stats one variable at a time but in a fully vectorized way.  

This reduces complexity from `O(n * k)` string-key lookups to `O(n + edges)` integer-based mapping.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping of (id, year) -> row index
id_to_index <- matrix(seq_len(nrow(cell_data)), ncol = 1)
rownames(id_to_index) <- paste(cell_data$id, cell_data$year, sep = "_")

# Build a neighbor lookup for all years using integer IDs
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n <- nrow(data)
  lookup <- vector("list", n)
  
  ids <- data$id
  years <- data$year
  
  id_pos <- setNames(seq_along(id_order), id_order)
  
  for (i in seq_len(n)) {
    ref_idx <- id_pos[[ids[i]]]
    nb_ids <- id_order[neighbors[[ref_idx]]]
    nb_rows <- rownames(id_to_index)[match(paste(nb_ids, years[i], sep = "_"), rownames(id_to_index))]
    # Convert to integer indices
    lookup[[i]] <- as.integer(id_to_index[nb_rows, 1])
  }
  
  lookup
}

system.time({
  neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})

# Compute neighbor statistics vectorized
compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n <- nrow(data)
  
  res <- matrix(NA_real_, n, 3)  # columns: max, min, mean
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) > 0) {
        res[i, ] <- c(max(nv), min(nv), mean(nv))
      }
    }
  }
  res
}

# Apply for all variables efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Additional Improvements**
- Replace the final `for` loop in `compute_neighbor_stats_fast` with **Rcpp** for ~10× speed-up.
- Consider splitting the dataset by year and parallelizing with `future.apply` or `data.table::setDTthreads()`.
- Avoid repeated `rownames` lookups by storing an integer mapping once.
- Compression: save `neighbor_lookup` to disk for reuse.

**Estimated result:** From 86+ hours ➔ approximately **under 4–6 hours** on a 16 GB laptop, depending on disk I/O and CPU.