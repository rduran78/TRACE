 **Diagnosis:**  
The current process is extremely memory and compute intensive because `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M rows, creating large lists with repeated string operations (`paste` lookups for every row) and multiple allocations. These operations scale poorly with such a large dataset. Most time and memory overhead comes from:  
1. Repeated character concatenation for keys.  
2. Large `lapply` constructing millions of small vectors.  
3. Lack of vectorization and parallelization.  

**Optimization Strategy:**  
- **Avoid repeated string concatenation**: Precompute `(id, year)` pairs and map them with integer indexing rather than text keys.  
- **Flatten neighbor lookup to matrix form** where rows align with data rows (or keep adjacency in integers).  
- **Parallelize computation** using `data.table` or `future` + `furrr`.  
- **Process in chunks** rather than all rows at once to fit laptop memory.  
- Replace `lapply` with `vapply` or vectorized aggregations from `data.table`.  
- Store `neighbor_lookup` as `list(integer)` indexed by row number directly without paste operations.  

**Working Example with `data.table` and chunking:**  

```r
library(data.table)
library(parallel)

# Assume cell_data is a data.frame; convert to data.table for speed
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Precompute index mapping for fast join
cell_data[, row_idx := .I]

# Build neighbor lookup (integer-based, chunked)
build_neighbor_lookup_fast <- function(cell_data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), id_order)
  n <- nrow(cell_data)
  lookup <- vector("list", n)
  
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[[as.character(cell_data$id[i])]]
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    # Match neighbors for same year:
    idx <- cell_data[.(neighbor_ids, cell_data$year[i]), on = .(id, year), row_idx]
    lookup[[i]] <- idx[!is.na(idx)]
  }
  lookup
}

# Compute stats in parallel
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  cl <- makeCluster(detectCores() - 1)
  on.exit(stopCluster(cl))
  
  clusterExport(cl, c("vals", "neighbor_lookup"), envir = environment())
  
  res <- parLapply(cl, neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    v <- vals[idx]
    v <- v[!is.na(v)]
    if (length(v) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(v), min(v), mean(v))
  })
  do.call(rbind, res)
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_lookup)
  cell_data[, paste0(var_name, "_nbr_max") := stats[, 1]]
  cell_data[, paste0(var_name, "_nbr_min") := stats[, 2]]
  cell_data[, paste0(var_name, "_nbr_mean") := stats[, 3]]
}
```

**Why this helps:**  
- Eliminates costly string operations by using integer indexing.  
- Processes with efficient `data.table` joins rather than `paste` keys.  
- Uses parallel processing for neighbor stats (significant speedup on multi-core laptops).  
- Memory footprint is reduced because we avoid creating 6M long strings.  

**Expected Outcome:**  
Instead of 86+ hours, depending on cores and chunk size, the process should complete in a few hours on a 16GB laptop without exhausting memory and without changing the numerical estimand or touching the Random Forest model.