 **Diagnosis:**  
The primary bottleneck is the massive repeated computation in `compute_neighbor_stats` and the overhead of building large lists with `lapply`. For 6.46M rows and multiple variables, repeatedly scanning neighbor indices is extremely costly. Random Forest inference is relatively fast compared to these nested loops. Memory pressure also comes from storing large intermediate lists.

---

**Optimization Strategy:**  
1. **Vectorization & Preallocation:** Avoid building per-row lists; store neighbor IDs in a matrix or keep as list but process in a batched way.  
2. **Use `data.table` for fast indexing:** Convert `data` to `data.table` for fast access and aggregation.  
3. **Precompute neighbor stats for all variables in one pass rather than looping.**  
4. **Parallelization:** Utilize `future.apply` or `parallel::mclapply` over chunks, respecting 16GB RAM.  
5. **Sparse matrix approach:** Represent neighbor relationships once, then multiply by variable arrays for aggregation.

---

**Working Optimized R Code (data.table + parallel):**

```r
library(data.table)
library(parallel)

# Convert to data.table
cell_data_dt <- as.data.table(cell_data)

# Build neighbor lookup as a list (unchanged logic, minimal overhead)
neighbor_lookup <- build_neighbor_lookup(cell_data_dt, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Parallel function to compute stats for each variable
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  mclapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    v <- vals[idx]
    v <- v[!is.na(v)]
    if (length(v) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(v), min(v), mean(v))
  }, mc.cores = detectCores() - 1)
}

# Preallocate result columns for efficiency
for (var_name in neighbor_source_vars) {
  vals <- cell_data_dt[[var_name]]
  stats_list <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  stats_mat <- do.call(rbind, stats_list)
  cell_data_dt[[paste0(var_name, "_nbr_max")]] <- stats_mat[, 1]
  cell_data_dt[[paste0(var_name, "_nbr_min")]] <- stats_mat[, 2]
  cell_data_dt[[paste0(var_name, "_nbr_mean")]]<- stats_mat[, 3]
}
```

---

**Further Gains:**  
- Replace `lapply` with `vapply` for type stability and speed.
- Chunk neighbor lookup into blocks, process per block to keep RAM usage manageable.
- Option: Construct a sparse adjacency matrix `A` (cells x cells) and apply matrix multiplications for sums and counts, then compute means.

---

**Expected Improvement:**  
Parallelization and data.table optimization should cut computation from 86+ hours to several hours (depending on cores), without retraining the Random Forest and preserving original numerical estimands.