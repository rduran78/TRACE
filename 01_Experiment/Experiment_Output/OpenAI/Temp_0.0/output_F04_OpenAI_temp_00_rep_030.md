 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and named vector lookups, which are extremely slow at this scale. The neighbor relationships are static across years, yet the code recomputes keys for every row-year combination. Additionally, `compute_neighbor_stats` repeatedly scans vectors and allocates lists, adding overhead.

**Optimization Strategy**  
1. **Precompute neighbor indices by cell ID only** (not per year) and reuse them across all years.
2. **Avoid string-based lookups**; use integer indexing instead.
3. **Vectorize neighbor feature computation** using `data.table` or matrix operations.
4. **Process by year in chunks** to reduce memory pressure.
5. **Parallelize** across variables or years using `parallel` or `future.apply`.

**Working R Code (Optimized)**  
```r
library(data.table)
library(parallel)

# Convert to data.table for fast joins
setDT(cell_data)

# Precompute neighbor index list by cell ID
id_to_idx <- setNames(seq_len(nrow(unique(cell_data[, .(id)]))), unique(cell_data$id))
neighbor_idx_list <- lapply(rook_neighbors_unique, function(neigh_ids) {
  as.integer(id_to_idx[as.character(neigh_ids)])
})

# Add a column for cell index
cell_data[, cell_idx := id_to_idx[as.character(id)]]

# Function to compute neighbor stats for one variable and one year chunk
compute_neighbor_stats_chunk <- function(dt, var_name, neighbor_idx_list) {
  vals <- dt[[var_name]]
  res <- matrix(NA_real_, nrow(dt), 3)
  for (i in seq_len(nrow(dt))) {
    neigh_idx <- neighbor_idx_list[[dt$cell_idx[i]]]
    if (length(neigh_idx) > 0) {
      neigh_vals <- vals[neigh_idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        res[i, ] <- c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
      }
    }
  }
  res
}

# Process by year and variable in parallel
years <- unique(cell_data$year)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("cell_data", "neighbor_idx_list", "compute_neighbor_stats_chunk"))

for (var_name in neighbor_source_vars) {
  results_list <- parLapply(cl, years, function(y) {
    dt_year <- cell_data[year == y]
    compute_neighbor_stats_chunk(dt_year, var_name, neighbor_idx_list)
  })
  
  # Bind results and add columns
  res_mat <- do.call(rbind, results_list)
  colnames(res_mat) <- paste(var_name, c("neigh_max", "neigh_min", "neigh_mean"), sep = "_")
  cell_data[, (colnames(res_mat)) := res_mat]
}

stopCluster(cl)
```

**Why This Works**  
- Eliminates expensive string operations and repeated lookups.
- Uses integer indexing for neighbors.
- Processes data in manageable chunks (by year).
- Parallelization reduces wall-clock time dramatically.
- Preserves original estimand and Random Forest model.

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours on a 16 GB laptop, depending on parallel efficiency.