 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, repeatedly performing vector lookups and allocations.  
- Neighbor stats are computed row-by-row instead of in a vectorized or grouped manner.  
- No parallelization or efficient data structures (e.g., `data.table`) are used.  
- Memory overhead from repeated list operations is high.  

**Optimization Strategy**  
- Precompute neighbor relationships once and store as integer indices.  
- Use `data.table` for fast joins and grouping.  
- Compute neighbor stats in a vectorized way by reshaping data into wide format by year, then aggregating by neighbor indices.  
- Optionally parallelize across years to reduce memory pressure.  
- Avoid raster focal analogy unless raster alignment is guaranteed; here, the panel structure and precomputed rook neighbors make a graph-based approach better.  

**Working R Code**  

```r
library(data.table)
library(parallel)

# Convert to data.table
setDT(cell_data)

# Ensure keys for fast joins
setkey(cell_data, id, year)

# Precompute neighbor pairs (directed)
# rook_neighbors_unique: list of integer vectors (spdep nb)
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Expand neighbor pairs across all years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[, .(id = rep(src, each = length(years)),
                                   nbr = rep(nbr, each = length(years)),
                                   year = rep(years, times = .N))]

# Join neighbor values for each variable
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  tmp <- merge(neighbor_dt,
               cell_data[, .(nbr = id, year, val = get(var_name))],
               by = c("nbr", "year"),
               all.x = TRUE)
  
  # Compute stats by (id, year)
  tmp_stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Rename columns
  setnames(tmp_stats, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var_name, "_nbr_max"),
             paste0(var_name, "_nbr_min"),
             paste0(var_name, "_nbr_mean")))
  
  tmp_stats
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Parallelize across variables
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("neighbor_dt", "cell_data", "compute_neighbor_stats_dt"))
clusterEvalQ(cl, library(data.table))

stats_list <- parLapply(cl, neighbor_source_vars, compute_neighbor_stats_dt)
stopCluster(cl)

# Merge all stats back into cell_data
for (stats_dt in stats_list) {
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor features added
```

**Why this works:**  
- Uses `data.table` for efficient joins and aggregation.  
- Computes neighbor stats in bulk rather than row-by-row.  
- Parallelizes across variables to utilize multiple cores.  
- Preserves original estimand and Random Forest model (no retraining).  

**Expected performance:**  
- From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk speed and CPU cores.  
- Memory footprint is manageable because operations are grouped and vectorized.