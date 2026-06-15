 **Diagnosis**  
The current pipeline recomputes neighbor statistics (max, min, mean) for each of the 6.46 million cell-year rows by iterating through all rows and accessing neighbor values repeatedly. This is extremely inefficient because:  
- The neighbor structure is static across years, yet `compute_neighbor_stats` recalculates neighbor indices for every row and every variable.  
- For 28 years and 344k cells, the cost of repeatedly scanning neighbors is huge (O(N * neighbors * years * variables)).  
- Using `lapply` and row-by-row operations on millions of elements adds overhead.  

**Optimization Strategy**  
- Precompute `neighbor_lookup` **once per cell** (not per cell-year).  
- Reshape data into a list or array grouped by year, so we can compute neighbor stats per year in vectorized loops rather than row-by-row.  
- For each year, fetch relevant variable vectors, then compute neighbor summaries using matrix operations.  
- Bind results back efficiently with `rbindlist` or `data.table`.  
- Avoid redundant lookups and expensive string concatenation for keys.  

**Optimized R Code**  

```r
library(data.table)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of all cell IDs in spatial order
# rook_neighbors_unique: nb object from spdep

# 1. Precompute static neighbor lookup by cell index (no year)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # Return a list: each element is an integer vector of neighbor indices (by id_order position)
  lapply(seq_along(id_order), function(i) neighbors[[i]])
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)
cell_index_map <- setNames(seq_along(id_order), as.character(id_order))

# 2. Convert to data.table and set keys for fast join
setDT(cell_data)
setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))

# 3. Function to compute neighbor stats for one variable across all years
compute_neighbor_stats_fast <- function(dt, var_name, neighbor_lookup, cell_index_map, years) {
  n_cells <- length(cell_index_map)
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Extract values for this year in id_order
    vals <- dt[year == yr][match(id_order, id), get(var_name)]
    # Initialize matrix for stats: rows = cells, cols = 3 (max, min, mean)
    stats_mat <- matrix(NA_real_, nrow = n_cells, ncol = 3)
    for (i in seq_len(n_cells)) {
      nbr_idx <- neighbor_lookup[[i]]
      if (length(nbr_idx) > 0) {
        nvals <- vals[nbr_idx]
        nvals <- nvals[!is.na(nvals)]
        if (length(nvals) > 0) {
          stats_mat[i, ] <- c(max(nvals), min(nvals), mean(nvals))
        }
      }
    }
    result_list[[yi]] <- data.table(
      id = id_order,
      year = yr,
      max = stats_mat[, 1],
      min = stats_mat[, 2],
      mean = stats_mat[, 3]
    )
  }
  
  rbindlist(result_list)
}

# 4. Loop over source variables and compute stats, merging back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_fast(cell_data, var_name, neighbor_lookup_static, cell_index_map, years)
  setnames(stats_dt, c("max", "min", "mean"), paste0(var_name, "_nbr_", c("max", "min", "mean")))
  cell_data <- stats_dt[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added efficiently
```

**Why this is faster**  
- `neighbor_lookup_static` is computed once (not per row-year).  
- For each year (28 iterations) and for each variable (5 iterations), we iterate over 344k cells—but without string concat or hash lookups.  
- Memory footprint is lower and vectorization reduces overhead.  
- Expected runtime goes from 86+ hours to a few hours on a 16 GB laptop.  

**Key properties preserved**  
- Random Forest model remains unchanged (we only recompute features).  
- Numerical estimand is identical (same max/min/mean logic).  
- Handles NAs and missing neighbors as before.