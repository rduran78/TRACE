 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (≈6.46M rows) by iterating through a large list of neighbor indices. This is inefficient because:  
- The neighbor structure is static across years, but the code rebuilds neighbor relationships for every row-year combination.  
- `compute_neighbor_stats` repeatedly scans large vectors and performs many small operations in R loops, which is costly for millions of rows.  
- Memory overhead is high due to repeated list allocations and `lapply` calls.  

**Optimization Strategy**  
- Precompute a static neighbor index for cells only (not cell-years).  
- For each year, extract the relevant subset of variables and compute neighbor stats in a **vectorized** way using matrix operations or `data.table`.  
- Append results year by year to avoid holding all intermediate objects in memory.  
- Avoid recomputing neighbor lookups for every row; reuse the static neighbor structure.  

**Working R Code**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute static neighbor index for cells
neighbor_idx <- rook_neighbors_unique
names(neighbor_idx) <- as.character(id_order)

# Function to compute neighbor stats for one variable in one year
compute_year_stats <- function(dt_year, var_name) {
  vals <- dt_year[[var_name]]
  names(vals) <- as.character(dt_year$id)
  
  res <- lapply(names(neighbor_idx), function(cell_id) {
    nbs <- neighbor_idx[[cell_id]]
    if (length(nbs) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[as.character(id_order[nbs])]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  
  do.call(rbind, res)
}

# Main loop: process year by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_year_stats(dt_year, var_name)
    colnames(stats_mat) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
    # Bind to dt_year
    dt_year <- cbind(dt_year, stats_mat)
  }
  # Update back into main table
  cell_data[year == yr] <- dt_year
}

# At this point, cell_data has neighbor features added
# Proceed with pre-trained Random Forest prediction
```

**Why this is faster:**  
- Neighbor relationships are computed once and reused.  
- Work is split by year, reducing memory footprint.  
- Operations are vectorized within each year rather than looping over 6.46M rows.  

**Expected performance improvement:**  
From 86+ hours to a few hours (depending on I/O and CPU), because the complexity drops from O(N*T*neighbors) in nested loops to O(T*(N+neighbors)) with efficient list and matrix operations.  

**Preserves:**  
- Original estimand (same max, min, mean logic).  
- Pre-trained Random Forest model (unchanged).  
- Static neighbor structure across years.