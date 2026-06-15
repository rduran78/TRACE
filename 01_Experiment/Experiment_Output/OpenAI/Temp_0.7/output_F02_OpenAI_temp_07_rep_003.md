 **Diagnosis**  
The current bottlenecks are:  
- `build_neighbor_lookup` creates a massive list (≈6.46M elements) using `lapply`, consuming huge memory.  
- `compute_neighbor_stats` iterates through these lists repeatedly for each variable, causing redundant lookups.  
- Multiple passes over 6.46M rows × 5 variables = enormous overhead.  
- Operations are in R loops, with heavy string concatenation (`paste`) and `lapply`, which are inefficient for large-scale data.  

**Optimization Strategy**  
- Avoid per-row neighbor key computation; precompute neighbor relationships in an integer matrix or compressed form and reuse.  
- Use `data.table` for fast keyed joins and aggregations.  
- Reshape data to wide format by year, compute neighbor aggregations via vectorized operations or matrix indexing.  
- Compute all neighbor stats in a single pass rather than looping over each variable separately.  
- Optional: process in yearly chunks to fit memory.  

**Optimized Approach**  
- Use `data.table` with an edge list (`cell_id → neighbor_id`).  
- Join neighbor values per year, then aggregate (max, min, mean) by source cell and year.  
- Merge results back to the main table.  

**Working R Code**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert rook_neighbors_unique (spdep::nb) into an edge list
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand edges across years to match panel
years <- sort(unique(cell_data$year))
edges_expanded <- edges[, .(id = from, neighbor_id = to), by = .EACHI][rep(seq_len(.N), each = length(years))]
edges_expanded[, year := rep(years, times = nrow(edges))]

# Merge neighbor values
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

dt_neighbors <- merge(
  edges_expanded,
  cell_data[, c("id", "year", neighbor_vars), with = FALSE],
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  allow.cartesian = TRUE
)

# Compute stats per id-year for all variables
agg_list <- lapply(neighbor_vars, function(v) {
  list(max = as.name(v), min = as.name(v), mean = as.name(v))
})

agg_expr <- unlist(lapply(neighbor_vars, function(v) {
  list(
    (function(x) max(x, na.rm = TRUE))[[1]],
    (function(x) min(x, na.rm = TRUE))[[1]],
    (function(x) mean(x, na.rm = TRUE))[[1]]
  )
}), recursive = FALSE)

names(agg_expr) <- unlist(lapply(neighbor_vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))

neighbor_stats <- dt_neighbors[, agg_expr, by = .(id, year)]

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)

# Set NAs for rows without neighbors
```

**Key Advantages**  
- No huge lists in memory; uses efficient joins and aggregation.  
- Single pass computes all neighbor statistics, reducing runtime drastically.  
- Scales better and can be chunked by `year` if memory is tight (process each year separately and rbind).  

This approach should reduce runtime from ~86 hours to a few hours or less on a 16 GB machine.