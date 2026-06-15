 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` repeatedly constructs neighbor indices by string concatenation and hash lookups. This is done once, but then `compute_neighbor_stats` iterates over all rows and repeatedly subsets vectors, causing heavy R-level overhead. The complexity is roughly O(N × avg_neighbors) with millions of small list operations in R, which is inefficient.  

**Optimization Strategy**  
- Precompute a reusable adjacency table (cell_id → neighbor_cell_ids) once, independent of year.  
- For each year, join the year-specific attributes to this adjacency table and compute neighbor statistics using **vectorized data.table operations** instead of per-row loops.  
- Avoid repeated string concatenation and list indexing.  
- Use `data.table` for fast joins and aggregations.  
- Keep memory usage manageable by processing one year at a time and writing results back.  

**Working R Code**  

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# 1. Build adjacency table once
build_adjacency_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]), use.names = FALSE)
  data.table(id = from, neighbor_id = to)
}

adjacency_dt <- build_adjacency_table(id_order, rook_neighbors_unique)
setkey(adjacency_dt, id)

# 2. Compute neighbor stats year by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is a data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Prepare output container
result_list <- vector("list", length = length(unique(cell_data$year)))

years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset data for this year
  year_dt <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_dt, id)
  
  # Join adjacency with year data for neighbors
  neighbor_dt <- adjacency_dt[year_dt, on = .(neighbor_id = id)]
  # neighbor_dt now has: id (focal), neighbor_id, ntl, ec, ...
  
  # Compute stats for each var
  stats_list <- lapply(neighbor_source_vars, function(var) {
    neighbor_dt[, .(
      max = max(get(var), na.rm = TRUE),
      min = min(get(var), na.rm = TRUE),
      mean = mean(get(var), na.rm = TRUE)
    ), by = id]
  })
  
  # Merge all stats into one table
  stats_dt <- Reduce(function(x, y) merge(x, y, by = "id", all = TRUE), stats_list)
  
  # Rename columns
  new_names <- c("id")
  for (var in neighbor_source_vars) {
    new_names <- c(new_names, paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean"))
  }
  setnames(stats_dt, old = names(stats_dt), new = new_names)
  
  # Merge back to year_dt
  year_dt <- merge(year_dt, stats_dt, by = "id", all.x = TRUE)
  
  result_list[[as.character(yr)]] <- year_dt
}

# Combine all years
cell_data_with_neighbors <- rbindlist(result_list)

# Now you can run predictions with the existing Random Forest model
# predictions <- predict(rf_model, newdata = cell_data_with_neighbors)
```

**Why This Is Faster**  
- Adjacency is computed once (O(#edges) ~ 1.37M).  
- For each year (~28 iterations), we do a single join and grouped aggregation using `data.table` (highly optimized in C).  
- Eliminates millions of R-level loops and string operations.  
- Memory stays within 16 GB because we process one year at a time.  

This approach should reduce runtime from 86+ hours to a few hours or less on a standard laptop.