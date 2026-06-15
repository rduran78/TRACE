 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for each cell-year row, which is extremely inefficient given 6.46M rows and 28 years. The `build_neighbor_lookup()` function is already a step toward optimization, but the main bottleneck is that neighbor stats are recomputed for every year without leveraging the static spatial adjacency. This leads to redundant lookups and memory overhead.  

**Optimization Strategy**  
- Build a **reusable neighbor adjacency table** keyed by `cell_id` only (not year), since spatial neighbors do not change over time.  
- For each year, **join yearly attributes** to this static adjacency table and compute neighbor statistics in a vectorized manner (using `data.table` or `dplyr`), rather than looping through 6.46M rows.  
- Avoid repeated `lapply` over millions of rows; instead, use grouped joins and aggregation.  
- Process one year at a time to keep memory within 16 GB.  

**Working R Code**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique is a list of integer vectors (spdep::nb object)
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique

# 1. Build static adjacency table (cell_id -> neighbor_id)
build_adjacency_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]))
  data.table(id = from, neighbor_id = to)
}

adj_table <- build_adjacency_table(id_order, rook_neighbors_unique)

# Convert cell_data to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 2. Compute neighbor stats year by year
compute_neighbor_features <- function(cell_data, adj_table, vars) {
  years <- sort(unique(cell_data$year))
  results_list <- vector("list", length(years))
  
  for (i in seq_along(years)) {
    yr <- years[i]
    message("Processing year: ", yr)
    
    # Subset data for this year
    year_data <- cell_data[year == yr, .(id, year, (vars))]
    setkey(year_data, id)
    
    # Join adjacency with year_data to get neighbor values
    joined <- adj_table[year_data, on = .(neighbor_id = id)]
    # joined has columns: id (from adj_table), neighbor_id, year, vars
    
    # Compute stats by original id
    agg <- joined[, lapply(.SD, function(x) {
      if (all(is.na(x))) return(c(NA_real_, NA_real_, NA_real_))
      c(max(x, na.rm = TRUE), min(x, na.rm = TRUE), mean(x, na.rm = TRUE))
    }), by = id, .SDcols = vars]
    
    # Flatten column names
    new_names <- unlist(lapply(vars, function(v) paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
    setnames(agg, old = names(agg)[-1], new = new_names)
    
    # Add year
    agg[, year := yr]
    results_list[[i]] <- agg
  }
  
  # Combine all years
  rbindlist(results_list)
}

neighbor_features <- compute_neighbor_features(cell_data, adj_table, neighbor_source_vars)

# Merge back into cell_data
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data, on = .(id, year)]

# Now cell_data has neighbor stats for each variable
# Proceed with Random Forest prediction using the existing trained model
```

**Why This Is Faster**  
- Adjacency is computed once (1.37M relationships) instead of per row.  
- Yearly processing uses efficient joins and aggregation in `data.table`.  
- Avoids 6.46M `lapply` calls; instead, uses vectorized operations and grouping.  
- Memory is controlled by processing one year at a time.  

This approach should reduce runtime from 86+ hours to a few hours on a standard laptop.