 **Diagnosis**  
The current approach is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length 6.46M (one per cell-year), each storing neighbor indices. This is huge in memory and expensive to compute.  
- `compute_neighbor_stats` repeatedly traverses large lists and subsets vectors, causing high overhead.  
- The process is inherently row-wise (`lapply` over millions of rows), which is inefficient in R.  

**Optimization Strategy**  
1. **Avoid per-row neighbor lookups**: Instead of building a massive `neighbor_lookup` for every cell-year, leverage vectorized joins or matrix operations.  
2. **Process by year**: Compute neighbor stats year-by-year to keep memory manageable.  
3. **Use `data.table` for speed**: Efficient grouping and joining.  
4. **Pre-expand neighbor pairs**: Create a long table of `(cell_id, neighbor_id)` pairs once, then join with yearly data.  
5. **Aggregate neighbor stats in bulk**: Compute `max`, `min`, `mean` using `data.table` aggregation instead of `lapply`.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (directed)
# rook_neighbors_unique: list of neighbors per cell_id in id_order
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Ensure neighbor_ids are in same id space as cell_data$id
setkey(neighbor_pairs, cell_id)

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(dt, var_name) {
  # Select only needed columns
  dt_sub <- dt[, .(id, year, value = get(var_name))]
  
  # Join neighbor pairs with dt_sub for each year
  # Process year by year to limit memory
  result_list <- vector("list", length(unique(dt_sub$year)))
  years <- sort(unique(dt_sub$year))
  
  for (yr in years) {
    dt_year <- dt_sub[year == yr]
    
    # Join neighbor values
    setkey(dt_year, id)
    joined <- neighbor_pairs[dt_year, on = .(neighbor_id = id), nomatch = 0]
    # joined: cell_id, neighbor_id, year, value
    
    # Aggregate neighbor stats by cell_id
    agg <- joined[, .(
      max_val = max(value, na.rm = TRUE),
      min_val = min(value, na.rm = TRUE),
      mean_val = mean(value, na.rm = TRUE)
    ), by = .(cell_id)]
    
    # Handle all-NA neighbors: replace Inf/-Inf with NA
    agg[is.infinite(max_val), max_val := NA]
    agg[is.infinite(min_val), min_val := NA]
    
    agg[, year := yr]
    result_list[[as.character(yr)]] <- agg
  }
  
  # Combine yearly results
  rbindlist(result_list)
}

# Compute and merge all neighbor features
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Processing ", var_name)
  neighbor_stats <- compute_neighbor_features(cell_data, var_name)
  
  # Merge back to cell_data
  setkey(cell_data, id, year)
  setkey(neighbor_stats, cell_id, year)
  
  cell_data <- neighbor_stats[cell_data]
  
  # Rename columns
  setnames(cell_data,
           old = c("max_val", "min_val", "mean_val"),
           new = paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
}

# cell_data now contains added neighbor features
```

---

### **Why This Works**
- **No giant list**: We never create a 6.46M-element list.  
- **Year-by-year processing**: Keeps memory footprint low.  
- **Bulk aggregation**: `data.table` computes neighbor stats in C-speed.  
- **Scales well**: Only stores neighbor pairs (~1.37M rows) and yearly slices (~344k rows each).  

**Expected runtime**: Minutes to a few hours on a laptop instead of 86+ hours. Memory stays within 16 GB.  

**Preserves**:  
- Original estimand and trained Random Forest model.  
- Numerical integrity of neighbor-based features.