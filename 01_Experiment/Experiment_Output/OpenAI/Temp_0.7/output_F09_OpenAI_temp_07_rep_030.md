 **Diagnosis**  
The current pipeline recomputes neighbor relationships for each cell-year row repeatedly, causing severe inefficiency. With 6.46M rows and ~1.37M neighbor relationships, looping through all rows for every variable leads to massive redundant work. The major bottleneck: `build_neighbor_lookup` and `compute_neighbor_stats` iterate over all 6.46M rows multiple times, and neighbor lookup is recalculated for every year-variable combination.

**Optimization Strategy**  
1. **Precompute adjacency once at the cell level**: Build a reusable neighbor table mapping each cell to its neighbors (using `rook_neighbors_unique`).  
2. **Use year-based joins instead of nested loops**: For each year, join cell attributes with neighbors via an expanded table, compute aggregate stats (`max`, `min`, `mean`) in a vectorized way.  
3. **Leverage `data.table` for speed and memory efficiency**: Avoid per-row `lapply`. Use `data.table` joins and group operations.  
4. **Do not retrain the Random Forest model**: Only recompute predictors efficiently, preserving the original estimand.  
5. Process in yearly chunks to avoid memory overload.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build adjacency table ONCE
# rook_neighbors_unique: list of integer vectors, same order as id_order
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(
  lapply(seq_along(adj_list), function(i) {
    if (length(adj_list[[i]]) == 0) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
  })
)

setkey(adj_dt, cell_id)

# Variables for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output storage
results_list <- vector("list", length(neighbor_source_vars))

# Process year by year to keep memory manageable
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for year
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(dt_year, id)
  
  # Join adjacency
  merged <- adj_dt[dt_year, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # merged has: cell_id, neighbor_id, year data from neighbor

  for (var_name in neighbor_source_vars) {
    # Compute neighbor stats for this var and year
    stats <- merged[, .(
      max = max(get(var_name), na.rm = TRUE),
      min = min(get(var_name), na.rm = TRUE),
      mean = mean(get(var_name), na.rm = TRUE)
    ), by = cell_id]
    
    # Handle NaN if all NA
    stats[is.infinite(max), max := NA]
    stats[is.infinite(min), min := NA]
    stats[is.nan(mean), mean := NA]
    
    # Rename columns
    setnames(stats, c("cell_id", paste0(var_name, "_nbr_max"), paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
    
    # Merge back to dt_year
    dt_year <- stats[dt_year, on = .(cell_id = id)]
  }
  
  # Append year results back to main data
  cell_data[year == yr, (setdiff(names(dt_year), c("id","year"))) := dt_year[, setdiff(names(dt_year), c("cell_id","year"))]]
}

# At this point, cell_data includes all neighbor-based predictors, ready for RF prediction
```

---

### **Why This is Faster**
- **Adjacency computed once**: No repeated construction of neighbor lookups.
- **Yearly chunking**: Limits memory footprint while processing 6.46M rows.
- **Vectorized group operations via `data.table`**: Avoids costly per-row `lapply`.
- **Single-pass per variable per year**: Reduces redundancy from 86+ hours to likely under 2–4 hours on a 16GB laptop.

**Preserves**:
- Original Random Forest model.
- Original numerical estimand.
- Same neighbor-based feature logic, just computed efficiently.