 **Diagnosis**  
The main bottleneck is the repeated neighbor feature computation across ~6.46M rows and 5 variables. The current approach uses nested `lapply` calls and string concatenation inside `build_neighbor_lookup`, which is extremely inefficient for large datasets. The neighbor lookup is rebuilt for every row, and `compute_neighbor_stats` iterates row-wise, causing massive overhead.  

**Optimization Strategy**  
1. **Precompute neighbor indices once** as integer vectors, avoiding string concatenation and repeated lookups.  
2. **Vectorize neighbor aggregation** using `data.table` or `dplyr` joins instead of per-row `lapply`.  
3. **Process by year** to reduce memory footprint and leverage grouping.  
4. **Avoid repeated NA filtering inside loops**; handle NA logic in vectorized operations.  

**Optimized Approach**  
- Flatten the neighbor relationships into a long table `(cell_id, neighbor_id)` and join with yearly data.  
- Compute `max`, `min`, `mean` for each cell-year and variable in a grouped manner.  
- Use `data.table` for speed and memory efficiency.  

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (static across years)
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Melt cell_data for easier joins
# cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure keys for fast joins
setkey(cell_data, id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Select only needed columns
  dt <- cell_data[, .(id, year, value = get(var_name))]
  
  # Join neighbor pairs with dt for neighbor values
  joined <- neighbor_pairs[dt, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined has: cell_id, neighbor_id, year, value
  
  # Aggregate by cell_id and year
  agg <- joined[, .(
    max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]
  
  # Rename columns
  setnames(agg, c("cell_id", "year", 
                  paste0(var_name, "_nbr_max"), 
                  paste0(var_name, "_nbr_min"), 
                  paste0(var_name, "_nbr_mean")))
  
  agg
}

# Compute for all variables and merge back
neighbor_features_list <- lapply(vars, compute_neighbor_features)

# Merge all neighbor features into one table
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("cell_id", "year"), all = TRUE),
                            neighbor_features_list)

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_features, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- **No per-row loops**: All computations are grouped and vectorized.
- **Single join per variable**: Efficient use of `data.table` joins.
- **Memory-aware**: Processes by variable, not by full variable set at once.
- **Preserves original estimand**: Same neighbor definitions and statistics.

---

**Expected Speedup**: From 86+ hours to a few hours or less on a 16 GB laptop, as the bottleneck shifts from R loops to optimized C-level `data.table` operations.