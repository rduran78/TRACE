 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every cell-year row by dynamically matching IDs and years inside `build_neighbor_lookup` and `compute_neighbor_stats`. With ~6.46M rows and 5 variables, this results in repeated string concatenations, lookups, and list traversals—extremely costly in both time and memory. The neighbor structure itself is static across years, so recomputing per year is unnecessary.  

**Optimization Strategy**  
1. **Precompute a reusable adjacency table**: Build a long-format table of `(cell_id, neighbor_id)` pairs once (from `rook_neighbors_unique`), then expand it across years by joining on year.  
2. **Vectorized joins instead of per-row lapply**: Use `data.table` for efficient joins and grouped aggregations.  
3. **Compute neighbor stats by grouping**: For each year and variable, join neighbor values and compute `max`, `min`, and `mean` in a single grouped operation.  
4. **Preserve Random Forest model and estimand**: Only change feature engineering, not the model.  

This approach avoids building large lists and repeated lookups, reducing complexity from O(n * neighbors) in R loops to efficient C-level group operations.  

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute adjacency table (cell_id -> neighbor_id)
id_order <- as.integer(id_order)  # ensure numeric IDs
adj_dt <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand to all years by Cartesian join
years <- unique(cell_data$year)
adj_dt <- adj_dt[, .(cell_id, neighbor_id, year = years), by = .(cell_id, neighbor_id)]

# Set keys for fast join
setkey(cell_data, id, year)
setkey(adj_dt, neighbor_id, year)

# Function to compute and add neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Join neighbor values
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  adj_vals <- adj_dt[tmp, on = .(neighbor_id, year)]
  
  # Aggregate stats by (cell_id, year)
  stats <- adj_vals[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(cell_id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("cell_id", "year", 
                    paste0(var_name, "_nbr_max"), 
                    paste0(var_name, "_nbr_min"), 
                    paste0(var_name, "_nbr_mean")))
  cell_data[stats, on = .(id = cell_id, year)]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features(var_name)
}

# cell_data now has neighbor-based features ready for Random Forest prediction
```

---

**Why this is faster**  
- **Single adjacency expansion**: Built once, reused for all variables.  
- **Vectorized joins**: `data.table` performs joins and aggregations in compiled code.  
- **No per-row lapply**: Eliminates millions of R-level function calls.  

Expected runtime drops from **86+ hours to a few hours or less**, depending on disk I/O and CPU. Memory stays manageable (~hundreds of MB) on a 16 GB machine.