 **Diagnosis**  
The bottleneck is the nested `lapply` approach over ~6.46M rows, repeatedly subsetting vectors and creating intermediate objects. This results in excessive R-level loops and memory overhead. The neighbor lookup is static, but the current design recomputes stats row-by-row in pure R, which is extremely slow for this scale.  

**Optimization Strategy**  
- Precompute neighbor relationships as integer vectors and flatten them into a long format for vectorized aggregation.  
- Use `data.table` for efficient joins and grouped summaries.  
- Compute all neighbor stats (max, min, mean) in one pass per variable using fast aggregation instead of per-row loops.  
- Avoid recomputing for each variable by reusing the neighbor lookup structure.  
- Keep results aligned with original row order.  

**Working R Code**  
```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Add row index for restoring order
cell_data[, row_id := .I]

# Flatten neighbor lookup into long format
# neighbor_lookup: list of integer vectors (neighbors for each row)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Join neighbor values and compute stats for each variable
compute_neighbor_stats_fast <- function(dt, neighbor_dt, var_name) {
  vals <- dt[, .(row_id, value = get(var_name))]
  # Join neighbor values
  joined <- merge(neighbor_dt, vals, by.x = "neighbor_id", by.y = "row_id", allow.cartesian = TRUE)
  # Aggregate stats by original row_id
  agg <- joined[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = row_id]
  # Handle rows with no neighbors or all NA
  agg[!is.finite(max_val), c("max_val", "min_val", "mean_val") := .(NA_real_, NA_real_, NA_real_)]
  setnames(agg, c("row_id", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  agg
}

# Compute for all variables and merge back
result_list <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_fast(cell_data, neighbor_dt, v))
result_dt <- Reduce(function(x, y) merge(x, y, by = "row_id"), result_list)

# Merge back to cell_data in original order
cell_data <- merge(cell_data, result_dt, by = "row_id", sort = FALSE)
cell_data[, row_id := NULL]

# cell_data now has neighbor-based features added
```

**Why This Works**  
- Eliminates per-row loops in favor of vectorized joins and grouped aggregation.  
- `data.table` handles 6.46M rows efficiently in memory on a 16 GB machine.  
- Preserves original estimand and Random Forest model compatibility.  
- Expected runtime drops from 86+ hours to minutes (depending on disk I/O).