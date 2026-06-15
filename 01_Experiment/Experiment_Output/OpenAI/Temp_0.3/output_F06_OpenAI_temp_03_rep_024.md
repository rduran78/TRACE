 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- `lapply` over ~6.46M rows repeatedly performs lookups and vector operations in pure R, which is inefficient.  
- Neighbor stats are computed row-by-row instead of using vectorized or matrix-based operations.  
- Memory overhead from repeatedly binding results and copying large objects.  

**Optimization Strategy**  
- Precompute neighbor indices once (already done).  
- Replace per-row `lapply` loops with **vectorized operations** using `data.table` or `matrix` indexing.  
- Compute all neighbor stats in a single pass per variable using fast aggregation.  
- Avoid repeated `paste` and character lookups inside loops.  
- Use `data.table` for efficient joins and grouping.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup as a list of integer vectors (already done)
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Convert neighbor_lookup to a long table for fast aggregation
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Function to compute neighbor stats for one variable using vectorized aggregation
compute_neighbor_stats_dt <- function(data, neighbor_dt, var_name) {
  vals <- data[[var_name]]
  # Attach values to neighbor_dt
  neighbor_dt[, val := vals[neighbor_id]]
  # Aggregate max, min, mean by row_id
  stats <- neighbor_dt[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = row_id]
  # Merge back into original data.table
  data[stats, on = .(row_id = .I),
       paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")) := .(max_val, min_val, mean_val)]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_dt(cell_data, neighbor_dt, var_name)
}

# cell_data now has neighbor features added efficiently
```

**Why This Works**  
- Converts neighbor relationships into a single long table, enabling fast group operations.  
- Uses `data.table` aggregation (`by = row_id`) for computing max, min, mean in one pass per variable.  
- Avoids repeated `lapply` over millions of rows.  
- Preserves original numerical estimand and trained Random Forest model.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to **minutes or a few hours**, depending on disk I/O, because aggregation and joins are highly optimized in `data.table`.  

**Memory Note**  
With 16 GB RAM, this is feasible since `neighbor_dt` will have ~1.37M rows, which is manageable.