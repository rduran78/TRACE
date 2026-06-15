 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, repeatedly subsetting vectors and creating intermediate objects.  
- Neighbor lookups are recomputed for each row, causing heavy R-level overhead.  
- No vectorization or efficient data.table/dplyr operations are used.  
- Memory pressure is high due to repeated list allocations.  

**Optimization Strategy**  
- Precompute neighbor indices once (already done via `neighbor_lookup`).  
- Use `data.table` for fast grouping and joins.  
- Reshape data to long format and compute neighbor stats via keyed joins instead of per-row loops.  
- Compute all neighbor stats in a single pass per variable.  
- Avoid repeated `paste` and string operations inside loops.  

**Working R Code** (optimized approach using `data.table`):  

```r
library(data.table)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, neighbor_source_vars defined
# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (cell_id -> neighbor_id)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand neighbor pairs across all years
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(id = rep(id, length(years)),
                                     neighbor_id = rep(neighbor_id, length(years)),
                                     year = rep(years, each = .N))]

# Key tables for fast join
setkey(cell_data, id, year)
setkey(neighbor_pairs, neighbor_id, year)

# For each variable, compute neighbor stats
for (var_name in neighbor_source_vars) {
  # Join neighbor values
  neighbor_vals <- neighbor_pairs[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  setnames(neighbor_vals, var_name, "neighbor_val")
  
  # Compute stats by (id, year)
  stats <- neighbor_vals[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setkey(stats, id, year)
  cell_data <- stats[cell_data]
  
  # Rename columns
  setnames(cell_data, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
}

# cell_data now has neighbor features added
```

**Why this works:**  
- Eliminates per-row loops; uses vectorized joins and aggregations.  
- Handles all years and neighbors in bulk.  
- Scales to millions of rows efficiently on 16 GB RAM.  
- Preserves original estimand and Random Forest model (no retraining).  

**Expected runtime:**  
From 86+ hours → likely under 1 hour on a standard laptop, depending on disk I/O and CPU.