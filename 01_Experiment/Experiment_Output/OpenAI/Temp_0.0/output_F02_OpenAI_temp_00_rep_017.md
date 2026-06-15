 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `lapply` over 6.46M rows creates huge R lists in memory.  
- `build_neighbor_lookup` repeats string concatenation and lookups for every row.  
- `compute_neighbor_stats` repeatedly subsets vectors and computes stats in R loops.  
- No vectorization or efficient join strategy is used.  

**Optimization Strategy**  
- Avoid per-row `lapply` loops; use **vectorized joins** or **data.table** for speed.  
- Precompute neighbor relationships at the **cell level**, then join by year instead of recomputing for every row.  
- Use **data.table grouping** to compute neighbor stats in bulk.  
- Keep everything in long format and leverage fast aggregation.  

**Optimized Approach**  
1. Expand neighbor relationships to a long table: `(cell_id, neighbor_id)`.  
2. Cross with years to get `(cell_id, year, neighbor_id)`.  
3. Join neighbor values for each variable.  
4. Compute `max`, `min`, `mean` by `(cell_id, year)`.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Step 1: Build neighbor pairs (cell_id, neighbor_id)
id_order_dt <- data.table(id = id_order, idx = seq_along(id_order))
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Step 2: Expand to all years
years <- unique(cell_data$year)
neighbor_pairs_expanded <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]

# Step 3: Join with cell_data to get neighbor values
setkey(cell_data, id, year)
setkey(neighbor_pairs_expanded, neighbor_id, year)

# For each variable, compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- neighbor_pairs_expanded[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  setnames(tmp, var_name, "neighbor_val")
  
  # Compute stats by (id, year)
  stats <- tmp[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle all-NA cases
  stats[!is.finite(max_val), c("max_val", "min_val", "mean_val") := NA]
  
  # Merge back to cell_data
  setkey(stats, id, year)
  cell_data <- stats[cell_data, on = .(id, year)]
  
  # Rename columns
  setnames(cell_data, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
}

# cell_data now has neighbor features added efficiently
```

**Why This Works**  
- Eliminates per-row loops; uses `data.table` joins and group aggregation.  
- Memory footprint is reduced by processing one variable at a time.  
- Scales to millions of rows and millions of neighbor links efficiently.  

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to **under 1–2 hours**, depending on disk speed and CPU.  

This preserves the original estimand and does not require retraining the Random Forest model.