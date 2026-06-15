 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` over 6.46M rows in both `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation, list indexing, and repeated lookups in R lists, which is extremely inefficient at this scale. The neighbor relationships are static across years, yet the current implementation recomputes them for every row-year combination.  

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** and reuse across years instead of recomputing for each row.  
2. **Vectorize operations**: Avoid `lapply` over millions of rows; use `data.table` for fast joins and grouped aggregation.  
3. **Reshape data to wide or keyed format** for efficient neighbor feature computation.  
4. **Compute neighbor stats in bulk** using joins rather than per-row loops.  

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure keys for fast joins
setkey(cell_data, id, year)

# Precompute neighbor pairs (directed)
# rook_neighbors_unique: list of neighbors per cell id in id_order
neighbor_pairs <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand neighbor pairs across all years
years <- sort(unique(cell_data$year))
neighbor_pairs <- neighbor_pairs[, .(id = from, neighbor_id = to), by = years]
setnames(neighbor_pairs, "years", "year")

# Join to get neighbor values for each variable
compute_neighbor_features <- function(dt, var_name) {
  # Join neighbor values
  merged <- neighbor_pairs[
    dt[, .(id, year, val = get(var_name))],
    on = .(neighbor_id = id, year),
    nomatch = 0
  ]
  
  # Compute stats by (id, year)
  stats <- merged[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back into original data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  dt[stats, on = .(id, year)]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_neighbor_features(cell_data, var_name)
}
```

**Why This Works**  
- **Single neighbor expansion**: We build a `(id, neighbor_id, year)` table once, not per row.  
- **Bulk aggregation**: `data.table` computes max/min/mean in C, avoiding millions of R-level loops.  
- **Memory efficiency**: Only relevant columns are joined; no large intermediate lists.  

**Expected Performance Gain**  
This approach reduces complexity from O(N × neighbors) per variable in R loops to efficient joins and grouped aggregation in C via `data.table`. On a 16 GB laptop, runtime should drop from 86+ hours to a few hours or less.  

**Preserves**  
- Original estimand  
- Trained Random Forest model  
- Numerical integrity of neighbor-based features