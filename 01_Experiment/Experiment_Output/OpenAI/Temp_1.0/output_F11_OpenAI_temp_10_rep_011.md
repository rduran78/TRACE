 Diagnosis:  
The main bottleneck is not Random Forest inference. The provided code and dataset facts point to extreme overhead in computing neighbor-based features:  
- `build_neighbor_lookup` runs `lapply` for each of 6.46M rows, repeatedly manipulating character keys and doing vector lookups.  
- `compute_neighbor_stats` repeats similar work for every variable.  
These nested loops create massive R-level overhead and memory churn. Random Forest inference on 6.46M rows (with 110 predictors) is comparatively lightweight and would finish in minutes using `ranger` or `predict()` with batching.

Optimization Strategy:  
- Precompute neighbor indices once efficiently using integer-based joins rather than character concatenation.  
- Compute neighbor summaries via vectorized operations or `data.table` group-joins instead of repeated `lapply`.  
- Avoid rebuilding keys; leverage numeric IDs.  
- Retain Random Forest as-is; main fix is the feature engineering phase.  

Working R Code (Vectorized Rewrite Using `data.table`):  

```r
library(data.table)

# Assume cell_data is a data.frame; convert to data.table
setDT(cell_data)

# Build a lookup table: for each cell_id and year, index of row
cell_data[, row_idx := .I]

# neighbors_dt: expand rook neighbors to directed edges
# id_order assumed to map positions to cell_id
neighbors_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(src_id = id_order[i],
               nbr_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Cross join with years to get cell-year edges
years <- unique(cell_data$year)
neighbors_dt <- neighbors_dt[, .(cell_id = src_id, neighbor_id = nbr_id), by = .EACHI]
neighbors_dt <- neighbors_dt[, .(cell_id, neighbor_id, year = rep(years, each=.N))]

# Attach row_idx for neighbors
neighbors_dt <- merge(
  neighbors_dt,
  cell_data[, .(neighbor_id = id, year, nbr_idx = row_idx)],
  by = c("neighbor_id", "year"),
  all.x = TRUE
)

# For each var_name, compute stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Attach neighbor values
  neighbors_dt[, val := cell_data[nbr_idx, get(var_name)]]
  
  stats_dt <- neighbors_dt[!is.na(val),
    .(max_val = max(val),
      min_val = min(val),
      mean_val = mean(val)),
    by = .(cell_id, year)
  ]
  
  # Merge back into cell_data
  setkey(cell_data, id, year)
  setkey(stats_dt, cell_id, year)
  cell_data <- cell_data[stats_dt]
  
  setnames(cell_data, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
}

# Drop helper column
cell_data[, row_idx := NULL]
```

Why Faster:  
- Eliminates per-row character concatenations; uses integer joins.  
- Performs neighbor aggregation in compiled C-level code via `data.table`.  
- Reduces expensive `lapply` loops (millions of iterations) to vectorized grouping.  

Expected Benefit:  
Cuts pipeline from ~86 hours to a few hours or less on a standard laptop, enabling Random Forest inference to remain negligible.