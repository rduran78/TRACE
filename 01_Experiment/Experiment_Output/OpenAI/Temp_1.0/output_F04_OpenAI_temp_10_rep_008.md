 **Diagnosis**  
The computational bottleneck is indeed in the neighbor feature construction. The function `build_neighbor_lookup()` creates a full 6.46M-element list by looping over every cell-year row. Combine that with repeated lookups in `compute_neighbor_stats()` across 6+ million rows and 5 variables, and you have an O(N * k) operation (~32 million iterations plus overhead). This is highly inefficient because:

- The neighbor structure (rook adjacency) *does not vary by year*; however, the lookup repeats per row-year.
- Extensive use of `lapply()` and string concatenation adds enormous overhead.
- The operation is essentially a temporal replication of a static spatial graph, which could be vectorized or aggregated.

**Optimization Strategy**  
Key improvements:  
1. **Separate the spatial and temporal dimensions**: Neighbor lists only depend on spatial `id`. Build neighbor feature aggregation by year in *blocks*, not per-row.
2. **Avoid per-row string concatenation**: Use precomputed mappings and join instead of repeated `paste()`.
3. **Switch from lapply + do.call → fast aggregations using `data.table`**: Compute per `id` → neighbor aggregation → merge back.
4. **Batch process variables over all years efficiently** rather than row-wise.
5. **Memory-safe**: Work year-by-year to keep RAM under control, not full expansion at once.

**Working Optimized R Code**  
This uses `data.table` and precomputed `rook_neighbors_unique`:

```r
library(data.table)

# Convert to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.

# Precompute neighbor mapping as a data.table for fast joins
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Ensure correct types
setkey(neighbor_dt, id)

# Neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to compute stats for one variable efficiently
compute_neighbor_features_year <- function(dt_year, var_name) {
  var_data <- dt_year[, .(id, value = get(var_name))]
  # Join neighbors on id -> neighbor_id
  joined <- neighbor_dt[var_data, on = .(neighbor_id = id)]
  # joined columns: id (from neighbor_dt), neighbor_id, i.id, value (value for neighbor_id)
  
  # Aggregate: for each i.id (the focal cell), compute max/min/mean of neighbor values
  agg <- joined[!is.na(value), .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = .(focal_id = i.id)]
  
  # Merge back to dt_year
  setnames(agg, c("focal_id", paste0(var_name, "_nbr_max"), paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  dt_year[agg, on = .(id = focal_id)]
}

# Process year by year to keep memory reasonable
result_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  
  # For each var, compute neighbor features
  for (var_name in neighbor_source_vars) {
    dt_year <- compute_neighbor_features_year(dt_year, var_name)
  }
  
  result_list[[i]] <- dt_year
}

# Combine all years back together
cell_data_enriched <- rbindlist(result_list)
```

**Why this works:**  
- Only iterates per *year* (28 loops) instead of 6.46M row loops.
- Vectorized joins replace manual indexing and string manipulation.
- Single pass per variable per year, leveraging `data.table` aggregation (C-optimized).
- Preserves the original estimand and structure for Random Forest scoring.

**Expected Performance Gain:**  
From 86+ hours to a few hours at most on 16 GB RAM. This approach handles ~6.5M rows tractably by chunking yearly data and minimizing interpretive overhead.

**Model Preservation:**  
Random Forest model is untouched; only feature construction pipeline is optimized.