 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics (`max`, `min`, `mean`) for every cell-year row (≈6.46M rows) by iterating over a large list of neighbor indices. This is inefficient because:  
- The neighbor structure is static across years, but the code redundantly rebuilds neighbor relationships for each row-year combination.  
- The computation is done in pure R with nested `lapply` calls, which is memory- and CPU-intensive.  
- No vectorization or grouping by year is used.  

**Optimization Strategy**  
- Precompute a static neighbor index mapping at the cell level (not cell-year).  
- For each year, extract the relevant subset of data and compute neighbor stats in a vectorized way using matrix operations or `data.table`.  
- Append results back to the full dataset.  
- Avoid recomputing the neighbor list repeatedly.  
- Use efficient joins instead of per-row loops.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute static neighbor lookup at cell level
# neighbor_dt: each row is (id, neighbor_id)
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(lapply(rook_neighbors_unique, function(x) id_order[x]))
)

# Ensure keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(dt, var_name) {
  # Join neighbors with data on the same year
  joined <- neighbor_dt[dt, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined now has: id (from neighbor_dt), neighbor_id, year, var_name
  # Compute stats grouped by (id, year)
  stats <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Replace NaN with NA
  for (col in c("max_val", "min_val", "mean_val")) {
    stats[[col]][is.infinite(stats[[col]]) | is.nan(stats[[col]])] <- NA
  }
  
  # Merge back into main data
  setnames(stats, c("id", "year", 
                    paste0(var_name, "_nbr_max"), 
                    paste0(var_name, "_nbr_min"), 
                    paste0(var_name, "_nbr_mean")))
  dt[stats, on = .(id, year), 
     `:=`( (paste0(var_name, "_nbr_max")) = i[[paste0(var_name, "_nbr_max")]],
           (paste0(var_name, "_nbr_min")) = i[[paste0(var_name, "_nbr_min")]],
           (paste0(var_name, "_nbr_mean")) = i[[paste0(var_name, "_nbr_mean")]] )]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(cell_data, var_name)
}
```

**Why this works efficiently**  
- `neighbor_dt` is built once and reused.  
- Joins and aggregations are done by `data.table` in C, which is highly optimized.  
- Memory footprint is controlled because we process variable by variable.  
- Preserves original estimand and does not retrain the Random Forest.  

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk speed and CPU.