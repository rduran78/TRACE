 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly perform expensive list lookups and string concatenations. The neighbor relationships are static across years, but the code recomputes neighbor indices for every row-year combination. This results in massive redundant computation and memory overhead.

---

**Optimization Strategy**  
1. **Precompute a reusable neighbor lookup by cell ID only (not year)** since spatial adjacency does not change over time.
2. For each year, **join yearly attributes to this static neighbor table** and compute neighbor statistics using vectorized operations or `data.table` grouping instead of per-row `lapply`.
3. Avoid repeated string concatenation and repeated indexing; use integer joins.
4. Use `data.table` for efficient joins and aggregations on large datasets.
5. Preserve the trained Random Forest model and original estimand by only changing feature engineering speed.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute static neighbor lookup by cell_id
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Join neighbor table with cell_data to get neighbor values
  tmp <- neighbor_dt[cell_data, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # tmp now has: id (focal), neighbor_id, year, var_name
  tmp[, focal_id := i.id]
  tmp[, focal_year := i.year]
  tmp[, val := get(var_name)]
  
  # Compute stats by focal_id and focal_year
  stats <- tmp[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(focal_id, focal_year)]
  
  # Merge back to cell_data
  setnames(stats, c("focal_id", "focal_year"), c("id", "year"))
  setkey(stats, id, year)
  cell_data[stats, paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats, paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats, paste0(var_name, "_nbr_mean") := i.mean_val]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(var_name)
}

# cell_data now has neighbor features added efficiently
```

---

**Why This Is Faster**  
- The neighbor relationships (~1.37M) are stored once in `neighbor_dt`.
- For each year, joins and aggregations are done in bulk using `data.table` rather than per-row loops.
- Eliminates repeated string concatenation and redundant lookups.
- Memory footprint is reduced because we avoid creating millions of intermediate lists.

**Expected Performance Gain**  
From 86+ hours to a few hours or less on a 16 GB laptop, depending on disk I/O and CPU speed.