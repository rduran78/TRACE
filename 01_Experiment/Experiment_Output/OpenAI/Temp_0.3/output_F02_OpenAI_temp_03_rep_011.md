 **Diagnosis**  
The bottleneck arises from:
- `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`.
- Repeated string concatenation (`paste`) and hash lookups in tight loops.
- Storing large lists in memory (neighbor_lookup for millions of rows).
- Lack of vectorization or efficient joins for neighbor feature computation.

This approach is inherently slow and memory-heavy because it builds a per-row neighbor index and repeatedly computes statistics in R loops.

---

**Optimization Strategy**  
1. **Avoid per-row `lapply`**: Use `data.table` for fast keyed joins and aggregations.
2. **Precompute neighbor relationships as a long table**: Expand neighbors once, then join with data for all years.
3. **Compute stats in a grouped manner**: Aggregate max, min, mean by `(id, year)` using `data.table`.
4. **Memory efficiency**: Drop intermediate objects early, use integer keys, and avoid string concatenation in loops.
5. **Parallelization**: Use `data.table`’s multithreading (built-in) for aggregation.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in same order as rook_neighbors_unique
# rook_neighbors_unique: list of integer vectors (neighbors)

# Step 1: Build neighbor pairs (id, neighbor_id)
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Step 2: Expand for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(id = id, year = years), on = "id", allow.cartesian = TRUE]
setnames(neighbor_dt, "id", "cell_id")

# Step 3: Join neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]

# neighbor_dt now has: cell_id, year, neighbor_id, var columns

# Step 4: Compute stats for each var
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_dt <- neighbor_dt[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Merge back into cell_data
  setnames(stats_dt, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats_dt[cell_data, on = .(cell_id = id, year)]
}

# Final cell_data now includes neighbor-based features
```

---

**Why this works**  
- Eliminates per-row loops and string operations.
- Uses `data.table` joins and aggregations, which are highly optimized in C.
- Handles all variables in bulk rather than recomputing neighbor lookups repeatedly.
- Scales to millions of rows with reasonable memory on a 16 GB machine.

**Expected Performance**  
From 86+ hours → likely under 1–2 hours on a standard laptop (depending on I/O and CPU).