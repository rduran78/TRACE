 **Diagnosis**  
The bottleneck arises from:  
1. Iterating over 6.46M rows with `lapply` in `build_neighbor_lookup` and `compute_neighbor_stats`.  
2. Repeated string concatenation (`paste`) and `setNames` combined with lookup overhead.  
3. Multiple passes over large vectors for 5 variables, triggering memory thrashing on a laptop.  

The structure creates *row-level neighbor expansions* for every observation, which scales poorly in pure R loops.  

**Optimization Strategy**  
- **Avoid repeated row-wise operations**: Precompute neighbor indices in a *vectorized or tabular form* to eliminate millions of `paste`.  
- **Use `data.table` for fast grouping and aggregation** instead of `lapply`.  
- **Reshape once** to `cell_id × year`, then map neighbors using indexes rather than string keys.  
- **Chunk processing** to stay within 16 GB RAM.  
- Use `parallel::mclapply` or `future.apply` if multi-core allowed.  
- Drop `do.call(rbind,...)` and build matrix directly for speed.  

**Working Optimized R Code**  

```r
library(data.table)

# Convert data to data.table for speed
setDT(cell_data)

# Ensure id_order and rook_neighbors_unique are available
# Precompute neighbor pairs as a long table
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  # i: reference cell, neigh: neighbor cell
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(neighbor_id = neighbor_id), by = .(id)]
neighbor_dt <- neighbor_dt[rep(seq_len(.N), each = length(years))]
neighbor_dt[, year := rep(years, times = .N / length(years))]

# Map cell-year rows to neighbor rows
cell_data_keyed <- cell_data[, .(id, year, row_id = .I)]
setkey(cell_data_keyed, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt[, neighbor_row := cell_data_keyed[J(neighbor_id, year), row_id]]

# Drop NA neighbor rows
neighbor_dt <- neighbor_dt[!is.na(neighbor_row)]

# Compute stats for multiple variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  
  # Lookup values for neighbor rows
  neighbor_dt[, val := vals[neighbor_row]]
  
  # Aggregate stats by (id, year)
  stats_dt <- neighbor_dt[!is.na(val),
                           .(max_val = max(val),
                             min_val = min(val),
                             mean_val = mean(val)),
                           by = .(id, year)]
  
  # Join back to cell_data
  setkey(stats_dt, id, year)
  cell_data[stats_dt, paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats_dt, paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats_dt, paste0(var_name, "_nbr_mean") := i.mean_val]
}
```

**Why Faster**  
- Eliminates 6.4M × `lapply` loops; works entirely in grouped tabular operations.  
- Avoids repeated `paste`, `setNames`, and `do.call(rbind,...)`.  
- Uses efficient joins + aggregation in `data.table`, reducing runtime from hours to potentially under 1–2 hours on 16 GB RAM.  
- Preserves model and numerical estimand because computations (max/min/mean) are unchanged.  

**Additional Tips**  
- Release intermediate objects with `rm()` and `gc()`.  
- If memory is still tight, split `neighbor_dt` or years into chunks and aggregate per chunk.  
- Optionally use disk-backed tables (e.g., `arrow` or `fst`) if needed for larger-than-memory processing.