 **Diagnosis**  
The current implementation is extremely slow because:  
- `lapply` loops over ~6.46M rows, creating millions of small vectors and lists.  
- Repeated string concatenation (`paste`) and lookups in named vectors for each row is very expensive.  
- Neighbor relationships are computed repeatedly for each row rather than vectorized.  
- No use of efficient data structures (e.g., integer indexing, matrix operations).  

**Optimization Strategy**  
- Precompute neighbor lookups as integer indices without string manipulation.  
- Use `data.table` for fast join and aggregation.  
- Store neighbor relationships in a long format (edge list) and compute max, min, mean via grouped operations.  
- Parallelize or use vectorized operations instead of millions of `lapply` calls.  
- Avoid building separate lists for each row; instead, compute statistics by joining the dataset to itself on neighbor relationships.  

**Working R Code**  
Below is an optimized approach using `data.table`:

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of unique cell ids in the same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Build edge list (directed)
neighbor_edges <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id      = id_order[i],
    nb_id   = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cross join with years to replicate panel structure
years <- unique(cell_data$year)
neighbor_edges <- neighbor_edges[, .(nb_id, year = years), by = id]

# Merge neighbor values (long format)
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  # Join neighbor values
  neighbor_vals <- cell_data[, .(nb_id = id, year, value = get(v))]
  dt <- merge(neighbor_edges, neighbor_vals, by = c("nb_id", "year"), all.x = TRUE)
  
  # Compute stats by (id, year)
  stats <- dt[, .(
    max_val  = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
    min_val  = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to main table
  setnames(stats, c("id", "year", paste0(v, "_nb_max"), paste0(v, "_nb_min"), paste0(v, "_nb_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor-based max, min, mean for each variable
```

**Why this is fast:**  
- Single pass per variable using `data.table` grouping instead of millions of small list operations.  
- Avoids per-row string concatenation and repeated lookups.  
- Scales well for millions of rows because `data.table` uses efficient hashing and memory.  

**Expected improvement:**  
- From 86+ hours to roughly minutes-to-a-few-hours on 16 GB RAM for 6.46M rows, depending on I/O and threads.  

This preserves the original rook-neighbor relationships, computes exact max/min/mean, and keeps the trained Random Forest model intact.