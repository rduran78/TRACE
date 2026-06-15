 **Diagnosis**  
- Current method uses nested `lapply` over ~6.46M rows with multiple variables → O(n × k) lookup, large object serialization overhead, repeated string construction (`paste` calls), and list-of-lists indexing.  
- `neighbor_lookup` is computed once but indexing occurs repeatedly, costing extra time.  
- Memory stress comes from storing huge lists and repeated conversions to integer vectors.  
- Main bottlenecks:  
  1. String-based keys (`paste`) for every observation.  
  2. Multiple passes through neighbor lists for 5 variables.  
  3. Non-vectorized approach (looping through millions of rows in R).  

**Optimization Strategy**  
1. Eliminate expensive key matching: precompute integer mapping of `(id, year)` → row index using vectorized join (e.g., `data.table` keyed operations).  
2. Represent neighbor relationships once as an integer matrix (mapping row → neighbor rows) for all years via fast joins.  
3. Compute stats using matrix/vector operations rather than `lapply`.  
4. Use `data.table` groups to compute max/min/mean in a single grouped step.  
5. Avoid recalculating neighbor lookups for each variable.  

**Optimized Working R Code**  
```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute all (id, year) combos
id_order_dt <- data.table(id = id_order)

# Expand rook neighbors to a long table
rook_pairs <- data.table(
  from_id = rep(id_order, lengths(rook_neighbors_unique)),
  to_id   = unlist(rook_neighbors_unique)
)

# Expand to panel years (Cartesian join)
years <- unique(dt$year)
rook_pairs <- rook_pairs[, .(to_id, year = years), by = from_id]

# Map to row indices in dt
rook_pairs[, from_idx := dt[rook_pairs, .I, on = .(id = from_id, year)]]
rook_pairs[, to_idx   := dt[rook_pairs, .I, on = .(id = to_id, year)]]
rook_pairs <- rook_pairs[!is.na(from_idx) & !is.na(to_idx)]

# For each variable, compute neighbor stats in a single grouped step
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  # Pull val for neighbors
  vals <- dt[[v]]
  rook_pairs[, val := vals[to_idx]]
  
  # Aggregate max, min, mean per from_idx
  stats <- rook_pairs[
    !is.na(val), .(
      max_val = max(val, na.rm = TRUE),
      min_val = min(val, na.rm = TRUE),
      mean_val = mean(val, na.rm = TRUE)
    ), by = from_idx
  ]
  
  # Join back to dt
  dt[stats$from_idx, paste0(v, "_nb_max") := stats$max_val]
  dt[stats$from_idx, paste0(v, "_nb_min") := stats$min_val]
  dt[stats$from_idx, paste0(v, "_nb_mean") := stats$mean_val]
}

cell_data <- as.data.frame(dt)
```

**Why Faster?**  
- Avoids repeated list traversal: single join operation builds neighbor relationships for all years.  
- Single large aggregation per variable using highly optimized `data.table`.  
- Eliminates repeated `paste` calls and complex lists → integer indexing only.  
- Scales to millions of rows efficiently, likely dropping runtime from **86+ hours to a few hours** (or less on SSD-equipped laptop).  

**Preserves:**  
- Original rook topology.  
- Exact numerical estimands (max, min, mean).  
- Model compatibility (no retraining).