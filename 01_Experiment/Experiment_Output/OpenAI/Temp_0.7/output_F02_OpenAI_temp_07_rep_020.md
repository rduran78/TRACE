 **Diagnosis**  
The current implementation is extremely slow and memory-heavy because:  
- `build_neighbor_lookup` creates a huge list of length 6.46M with repeated string concatenations and lookups.  
- `compute_neighbor_stats` repeatedly traverses large vectors using `lapply`, causing excessive R interpreter overhead and memory allocation.  
- Everything is computed in pure R loops; this is inefficient for 6.46M rows × 5 variables.  

---

**Optimization Strategy**  
1. **Avoid building massive lists in R**: Instead of precomputing a per-row neighbor index list, work at the *cell level* and then broadcast to years.  
2. **Use matrix/vectorized operations**: Precompute neighbor statistics per year using numeric indexing and `tapply`/`rowsum` or `data.table`.  
3. **Process year-by-year**: Reduces memory footprint dramatically.  
4. **Use `data.table` for speed**: Handles group operations efficiently.  
5. **Leverage sparse representation**: Store neighbor links as two integer vectors (from, to) and join instead of building nested lists.  

---

### **Optimized Approach (data.table)**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Ensure keys
setkey(cell_data, id, year)

# Neighbor edges: expand to all years once
# rook_neighbors_unique is list: each entry = neighbors of id_order[i]
edges <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Cross with years (Cartesian join)
years <- sort(unique(cell_data$year))
edges_full <- edges[, .(from, to, year = rep(years, each = .N))]

# Merge to attach variable values (for each neighbor)
compute_neighbor_stats_fast <- function(dt, var) {
  # Attach neighbor values
  vals <- dt[, .(id, year, v = get(var))]
  setkey(vals, id, year)
  e <- edges_full[vals, on = .(to = id, year)]
  
  # Now compute stats grouped by (from, year)
  stats <- e[!is.na(v), .(
    max_v = max(v),
    min_v = min(v),
    mean_v = mean(v)
  ), by = .(from, year)]
  
  # Merge back to main table
  setnames(stats, c("from","year","max_v","min_v","mean_v"),
           c("id","year",
             paste0(var,"_nb_max"),
             paste0(var,"_nb_min"),
             paste0(var,"_nb_mean")))
  dt[stats, on = .(id, year)]
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  cell_data <- compute_neighbor_stats_fast(cell_data, var)
}
```

---

**Why this works:**  
- `edges_full` acts as a long table of neighbor relationships per year (≈ 1.3M × 28 ≈ 38M rows, but handled in chunks if needed).  
- Each join and aggregation is in C via `data.table`, vastly faster than R loops.  
- No giant list objects; everything is columnar.  
- Can further chunk by year if memory is tight: process one year at a time and `rbindlist`.  

**Expected speed-up:** From 86+ hours to a few hours or less on a laptop.  

**Memory tip:** If 38M rows is too large, process year-by-year:
```r
result_list <- vector("list", length(years))
for (i in seq_along(years)) {
  y <- years[i]
  e_year <- edges[, .(from, to)]
  vals <- cell_data[year == y, .(id, v1 = ntl, v2 = ec, ...)]
  # Compute per var in this subset
}
cell_data <- rbindlist(result_list)
```

This preserves the trained Random Forest model and the original estimand, while making the pipeline practical on a 16 GB machine.