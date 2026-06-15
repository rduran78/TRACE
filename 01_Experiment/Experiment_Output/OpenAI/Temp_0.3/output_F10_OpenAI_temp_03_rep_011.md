 **Diagnosis**  
The current pipeline is slow because:  
- `build_neighbor_lookup` constructs neighbor indices for every row, repeating work across 28 years.  
- `compute_neighbor_stats` uses `lapply` over millions of rows, repeatedly scanning neighbor indices.  
- No vectorization or efficient aggregation; memory overhead is large.  
- The neighbor graph is recomputed per row-year instead of reusing a static topology.  

**Optimization Strategy**  
- Precompute a static neighbor index for cells (graph topology).  
- Use vectorized operations with `data.table` for grouping and aggregation instead of nested `lapply`.  
- Compute neighbor statistics year-by-year using joins rather than looping over rows.  
- Avoid repeated concatenation of keys; rely on integer indices.  
- Preserve numerical equivalence by computing `max`, `min`, and `mean` exactly as before.  
- Keep the Random Forest model intact; only optimize feature engineering.  

---

### **Efficient Implementation in R**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (graph topology)
# rook_neighbors_unique: list of integer vectors (neighbors for each cell)
cell_ids <- id_order
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = cell_ids[i], nbr = cell_ids[rook_neighbors_unique[[i]]])
}))

# Set keys for fast joins
setkey(cell_data, id, year)
setkey(neighbor_pairs, src)

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(dt, var_name, neighbor_pairs) {
  # Join neighbor pairs with cell_data for same year
  expanded <- neighbor_pairs[dt, on = .(src = id), allow.cartesian = TRUE]
  expanded <- expanded[dt, on = .(nbr = id, year), nomatch = 0L,
                       .(src, year, val = get(var_name))]
  
  # Aggregate stats per (src, year)
  stats <- expanded[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(src, year)]
  
  # Merge back to original
  setnames(stats, c("src", "year"), c("id", "year"))
  dt[stats, paste0(var_name, "_nbr_max") := i.max_val, on = .(id, year)]
  dt[stats, paste0(var_name, "_nbr_min") := i.min_val, on = .(id, year)]
  dt[stats, paste0(var_name, "_nbr_mean") := i.mean_val, on = .(id, year)]
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_features(cell_data, var_name, neighbor_pairs)
}

# cell_data now has neighbor features; ready for Random Forest prediction
```

---

**Why This Is Efficient**  
- Graph topology (`neighbor_pairs`) is built once and reused.  
- Uses `data.table` joins and group aggregation instead of per-row loops.  
- Avoids constructing millions of keys; works with integers.  
- Memory footprint is reduced; computation scales linearly with rows and edges.  

**Expected Performance**  
- From 86+ hours to a few hours (or less) on 16 GB RAM laptop, depending on disk I/O.  
- Preserves numerical equivalence and Random Forest model integrity.