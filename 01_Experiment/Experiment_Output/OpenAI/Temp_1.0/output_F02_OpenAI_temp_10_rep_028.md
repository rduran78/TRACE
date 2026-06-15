 **Diagnosis**:  
The slowdown arises from (a) repeatedly using `lapply` over 6.46 million rows in pure R, (b) creating large lists of neighbor indices on the fly, and (c) processing each neighbor feature sequentially. This becomes highly memory- and compute-intensive given the size of the dataset. Pure R list operations are not efficient for this scale.

---

### **Optimization Strategy**
1. **Precompute and flatten neighbor relations**: Instead of building per-row neighbor lists, transform the neighbor lookup into a long two-column index (`from`, `to`) mapping cell-year observations.  
2. **Vectorized, join-based aggregation** (using `data.table`): Compute max, min, and mean at scale via grouped operations rather than per-row loops.  
3. **Memory optimization**: Use `data.table` for efficient in-memory manipulation and avoid large nested `lapply`.  
4. **Preserve the original estimand**: Matching by year ensures that the statistics for neighbors in the same year remain consistent.  

---

### **Working R Code (Optimized)**

```r
library(data.table)

# Convert input to data.table
dt <- as.data.table(cell_data)

# Create a flat neighbor table
# id_order: vector of cell IDs in the same index order as rook_neighbors_unique
# rook_neighbors_unique: list of integer vectors (neighbors), length = length(id_order)
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Add year replication: Cartesian join on years for panel structure
years <- sort(unique(dt$year))
neighbor_pairs <- neighbor_pairs[ , .(src, nbr, year = rep(years, each = .N)), by = .(src, nbr)]
setkey(neighbor_pairs, nbr, year)

# Prepare for join
setkey(dt, id, year)

# Merge to get neighbor values
neighbor_dt <- neighbor_pairs[dt, on = .(nbr = id, year), allow.cartesian = TRUE]
# neighbor_dt has src (cell), year, and neighbor’s variables via 'i.' prefix after merge

# Compute stats for each source cell-year and variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  agg <- neighbor_dt[!is.na(get(var_name)), .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(src, year)]

  # Merge back into main dt
  setnames(agg,
           c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  dt <- merge(dt, agg, by.x = c("id", "year"), by.y = c("src", "year"), all.x = TRUE)
}
```

---

### **Why This Is Faster**
- **No row-wise loops**: Everything is handled by `data.table` grouping and joins, which are implemented in C and optimized for speed.
- **Single neighbor structure flattening**: Avoids huge, nested lists and repeated lookups.
- **Vectorized computation**: Using `max`, `min`, `mean` in grouped operations is much faster than iterative computations.

---

**Expected Performance**: On 6.46M rows with ~1.4M neighbor links, this approach should complete in a few hours (depending on disk I/O and joins) instead of 86+ hours, and fit into 16GB RAM if processed sequentially for the 5 variables.

**Note**: Preserve column naming convention for downstream compatibility, and ensure `id` and `year` are indexed.