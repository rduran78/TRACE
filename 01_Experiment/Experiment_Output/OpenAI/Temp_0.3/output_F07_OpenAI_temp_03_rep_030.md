 **Diagnosis**  
The current approach is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups in R lists.  
- Neighbor statistics are computed row-wise instead of vectorized or batched.  
- Memory overhead from repeated `paste` and `setNames` calls.  
- No precomputation of neighbor-year indices; recomputation for every row and variable.  

**Optimization Strategy**  
1. **Precompute neighbor indices once for all years**: Expand the neighbor relationships to full panel upfront.  
2. **Use `data.table` for fast joins and aggregation**: Avoid per-row loops; compute max, min, mean using grouped operations.  
3. **Vectorize across variables**: Melt data long, join neighbors, aggregate, then dcast back.  
4. **Memory efficiency**: Work in chunks if needed, but `data.table` with keys should fit in 16 GB for ~6.5M rows.  
5. **Preserve model**: Do not retrain; only add new columns.  

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# 1. Build neighbor pairs (cell_id -> neighbor_id)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# 2. Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_panel <- neighbor_dt[, .(id = rep(id, each = length(years)),
                                  neighbor_id = rep(neighbor_id, each = length(years)),
                                  year = rep(years, times = .N))]

# 3. Merge neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_long <- melt(cell_data, id.vars = c("id", "year"), measure.vars = vars,
                       variable.name = "var", value.name = "value")

neighbor_panel_long <- merge(neighbor_panel, cell_data_long,
                             by.x = c("neighbor_id", "year"),
                             by.y = c("id", "year"),
                             allow.cartesian = TRUE)

# 4. Compute neighbor stats by (id, year, var)
neighbor_stats <- neighbor_panel_long[, .(
  neigh_max = max(value, na.rm = TRUE),
  neigh_min = min(value, na.rm = TRUE),
  neigh_mean = mean(value, na.rm = TRUE)
), by = .(id, year, var)]

# Replace NaN with NA
for (col in c("neigh_max", "neigh_min", "neigh_mean")) {
  set(neighbor_stats, which(is.nan(neighbor_stats[[col]])), col, NA)
}

# 5. Reshape back to wide
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var,
                             value.var = c("neigh_max", "neigh_min", "neigh_mean"))

# 6. Merge back to original data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why this is efficient**  
- No per-row loops: All operations are vectorized with `data.table`.  
- Single join and aggregation instead of millions of small lookups.  
- Handles all variables in one pass.  

**Expected runtime**  
On 6.5M rows and ~1.3M neighbor pairs, this should complete in **minutes to a few hours** on a 16 GB laptop, versus 86+ hours previously.  

**Preserves**  
- Original rook-neighbor topology.  
- Exact max, min, mean per cell-year.  
- Trained Random Forest model remains unchanged.