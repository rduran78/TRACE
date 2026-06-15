 **Diagnosis:**  
The bottleneck arises from:  
1. `lapply` loops over ~6.46M rows → massive overhead.  
2. Repeated string-based lookups (`paste` keys, `setNames`, `idx_lookup`) → memory- and CPU-heavy.  
3. Nested list operations in `compute_neighbor_stats` create overhead due to `do.call(rbind, ...)`.  

Goal: Calculate neighbor statistic features (max, min, mean for 5 variables) for 6.46M rows efficiently within 16 GB RAM **without retraining the RF model**.  

---

### **Optimization Strategy**
- **Avoid full per-row R loops:** Use vectorized joins or `data.table` grouping.
- **Precompute neighbor relationships in a long form table** (source_id, target_id) so stats can be calculated via fast grouped summaries rather than lapply.
- **Leverage `data.table`** for aggregation to reduce RAM and speed up processing.
- Use integer IDs instead of string keys.
- **Compute all years in one go:** Use year as join key, no repeated string concatenation.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Assume `id_order` aligns with rook_neighbors_unique
# Build a long neighbor mapping (source -> neighbor)
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[i], nb_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand neighbor relationships across all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(src_id, nb_id)]

# Melt cell_data to long form for neighbor vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_vars_long <- melt(cell_data, measure.vars = neighbor_source_vars, 
                       variable.name = "var", value.name = "val")

# Join to get neighbor values
setkey(cell_vars_long, id, year)
neighbor_dt <- cell_vars_long[neighbor_dt, on = .(id = nb_id, year)]

# Aggregate neighbor stats (max, min, mean) per src_id-year-var
neighbor_stats <- neighbor_dt[!is.na(val),
                               .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                               by = .(id = src_id, year, var)]

# Reshape back to wide form by variable
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var, 
                             value.var = c("nb_max", "nb_min", "nb_mean"))

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- Single pass joins + grouped aggregations → replaces millions of small lapply calls.
- `data.table` handles 6.5M rows + neighbor expansions efficiently within 16 GB.
- Avoids expensive string manipulation on every row.
- Preserves original model and numerical meaning (same statistics computed).

**Expected Runtime:** Minutes to a couple of hours instead of 86+ hours, depending on disk speed and RAM.