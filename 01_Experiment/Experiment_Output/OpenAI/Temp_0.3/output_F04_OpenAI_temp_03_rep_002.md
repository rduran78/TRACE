 **Diagnosis**  
The main bottleneck is the repeated, row-wise neighbor lookup and aggregation across ~6.46M rows. Specifically:  
- `build_neighbor_lookup` constructs a list of neighbor indices for every row (cell-year), which is extremely costly because it repeats neighbor ID matching for each year.  
- `compute_neighbor_stats` then iterates over this large list multiple times (once per variable), causing redundant computations.  
- Both functions use `lapply` over millions of elements, which is inefficient in R.  

The Random Forest inference is not the issue; the spatial feature construction dominates runtime.

---

**Optimization Strategy**  
1. **Precompute neighbor relationships at the cell level only once**, then reuse them across years.  
2. **Vectorize aggregation**: Instead of looping over rows, join data to its neighbors and compute stats using `data.table` or `dplyr`.  
3. **Process all variables in one pass** to avoid repeated neighbor lookups.  
4. **Use efficient joins**: Expand neighbor pairs across years, merge with data, and compute grouped summaries.  

This reduces complexity from *O(n × neighbors × variables)* to roughly *O(n × neighbors)* with vectorized aggregation.

---

**Optimized R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs at cell level
# rook_neighbors_unique: list where each element i contains neighbors of id_order[i]
neighbor_pairs <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand neighbor pairs across all years
years <- unique(cell_data$year)
neighbor_pairs_expanded <- neighbor_pairs[, .(id = from, neighbor_id = to), by = years]
setnames(neighbor_pairs_expanded, "years", "year")

# Merge to get neighbor values
neighbor_dt <- merge(
  neighbor_pairs_expanded,
  cell_data[, .(neighbor_id = id, year, ntl, ec, pop_density, def, usd_est_n2)],
  by = c("neighbor_id", "year"),
  all.x = TRUE
)

# Compute neighbor stats for each variable in one pass
stats_dt <- neighbor_dt[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Replace NaN (from all NA neighbors) with NA
for (j in names(stats_dt)) {
  set(stats_dt, which(is.nan(stats_dt[[j]])), j, NA)
}

# Merge back to cell_data
cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works**  
- `neighbor_pairs` is computed once for ~1.37M relationships.  
- Expansion across 28 years creates ~38M rows, but operations are fully vectorized and memory-efficient with `data.table`.  
- All neighbor stats for all variables are computed in a single grouped aggregation.  
- No repeated `lapply` over millions of rows.  

**Expected runtime**: From 86+ hours to under an hour on a 16 GB laptop (depending on I/O).  

**Preserves**:  
- Original numerical estimand (same neighbor definitions, same stats).  
- Trained Random Forest model remains untouched.