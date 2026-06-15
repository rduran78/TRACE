 **Diagnosis**  
The bottleneck is the repeated use of `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. Additionally, `compute_neighbor_stats` repeatedly subsets vectors and computes aggregates in pure R loops, which is inefficient for millions of rows.

**Optimization Strategy**  
1. **Precompute neighbor relationships once** in a long format table (cell_id, year, neighbor_id).
2. **Use vectorized joins and grouped aggregation** (via `data.table`) instead of per-row `lapply`.
3. **Avoid string concatenation for keys**; use integer IDs for fast joins.
4. **Compute all neighbor stats in one pass** for all variables, rather than looping over variables.

This approach reduces complexity from O(N × neighbors) in R loops to efficient C-level operations in `data.table`.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Map cell IDs to integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build neighbor pairs (cell_id -> neighbor_id)
neighbor_pairs <- data.table(
  cell_id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand to all years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[, .(year = years), by = .(cell_id, neighbor_id)]

# Merge neighbor values
cell_data[, cell_idx := id_to_idx[id]]
neighbor_dt[, neighbor_idx := id_to_idx[neighbor_id]]

# Join neighbor values for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_dt <- merge(neighbor_dt, cell_data[, c("cell_idx", "year", vars), with = FALSE],
                 by.x = c("neighbor_idx", "year"), by.y = c("cell_idx", "year"),
                 all.x = TRUE)

# Compute neighbor stats grouped by (cell_id, year)
agg_dt <- long_dt[, .(
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
), by = .(cell_id, year)]

# Merge back to main data
cell_data <- merge(cell_data, agg_dt, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)

# Clean up
cell_data[, cell_idx := NULL]
```

---

**Why this works:**  
- Eliminates per-row loops and string operations.
- Uses `data.table` joins and grouped aggregation in C for speed.
- Computes all neighbor stats in one grouped operation.
- Memory footprint is manageable: ~1.37M neighbor pairs × 28 years ≈ 38M rows in `long_dt`, which fits in 16 GB RAM with careful use of `data.table`.

**Expected runtime:**  
From 86+ hours to **under 1 hour** on a standard laptop, assuming efficient `data.table` operations.

**Preserves:**  
- Original estimand.
- Trained Random Forest model.
- All neighbor-based features.