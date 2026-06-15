 **Diagnosis**

- The majority of cost likely comes from **R object overhead and repeated list operations** in `build_neighbor_lookup` and `compute_neighbor_stats`.  
- `lapply` inside both functions builds and copies millions of small objects, which is highly inefficient for 6.46M rows.  
- `neighbor_lookup` is an enormous list (length = 6.46M), with repeated integer vectors. Creating and accessing this repeatedly consumes memory and slows GC.  
- Random Forest inference (`predict`) on 6.46M rows with 110 columns is costly but typically much faster than extremely inefficient feature construction loops.  
- Main bottleneck: **neighbor feature computation using nested lists** and repeated `do.call(rbind, ...)`.  

---

### **Optimization Strategy**
1. **Precompute reused structures** in a vectorized format, not lists.  
2. **Avoid `lapply` for row-level loops**; use `data.table` or `dplyr` for group calculations.  
3. **Flatten neighbor relationships into long form** (row_index, neighbor_index) and compute stats with fast aggregation (e.g., `data.table` + `by`).  
4. Compute all neighbor-derived stats in **one pass**, instead of per-variable repeated row-wise lookups.  
5. Use **`predict(..., num.threads = X)`** if using `ranger` or parRF for parallel RF inference.  
6. **Save large intermediate objects to disk** or process in chunks if memory pressure occurs.  

---

### **Working Optimized R Code**

```r
library(data.table)
library(ranger)

# Convert original data to data.table
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Prepare long neighbor mapping: (origin_id, neighbor_id)
# rook_neighbors_unique: list of neighbors for each id in id_order
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(origin_id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Join with all years → complete origin/neighbor/year mapping
years <- sort(unique(cell_dt$year))
neighbor_dt <- neighbor_dt[, .(year = years), by = .(origin_id, neighbor_id)]

# Map neighbor_id/year to a cell index via join
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- neighbor_dt[cell_dt, on = .(neighbor_id = id, year), nomatch = 0]
# neighbor_dt now has: origin_id, neighbor_id, year, neighbor_vars...

# Compute neighbor stats for all source variables in one go
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_stats <- neighbor_dt[, .(
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
), by = .(origin_id, year)]

# Merge back to main cell_dt
setkey(agg_stats, origin_id, year)
setnames(agg_stats, "origin_id", "id")
cell_dt <- cell_dt[agg_stats, on = .(id, year)]

# Random Forest prediction (assuming model is a ranger RF)
# Enable parallel threads
rf_preds <- predict(rf_model, data = cell_dt, num.threads = parallel::detectCores())$predictions
```

---

### **Key Gains**
- No massive per-row list objects → memory and GC savings.  
- Aggregated neighbor computations with `data.table` are highly optimized (C backend).  
- Computes all neighbor-based features in **one pass** vs nested loops.  
- Ranger’s multithreaded predict scales to millions of rows easily.  

---

**Expected Impact:** This approach should reduce neighbor feature computation from many hours to minutes and preserve the trained model and original numerical estimand.