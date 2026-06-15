 ### **Diagnosis**
- The neighbor graph (rook adjacency) is **static across years**, but the current implementation builds metrics for every one of the ~6.46M cell-year rows individually, repeatedly scanning neighbors for each row and variable.
- This results in **28× redundant computation** (since neighbor structure doesn’t change) and excessive use of `lapply` across millions of rows.
- Each call to `compute_neighbor_stats` iterates over all rows and neighbors sequentially in **R loops** → extremely slow and memory-inefficient.

---

### **Optimization Strategy**
1. **Precompute neighbor index once per cell**, not per cell-year.  
   Use that static structure to aggregate stats year by year with **vectorized or grouped operations**.
2. **Avoid nested lapply for 6.46M rows** – instead, compute for all cells simultaneously per year.
3. Use **data.table** or **fast vectorized joins** for grouping and aggregation.
4. Maintain numerical equivalence by applying the same max, min, mean definitions, handling `NA`s the same way.

---

### **Optimized Working Code**

```r
library(data.table)

# Convert to data.table for fast operations
setDT(cell_data)

# Build static neighbor mapping (list: cell_id -> neighbor_ids)
# id_order: vector of unique cell ids in adjacency order
nb_lookup <- setNames(rook_neighbors_unique, id_order)

# Convert to a long format mapping: cell_id, neighbor_id
neighbor_dt <- rbindlist(lapply(names(nb_lookup), function(k) {
  if (length(nb_lookup[[k]]) == 0) return(NULL)
  data.table(cell_id = as.integer(k), neighbor_id = as.integer(id_order[nb_lookup[[k]]]))
}))

# Key tables
setkey(cell_data, id)  # assumed 'id' in cell_data
setkey(neighbor_dt, cell_id)

# Prepare output structure once
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# For each year, compute neighbor features
results_list <- vector("list", length(neighbor_source_vars) * 3)  # max, min, mean
names(results_list) <- unlist(lapply(neighbor_source_vars, function(v) {
  paste(v, c("neighbor_max", "neighbor_min", "neighbor_mean"), sep = "_")
}))

# Initialize columns in cell_data
for (nm in names(results_list)) set(cell_data, j = nm, value = NA_real_)

years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for the year
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  
  # Join neighbors: replicate rows for each neighbor
  joined <- neighbor_dt[dt_year, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined: cell_id | neighbor_id | year | vars...
  
  # For each variable, aggregate neighbor stats by cell_id
  for (var_name in neighbor_source_vars) {
    stats <- joined[ , .(
      max_val = if (.N==0) NA_real_ else max(get(var_name), na.rm = TRUE),
      min_val = if (.N==0) NA_real_ else min(get(var_name), na.rm = TRUE),
      mean_val = if (.N==0) NA_real_ else mean(get(var_name), na.rm = TRUE)
    ), by = .(cell_id)]
    
    # Update cell_data for that year
    cell_data[year == yr & id %in% stats$cell_id,
              paste0(var_name, "_neighbor_max") := stats$max_val[match(id, stats$cell_id)]]
    cell_data[year == yr & id %in% stats$cell_id,
              paste0(var_name, "_neighbor_min") := stats$min_val[match(id, stats$cell_id)]]
    cell_data[year == yr & id %in% stats$cell_id,
              paste0(var_name, "_neighbor_mean") := stats$mean_val[match(id, stats$cell_id)]]
  }
}

# The Random Forest model can now be used on the augmented `cell_data`
```

---

### **Why This Is Fast**
- **Static neighbor graph** handled once as `neighbor_dt`.
- Work done **per year (28x)** instead of per cell-year row.
- Aggregations done in **compiled C code inside data.table** instead of millions of R loops.
- Memory footprint much smaller: joining only the rows of relevant year at any time.

---

### **Expected Performance Gain**
- From 86+ hours → likely **minutes to a couple hours** on a 16 GB laptop for 6.46M rows because:
  - Vectorized group aggregation eliminates redundant loops.
  - `allow.cartesian` join + fast indexed lookups drastically reduce overhead.

**Trained Random Forest remains untouched, and estimand is preserved.**