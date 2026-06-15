 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and list indexing, which is extremely inefficient at this scale. The neighbor relationships are static across years, yet the current implementation recomputes neighbor keys for every row-year combination. This results in massive redundant work and memory overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices by cell ID only** (not by year) and reuse them across all years.
2. **Vectorize operations** using `data.table` or matrix-based joins instead of per-row `lapply`.
3. **Avoid string concatenation for lookups**; use integer indexing.
4. **Compute neighbor stats in a grouped manner**: reshape data to wide or use `data.table` joins by year.
5. **Parallelize** where possible using `future.apply` or `data.table` multithreading.

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup by cell ID (static)
id_to_ref <- setNames(seq_along(id_order), id_order)
neighbor_list <- lapply(seq_along(id_order), function(i) id_order[rook_neighbors_unique[[i]]])
names(neighbor_list) <- id_order

# Melt data for fast joins
cell_data_long <- cell_data[, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]

# Prepare neighbor pairs (cell_id -> neighbor_id)
neighbor_dt <- rbindlist(lapply(names(neighbor_list), function(cid) {
  data.table(id = cid, neighbor_id = neighbor_list[[cid]])
}))

# Join on year to get neighbor values
setkey(cell_data_long, id, year)
setkey(neighbor_dt, neighbor_id)

compute_neighbor_features <- function(var_name) {
  # Join neighbor values for the variable
  joined <- neighbor_dt[cell_data_long, on = .(neighbor_id = id), allow.cartesian = TRUE]
  # joined now has: id (source), neighbor_id, year, var_name
  stats <- joined[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  # Merge back to original data
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                    paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  cell_data_long[stats, on = .(id, year)]
}

# Apply for all neighbor source variables
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  cell_data_long <- compute_neighbor_features(var_name)
}

# Convert back to original structure if needed
cell_data <- cell_data_long
```

---

**Why This Works**  
- Eliminates per-row string operations.
- Uses `data.table` joins and group aggregation, which are highly optimized in C.
- Avoids redundant neighbor computation across years.
- Scales well for millions of rows and can leverage multithreading.

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop, depending on disk I/O and CPU cores.