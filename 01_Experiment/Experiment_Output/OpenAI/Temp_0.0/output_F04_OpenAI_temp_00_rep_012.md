 **Diagnosis**  
The bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each row recomputes neighbor indices and stats in pure R lists, causing massive overhead. The neighbor structure is static across years, so recomputing per row is unnecessary. Additionally, `lapply` + `do.call(rbind, ...)` on millions of elements is extremely slow and memory-inefficient.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per cell-year).
2. **Vectorize across years**: Expand neighbor relationships to all years using integer indexing rather than string keys.
3. **Use `data.table` for fast joins and aggregation** instead of nested `lapply`.
4. **Compute all neighbor stats in a single grouped operation** rather than looping over variables.
5. **Parallelize** if possible (optional).

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as integer indices per cell
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- lapply(rook_neighbors_unique, function(nbs) id_order[nbs])

# Expand neighbor relationships for all years
years <- sort(unique(cell_data$year))
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(neighbor_list[[i]]) == 0) return(NULL)
  data.table(
    id = rep(id_order[i], each = length(neighbor_list[[i]]) * length(years)),
    year = rep(years, each = length(neighbor_list[[i]]), times = 1),
    neighbor_id = rep(neighbor_list[[i]], times = length(years))
  )
}))

# Join neighbor values for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_dt <- merge(neighbor_dt, cell_data[, c("id", "year", vars), with = FALSE],
                     by.x = c("neighbor_id", "year"), by.y = c("id", "year"),
                     all.x = TRUE)

# Compute neighbor stats grouped by (id, year)
agg_dt <- neighbor_dt[, .(
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

# Merge back into main data
cell_data <- merge(cell_data, agg_dt, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works**  
- Eliminates per-row string concatenation and repeated lookups.
- Uses `data.table` joins and grouped aggregation, which are highly optimized in C.
- Processes all variables in one pass instead of looping.
- Memory footprint is manageable because neighbor_dt size ≈ 6.46M * avg_neighbors (~4–5), which fits in 16 GB with efficient types.

**Expected Speedup**: From 86+ hours to under 1 hour on a standard laptop.