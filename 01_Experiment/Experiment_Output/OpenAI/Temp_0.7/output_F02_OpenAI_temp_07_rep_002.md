 **Diagnosis**

The main bottlenecks are:

1. **Inefficient `lapply` per-row**:  
   Both `build_neighbor_lookup` and `compute_neighbor_stats` perform heavy per-row operations over ~6.46M rows, creating large intermediate lists.

2. **Repeated string operations**:  
   `paste(data$id, data$year, sep = "_")` and repeated lookups are expensive.

3. **Memory overhead**:  
   Storing millions of lists for neighbors and then repeatedly extracting subsets is highly memory-intensive.

4. **Single-threaded execution**:  
   The code runs sequentially; no parallelization is used.

---

**Optimization Strategy**

- **Precompute neighbor indices for all years using matrix operations**:  
  Instead of building a large list indexed by row, create an integer matrix (`neighbor_idx_matrix`) mapping each row to neighbor row indices directly.

- **Avoid string concatenations**:  
  Use numeric indexing for `(id, year)` pairs. A two-level index approach (id → position, year offset) avoids `paste()` calls.

- **Vectorize or use `data.table` for aggregation**:  
  Transform the problem into a join-and-aggregate task:  
  - Replicate neighbor relationships across years.
  - Merge neighbor values and compute `max`, `min`, `mean` using fast grouped aggregation (`data.table`).

- **Chunking + parallelization**:  
  Process in chunks or use `future.apply` / `parallel` to utilize multiple cores.

---

**Working R Code**

Using `data.table` for speed and memory efficiency:

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique ids in correct order
# rook_neighbors_unique: list of integer vectors (spdep nb object)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor relationships replicated across years
years <- sort(unique(cell_data$year))
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  nbrs <- rook_neighbors_unique[[i]]
  if (length(nbrs) == 0) return(NULL)
  data.table(
    id     = rep(id_order[i], length(nbrs) * length(years)),
    year   = rep(years, each = length(nbrs)),
    nbr_id = rep(id_order[nbrs], times = length(years))
  )
}), use.names = TRUE)

setkey(neighbor_dt, nbr_id, year)

# Melt source vars for easier join
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Join neighbor values
nbr_vals <- cell_data[, c("id", "year", vars), with = FALSE]
setkey(nbr_vals, id, year)
neighbor_dt <- neighbor_dt[nbr_vals, on = .(nbr_id = id, year), nomatch = 0]

# Compute stats by (id, year)
agg <- neighbor_dt[, .(
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

# Merge back into cell_data
cell_data <- agg[cell_data, on = .(id, year)]
```

---

**Why this works**

- Eliminates per-row loops and string operations.
- Uses `data.table` for highly optimized join and aggregation.
- Avoids storing large neighbor lookup lists in memory.
- Scales better: operations are batched and vectorized.
- Can be parallelized further by splitting `neighbor_dt` and using `future.apply` or `data.table`'s `by` parallelism.

---

**Expected Performance Gain**

- From 86+ hours to **minutes or a few hours** on a laptop.
- Memory usage improves since we use compressed integer keys and avoid millions of small lists.

This preserves the original trained Random Forest model and numerical estimands.