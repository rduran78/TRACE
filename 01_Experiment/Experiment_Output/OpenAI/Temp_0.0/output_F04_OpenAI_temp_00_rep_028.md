 **Diagnosis**  
The bottleneck is the repeated use of `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation, hash lookups, and list indexing in R, which is extremely slow at this scale. The neighbor relationships are static across years, but the current code recomputes lookups for every row. Additionally, `compute_neighbor_stats` repeatedly traverses lists and allocates small vectors, causing massive overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per cell-year).  
2. **Vectorize across years**: Expand neighbor indices for all years using matrix operations instead of per-row loops.  
3. **Use `data.table` for fast joins and aggregation** instead of `lapply`.  
4. **Avoid string concatenation**: Use integer keys for `(id, year)` mapping.  
5. **Compute all neighbor stats in one pass** for all variables using grouped operations.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as integer indices (per cell)
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
}), use.names = TRUE)

setkey(neighbor_dt, neighbor_id, year)

# Join neighbor values for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), (vars) := mget(vars)]

# Compute stats by (id, year)
agg_exprs <- lapply(vars, function(v) {
  list(
    as.name(paste0(v, "_nb_max")) := max(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nb_min")) := min(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nb_mean")) := mean(get(v), na.rm = TRUE)
  )
})

agg_dt <- neighbor_dt[, eval(as.call(c(quote(list), agg_exprs))), by = .(id, year)]

# Merge back to cell_data
cell_data <- agg_dt[cell_data, on = .(id, year)]

```

---

**Why This Works**  
- Eliminates per-row loops (replaced with vectorized joins and aggregations).
- Uses `data.table` for efficient memory and speed.
- Computes all neighbor stats in a single grouped aggregation.
- Preserves the original estimand and Random Forest model.

**Expected Performance**  
On 6.46M rows and ~1.37M neighbor links, this approach should reduce runtime from 86+ hours to under 1 hour on a 16 GB laptop.