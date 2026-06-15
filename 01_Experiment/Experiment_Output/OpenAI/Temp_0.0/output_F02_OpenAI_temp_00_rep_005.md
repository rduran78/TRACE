 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length ≈ 6.46M (one per cell-year), each containing neighbor indices. This is expensive in both memory and time.  
- `compute_neighbor_stats` repeatedly traverses these lists and subsets vectors, causing heavy R-level overhead.  
- The process is purely interpreted R loops (`lapply`), which is inefficient for millions of iterations.  
- The neighbor structure is static across years, but the code redundantly recomputes lookups for every row.  

**Optimization Strategy**  
1. **Avoid per-row lists**: Instead of building a 6.46M-element list, use a long-format edge table (cell-year → neighbor-year) and compute stats via `data.table` or `dplyr` group operations.  
2. **Exploit static neighbors**: Neighbors depend only on cell IDs, not years. Precompute neighbor pairs once, then replicate across years.  
3. **Vectorize aggregation**: Use `data.table` joins and `by` grouping to compute max/min/mean in bulk.  
4. **Memory efficiency**: Process one variable at a time, avoid large intermediate lists.  
5. **Parallelization**: Use `data.table`’s multithreading or `future.apply` for parallel aggregation.  

**Working R Code (data.table approach)**  
```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in order
# rook_neighbors_unique: list of integer vectors (neighbors per cell)

# 1. Build neighbor edge table (cell_id -> neighbor_id)
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# 2. Expand across years
years <- unique(cell_data$year)
edges <- edges[CJ(year = years), on = .(dummy = NULL)]
setnames(edges, "year", "year")  # edges now has id, neighbor_id, year

# 3. Join neighbor values for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- merge(edges, cell_data[, .(neighbor_id = id, year, val = get(var_name))],
               by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate stats by (id, year)
  stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor-based features added
```

**Why this is faster**  
- Eliminates 6.46M-element lists and repeated subsetting.  
- Uses efficient `data.table` joins and group aggregation in C.  
- Memory footprint is reduced because we store edges once and reuse them.  
- Scales well with multithreading (`setDTthreads()` can be used).  

**Expected performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to a few hours or less, depending on disk I/O and CPU cores.  

**Preserves**  
- Original estimand (same neighbor stats).  
- Trained Random Forest model (unchanged).  
- Numerical integrity of features.