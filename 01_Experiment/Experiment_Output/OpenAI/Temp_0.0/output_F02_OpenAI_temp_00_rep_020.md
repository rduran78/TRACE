 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length ≈ 6.46M (one per cell-year), each storing neighbor indices. This is huge in memory and expensive to compute.  
- `compute_neighbor_stats` repeatedly traverses large lists and subsets vectors, causing heavy R-level overhead.  
- The process is purely in R loops (`lapply`), which is inefficient for millions of rows.  
- Repeated string concatenation (`paste`) and lookups in large named vectors are costly.  

**Optimization Strategy**  
1. **Avoid per-row neighbor lists**: Instead of building a 6.46M-element list, compute neighbor relationships in a long format (edges) and join.  
2. **Vectorize aggregation**: Use `data.table` or `dplyr` to compute neighbor statistics in a grouped manner.  
3. **Precompute static structures**: The neighbor graph is static across years; replicate edges by year efficiently.  
4. **Memory efficiency**: Work with integer IDs, avoid character keys, and use `data.table` for joins and aggregations.  
5. **Parallelization**: Use `data.table`’s multithreading for aggregation.  

**Working R Code (Optimized)**  
```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# Map cell id to index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build edge list (cell_id -> neighbor_id)
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Expand edges for all years
years <- sort(unique(cell_data$year))
edges_expanded <- edges[, .(id = rep(src, each = length(years)),
                             nbr = rep(nbr, each = length(years)),
                             year = rep(years, times = .N))]
# Key for fast join
setkey(edges_expanded, nbr, year)
setkey(cell_data, id, year)

# Join neighbor values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- cell_data[, .(nbr_val = get(var_name)), by = .(id, year)]
  setnames(tmp, "id", "nbr")
  edges_expanded <- tmp[edges_expanded, on = .(nbr, year)]
  
  # Compute stats by (id, year)
  stats <- edges_expanded[!is.na(nbr_val),
                           .(max = max(nbr_val),
                             min = min(nbr_val),
                             mean = mean(nbr_val)),
                           by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("max", "min", "mean"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added
```

**Why this is faster**  
- Eliminates 6.46M-element list and repeated `lapply`.  
- Uses `data.table` joins and group operations in C for speed.  
- Memory footprint is reduced by working in long format and avoiding character concatenation.  
- Scales well with multithreading on a standard laptop.  

**Expected performance**  
- From 86+ hours to a few minutes to an hour, depending on disk I/O and CPU cores.  
- Preserves original estimand and trained Random Forest model.