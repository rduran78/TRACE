 **Diagnosis**  
The bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation, hash lookups, and list indexing in R, which is extremely slow at this scale. The neighbor structure is static across years, but the code redundantly recomputes neighbor indices for every cell-year row. This results in ~6.46M × 5 passes through large lists, causing the 86+ hour runtime.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell (not per cell-year)** and reuse them across all years.
2. **Vectorize neighbor feature computation** using `data.table` or `dplyr` joins instead of per-row `lapply`.
3. **Avoid string concatenation for keys**; use integer indices for fast joins.
4. **Compute neighbor stats in a grouped manner**: melt data to long format, join neighbor relationships, aggregate with `max`, `min`, `mean`.

This reduces complexity from O(N × neighbors × years) in R loops to efficient grouped aggregation in C-optimized backends.

---

**Working R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Precompute neighbor pairs (cell_id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors, length = number of cells
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand neighbor pairs across all years
years <- unique(dt$year)
neighbor_pairs <- neighbor_pairs[, .(id, neighbor_id)][
  , .(id = rep(id, length(years)),
      neighbor_id = rep(neighbor_id, length(years)),
      year = rep(years, each = .N))
]

# Merge neighbor values for each variable and compute stats
compute_neighbor_features <- function(dt, neighbor_pairs, var_name) {
  # Join neighbor values
  merged <- merge(neighbor_pairs, dt[, .(neighbor_id = id, year, val = get(var_name))],
                  by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate stats by (id, year)
  agg <- merged[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Rename columns
  setnames(agg, c("id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  
  # Merge back to main dt
  dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  return(dt)
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  dt <- compute_neighbor_features(dt, neighbor_pairs, var_name)
}

# Convert back to data.frame if needed
cell_data <- as.data.frame(dt)
```

---

**Why This Works**  
- Eliminates per-row loops and string operations.
- Uses `data.table` joins and aggregations in C for speed.
- Precomputes neighbor relationships once and reuses them.
- Memory footprint is manageable: ~1.37M neighbor pairs × 28 years ≈ 38M rows, which fits in 16 GB RAM with `data.table`.

**Expected Runtime**: From 86+ hours to under 1 hour on a standard laptop.