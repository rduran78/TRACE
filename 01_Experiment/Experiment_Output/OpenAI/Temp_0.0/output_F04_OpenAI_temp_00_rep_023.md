 **Diagnosis**  
The bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each row recomputes neighbor indices and stats in pure R lists, causing massive overhead. The neighbor structure is static across years, so recomputing per row is unnecessary. Additionally, `lapply` + `do.call(rbind, ...)` on millions of elements is extremely slow and memory-inefficient.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** (not per cell-year).
2. **Vectorize across years**: Expand neighbor relationships for all years using fast joins instead of nested loops.
3. Use **`data.table`** for efficient grouping and aggregation.
4. Compute all neighbor stats in a single pass per variable using `data.table` joins rather than repeated `lapply`.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs (cell_id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand for all years
years <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(id = rep(id, length(years)),
                                     neighbor_id = rep(neighbor_id, length(years)),
                                     year = rep(years, each = .N))]

# Merge with cell_data to get neighbor values
setkey(cell_data, id, year)
setkey(neighbor_pairs, neighbor_id, year)

# Function to compute and add neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Join neighbor values
  tmp <- neighbor_pairs[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
  tmp[, neighbor_val := get(var_name)]
  
  # Aggregate stats by (id, year)
  stats <- tmp[!is.na(neighbor_val),
               .(max_val = max(neighbor_val),
                 min_val = min(neighbor_val),
                 mean_val = mean(neighbor_val)),
               by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data[stats, on = .(id, year), (names(stats)[3:5]) := mget(names(stats)[3:5])]
}

# Apply for all neighbor source variables
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  compute_neighbor_features(var_name)
}
```

---

**Why This Works**  
- Eliminates per-row `lapply` loops.
- Uses `data.table` joins and group aggregation, which are highly optimized in C.
- Precomputes neighbor relationships once and reuses them for all variables.
- Memory footprint is controlled by processing one variable at a time.

---

**Expected Performance**  
This approach reduces complexity from O(N × neighbors × years) in R loops to efficient vectorized joins and aggregations. On a 16 GB laptop, this should cut runtime from 86+ hours to a few hours or less, depending on disk I/O and CPU speed, without changing the trained Random Forest model or the estimand.