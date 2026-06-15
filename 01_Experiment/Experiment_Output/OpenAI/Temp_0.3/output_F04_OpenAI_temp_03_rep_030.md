 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`), hash lookups, and list indexing, which is extremely costly at this scale. The neighbor relationships are static across years, yet the current code redundantly recomputes neighbor indices for every row-year combination. Additionally, `compute_neighbor_stats` repeatedly traverses lists and allocates small vectors, causing severe overhead.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per cell** and reuse them across all years instead of recomputing per row.
2. **Vectorize computations** using `data.table` or `dplyr` to avoid millions of small R function calls.
3. **Reshape data to wide or grouped form** by `(id, year)` for efficient joins and aggregations.
4. **Compute neighbor stats in bulk** using joins rather than nested loops.
5. **Avoid string concatenation** for lookups; use integer indexing instead.

---

**Working R Code (data.table approach)**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure id and year are integers
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Precompute neighbor pairs (id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors
neighbor_pairs <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand neighbor pairs across all years
years <- sort(unique(cell_data$year))
neighbor_pairs <- neighbor_pairs[, .(year = years), by = .(id, neighbor_id)]

# Merge to get neighbor values
# Keep only necessary columns for efficiency
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_small <- cell_data[, c("id", "year", vars), with = FALSE]

# Join neighbor values
setkey(cell_data_small, id, year)
setkey(neighbor_pairs, neighbor_id, year)
neighbor_data <- neighbor_pairs[cell_data_small, on = .(neighbor_id = id, year), nomatch = 0]

# neighbor_data now has: id, neighbor_id, year, ntl, ec, ...
# Compute stats by (id, year)
for (v in vars) {
  stats <- neighbor_data[, .(
    max = max(get(v), na.rm = TRUE),
    min = min(get(v), na.rm = TRUE),
    mean = mean(get(v), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Rename columns
  setnames(stats, c("id", "year", "max", "min", "mean"),
           c("id", "year", paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")))
  
  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added efficiently
```

---

**Why This Works**  
- Eliminates 6.46M `lapply` calls and string operations.
- Uses efficient keyed joins and grouped aggregations in `data.table`.
- Computes neighbor stats in bulk rather than row-by-row.
- Memory-friendly: processes only required columns and uses integer keys.

**Expected Performance**  
On a 16 GB laptop, this approach should reduce runtime from 86+ hours to **minutes or a few hours**, depending on I/O and CPU speed, without retraining the Random Forest and preserving the original estimand.