 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M cell-year rows, `build_neighbor_lookup` constructs neighbor indices dynamically by string concatenation and repeated lookups. This is repeated for every variable in the outer loop, causing massive redundant computation. Additionally, `compute_neighbor_stats` repeatedly scans large vectors. The complexity is roughly *O(N × neighbors × vars)* with heavy overhead in R lists and string operations.

**Optimization Strategy**  
- Build a **single reusable neighbor lookup table** that maps each `(cell_id, year)` row to its neighbors’ row indices.
- Compute neighbor statistics in a **vectorized or batched manner** rather than per-row loops.
- Avoid repeated string concatenation and repeated joins by precomputing a numeric index mapping for all `(id, year)` combinations.
- Use `data.table` for efficient joins and aggregation.
- Memory: 6.46M rows × 5 variables × 3 stats is feasible on 16 GB if processed efficiently.

---

### **Optimized Approach**
1. Precompute a `data.table` with columns: `row_id`, `id`, `year`.
2. Build a long-format neighbor table: `(row_id, neighbor_row_id)` using numeric indices only.
3. Join yearly attributes onto this neighbor table once per variable and compute `max`, `min`, `mean` using `data.table` group operations.
4. Merge results back to the main table.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
cell_data[, row_id := .I]

# Precompute neighbor pairs (static across years)
# rook_neighbors_unique: list of integer vectors (neighbors for each id_order position)
id_order <- unique(cell_data$id)
id_to_idx <- setNames(seq_along(id_order), id_order)

neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand neighbor pairs across years
years <- unique(cell_data$year)
neighbor_pairs_full <- CJ(year = years, id = id_order)[
  , .(id, year)
][neighbor_pairs, on = "id", allow.cartesian = TRUE]

# Map to row indices
neighbor_pairs_full <- neighbor_pairs_full[
  cell_data[, .(id, year, row_id)], on = .(id, year)
][
  cell_data[, .(neighbor_id = id, year, neighbor_row_id = row_id)],
  on = .(neighbor_id, year)
][!is.na(row_id) & !is.na(neighbor_row_id),
  .(row_id, neighbor_row_id)
]

# Now neighbor_pairs_full has (row_id, neighbor_row_id)
# Compute neighbor stats for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[, .(neighbor_row_id = row_id, val = get(var_name))]
  tmp <- neighbor_pairs_full[vals, on = "neighbor_row_id"]
  stats <- tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = row_id]
  
  # Merge back
  setnames(stats, c("row_id", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data <- stats[cell_data, on = "row_id"]
}

# Drop helper column
cell_data[, row_id := NULL]
```

---

### **Why This Is Faster**
- Neighbor relationships are computed **once** and reused for all variables.
- Eliminates per-row string concatenation and repeated lookups.
- Uses `data.table` joins and group operations, which are highly optimized in C.
- Memory footprint is controlled by processing variable-by-variable.

---

**Expected Runtime Improvement:** From 86+ hours to a few hours (or less) on a standard laptop, depending on disk I/O and CPU speed, because the algorithm now scales linearly with data size and avoids redundant operations.

**Preserves:**  
- Original Random Forest model (not retrained).  
- Original estimand (same neighbor-based features, just computed efficiently).