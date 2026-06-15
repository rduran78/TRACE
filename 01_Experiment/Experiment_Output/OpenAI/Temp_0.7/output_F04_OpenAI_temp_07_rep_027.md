 **Diagnosis**  
The bottleneck is the nested `lapply` operations in `build_neighbor_lookup` and `compute_neighbor_stats`. For ~6.46M rows, repeatedly performing string concatenation, named lookups, and list allocations is extremely inefficient and memory-heavy. The neighbor structure is static across years, yet the code recomputes key-based lookups for every row. Additionally, `rbind` on millions of elements via `do.call` is slow.

---

**Optimization Strategy**  
1. **Precompute persistent neighbor index mapping once** and reuse across all years.
2. **Vectorize operations with `data.table`** instead of millions of `lapply` calls.
3. **Avoid string concatenation per row**—use numeric joins on `(id, year)`.
4. **Compute all neighbor statistics in one pass per variable** using joins and grouped aggregation.
5. **Parallelize aggregations** where possible.

---

**Optimized Approach**  
- Convert `cell_data` to `data.table`.
- Expand neighbor relationships across years by joining on year rather than iterating row-wise.
- Aggregate neighbor values in bulk using `by = .(id, year)`.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Extract neighbor pairs (id -> neighbor_id)
# rook_neighbors_unique: list of neighbors indexed by id_order
nb_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Expand neighbor pairs across years
years <- sort(unique(cell_data$year))
nb_pairs <- nb_pairs[, .(year = years), by = .(id, neighbor_id)]

# Melt cell_data to long format for selected variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# For efficiency, create a lookup for each variable
for (var_name in neighbor_source_vars) {
  dt_var <- cell_data[, .(neighbor_id = id, year, value = get(var_name))]

  # Join neighbor values
  nb_data <- merge(nb_pairs, dt_var, by = c("neighbor_id", "year"), all.x = TRUE)

  # Compute stats by (id, year)
  stats <- nb_data[, .(
    max_val = if (.N > 0) max(value, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(value, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]

  # Merge back into cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# Result: cell_data now has neighbor-based features
```

---

**Why This Works Faster**  
- Removes per-row loops and string concatenations.
- Processes all neighbor relationships in bulk, leveraging efficient joins and grouped aggregation in `data.table`.
- Memory-friendly and scalable for millions of rows.
- Preserves original estimand and works with the already-trained Random Forest model.

**Expected Performance Gain**  
On a 16 GB laptop, this approach should reduce runtime from **86+ hours to a few hours or less**, depending on I/O speed and available threads. Further speedup possible with `setDTthreads()` for parallel aggregation.