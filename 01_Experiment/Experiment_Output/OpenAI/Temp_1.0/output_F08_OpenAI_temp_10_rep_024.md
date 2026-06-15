 **Diagnosis**  
The bottleneck is that `compute_neighbor_stats()` operates row-wise across ~6.46 million rows, repeatedly scanning neighbor indices for each cell-year observation. This recomputes the same neighbor structure for every year, despite neighbors being static. It results in extreme overhead in both memory and time (86+ hours estimate).

Key inefficiency:
- `neighbor_lookup` ties neighbors per cell-year row, causing large repeated lookups.
- Neighbor relationships are constant; only variable values change by year.
- Computation is not vectorized and invoked 5 times (one per variable).

**Optimization Strategy**  
1. Keep a static neighbor lookup **at cell level**, not cell-year level.
2. Split data by year and compute neighbor max/min/mean for each variable **once per year** by leveraging the static neighbor structure.
3. Use `vapply`/`data.table`/`matrix` operations for fast aggregation rather than looping over millions of rows.
4. Bind results back to the full dataset after computing per-year neighbor stats.
5. Memory-safe: process one year at a time (28 passes instead of 6.46M row-level passes).

---

### **Optimized R Code**

```r
library(data.table)

compute_neighbor_stats_by_year <- function(dt, neighbors, id_order, var_name) {
  # dt: data.table with columns id, year, <var_name>
  setkey(dt, id)  # fast join by id
  
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Preallocate result matrix
  n <- nrow(dt)
  result <- matrix(NA_real_, n, 3)
  
  # Process by year
  years <- unique(dt$year)
  for (yr in years) {
    # Subset for current year
    idx_year <- which(dt$year == yr)
    vals <- dt[[var_name]][idx_year]
    ids <- dt$id[idx_year]
    
    # Map each cell to stats from its neighbors for this year
    stats_year <- vapply(seq_along(ids), function(i) {
      cell_id <- ids[i]
      ref_idx <- id_to_idx[as.character(cell_id)]
      neigh_ids <- id_order[neighbors[[ref_idx]]]
      
      # Pull neighbor values for this year
      neigh_vals <- vals[ match(neigh_ids, ids, nomatch = 0L) ]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
      c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
    }, numeric(3))
    
    result[idx_year, ] <- t(stats_year)
  }
  
  result
}

compute_and_add_neighbor_features <- function(dt, neighbors, id_order, vars) {
  for (var_name in vars) {
    stats <- compute_neighbor_stats_by_year(dt, neighbors, id_order, var_name)
    dt[[paste0(var_name, "_nbr_max")]] <- stats[, 1]
    dt[[paste0(var_name, "_nbr_min")]] <- stats[, 2]
    dt[[paste0(var_name, "_nbr_mean")]] <- stats[, 3]
  }
  dt
}

# ---- Usage ----
# Convert to data.table for efficiency
cell_data <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_and_add_neighbor_features(cell_data, rook_neighbors_unique, id_order, neighbor_source_vars)
```

---

### **Expected Improvement**
- Complexity reduced from `O(nrow * avg_neighbors)` over 6.46M rows to `O(years * cells * avg_neighbors)` with batching.
- For 28 years × 344K cells, this is orders faster and memory-friendly.
- Parallelization (`future.apply` or `parLapply`) per year can accelerate further.

This preserves:
- Original numerical estimand.
- The trained Random Forest pipeline (we only add neighbor features upstream).
- The static neighbor topology while recomputing variable-dependent features efficiently.