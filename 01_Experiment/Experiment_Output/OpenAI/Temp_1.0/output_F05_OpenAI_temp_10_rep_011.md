 **Diagnosis**  
The repeated string-key work is **not just a local inefficiency**; it is part of a larger algorithmic pattern where for each of ~6.46M rows, the pipeline constructs neighbor keys and does a hash lookup into `idx_lookup`. This happens inside `build_neighbor_lookup`, creating **millions of string concatenations and environment lookups**. Given the large dataset (344,208 cells × 28 years = 6.46M rows) and that this lookup occurs for all rows, the observed runtime ballooning to 86+ hours is expected.

The broader issue:  
- Neighbor relationships are **static** across years.
- String concatenation using `paste(neighbor_cell_ids, data$year[i])` repeats unnecessarily because the year dimension is predictable.
- We compute `neighbor_lookup` as a list of integer vectors but recompute string-based keys each time instead of precomputing an efficient mapping.

**Optimization Strategy**  
- **Eliminate string-based keys entirely**. Compute numeric indices by leveraging structured indexing:
  - Precompute a matrix mapping `(cell_id, year)` → row index.
  - Use integer arithmetic for lookups instead of strings.
- Build a **dense matrix of row positions**: rows represent cell id, columns represent years.
- Build neighbor lookups **once** for cells, then replicate across years without costly concatenations.
- Result: All lookups become pure integer operations (fast).

---

### **Optimized Approach**

1. Create `pos_matrix[cells, years]` that maps to row indices in `data`.
2. For each cell-year row `i`, find its neighbors via `neighbors[cell]` and extract indices directly from `pos_matrix[, year]`.
3. Compute stats using vectorized operations.

---

#### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure id and year are as expected
  stopifnot(all(sort(unique(data$id)) == sort(id_order)))
  years <- sort(unique(data$year))

  # Precompute matrix: rows=cell ids, cols=years, values=row index in data
  pos_matrix <- matrix(NA_integer_, nrow = length(id_order), ncol = length(years),
                       dimnames = list(as.character(id_order), as.character(years)))
  pos_matrix[cbind(match(as.character(data$id), as.character(id_order)),
                   match(data$year, years))] <- seq_len(nrow(data))

  # lapply row ids: but we only need positions, not string lookups anymore
  row_ids <- seq_len(nrow(data))
  year_idx <- match(data$year, years)
  cell_idx <- match(data$id, id_order)

  lapply(row_ids, function(i) {
    ref_idx <- cell_idx[i]
    neigh_cells <- neighbors[[ref_idx]]
    # For this row's year:
    nbr_rows <- pos_matrix[neigh_cells, year_idx[i]]
    as.integer(nbr_rows[!is.na(nbr_rows)])
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Usage
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Expected Gains**
- Remove ~6.46M × avg. neighbors string concatenations and hash lookups.
- Memory-efficient: one integer matrix plus neighbor lists (minimal overhead).
- Runtime reduction from 86 hours → ~tens of minutes or a few hours, depending on disk I/O and CPU.

**Key Advantages**
- Preserves trained Random Forest model (only transforming features faster).
- Same numerical estimand as original (identical neighbor lookup logic but integer-based).
- Leverages stable `id_order` and `years` for direct indexing rather than dynamic string maps.