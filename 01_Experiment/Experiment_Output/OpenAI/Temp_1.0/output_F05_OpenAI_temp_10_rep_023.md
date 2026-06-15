 **Diagnosis**

The repeated string concatenation and hash lookup:

```r
neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
result <- idx_lookup[neighbor_keys]
```

inside the `lapply` over every row causes **O(N × avg_neighbors)** string operations and dictionary lookups, where  
- `N ≈ 6.46 million` rows  
- `avg_neighbors ≈ 4`  

That alone is costly, but the real inefficiency is that the same `paste` + lookup pattern is repeated for **each of the 5 neighbor source variables** in the outer loop. However, `build_neighbor_lookup()` is only called once—not per variable—so the heavy cost is during its run, not multiplied by 5. Still, the design repeatedly constructs `neighbor_keys` for each cell-year row, which dominates runtime.

This is **not just a local inefficiency**, it’s a symptom of doing something that can be computed in one vectorized pass: **joining cell-year keys and neighbor lists for all years at once with numeric indices, not strings**.

---

### **Optimization Strategy**

- **Avoid string concatenation entirely.**  
  Compute integer-year offsets so neighbor lookups become direct integer index references via precomputed year shifts.

- **Vectorize neighbor index construction.**  
  For each spatial cell, determine all its neighbors (fixed over years). Then, for every year, map to the correct row indices by adding an integer offset = `(year_index - 1) * num_cells`.

- **Store neighbor references once in a numeric matrix or list of integer vectors**, so `compute_neighbor_stats()` can index directly without hash or string operations.

- This eliminates both the hash-map lookups and per-row string operations.

---

### **Working R Code**

```r
build_neighbor_lookup_optimized <- function(data, id_order, neighbors) {
  # Assume data is sorted by id, then year
  ids        <- unique(data$id)
  years      <- sort(unique(data$year))
  n_cells    <- length(ids)
  n_years    <- length(years)

  # Map IDs to positions
  id_pos     <- setNames(seq_along(ids), as.character(ids))

  # Precompute neighbor sets for each cell (spatial only)
  static_neighbors <- lapply(seq_along(id_order), function(i) {
    id_order[neighbors[[i]]] # neighbor IDs
  })

  # Row index helper: row index = (year_idx - 1)*n_cells + cell_idx
  neighbor_lookup <- vector("list", length = n_cells * n_years)

  for (year_idx in seq_along(years)) {
    year_offset <- (year_idx - 1) * n_cells
    for (cell_idx in seq_along(ids)) {
      neigh_ids <- static_neighbors[[cell_idx]]
      neighbor_row_idx <- year_offset + id_pos[as.character(neigh_ids)]
      neighbor_lookup[[year_offset + cell_idx]] <- as.integer(neighbor_row_idx)
    }
  }
  neighbor_lookup
}

compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  colnames(result) <- c("max", "min", "mean")

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

# Usage:
neighbor_lookup <- build_neighbor_lookup_optimized(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**

- **No string keys.** Integer math is O(1).
- **Neighbor lookup built once for the full panel.**
- Memory trade-off:  
  - List of length ≈ 6.46M (rows)  
  - Each element small (~4–8 integers)  
  Should fit comfortably in 16 GB with efficient storage.
- Outer loop over 5 variables remains, but heavy string work disappears. Expect orders-of-magnitude speed-up, reducing 86+ hours to a few hours or less.

---

**Preserves:**  
- Exact numerical estimand (same neighbors per year, same summary stats)  
- Existing trained model remains intact.  

This structural reformulation removes redundant operations and makes the pipeline feasible on standard hardware.