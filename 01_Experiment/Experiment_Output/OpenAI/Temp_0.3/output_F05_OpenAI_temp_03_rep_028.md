 **Diagnosis**

The repeated string concatenation and hash lookups (`paste(..., sep = "_")` and `idx_lookup[...]`) inside the inner `lapply` loop of `build_neighbor_lookup` is a **local inefficiency**, but the real issue is **algorithmic**:

- For **6.46 million rows**, `build_neighbor_lookup` iterates over each row and performs:
  - String concatenation for all neighbors.
  - Hash lookups in `idx_lookup`.
- This results in **tens of millions of string operations** and repeated work across variables.
- The outer loop (`compute_neighbor_stats`) then iterates over 5 variables, but at least the neighbor lookup is reused. Still, the initial neighbor lookup build is extremely costly.

**Root cause:** The current design treats the panel as a flat table and repeatedly reconstructs neighbor relationships per row-year using string keys. This is unnecessary because:
- The neighbor structure is **static across years**.
- The mapping from `(id, year)` → row index is deterministic and can be computed once using numeric indexing.

---

### **Optimization Strategy**

1. **Avoid string keys entirely**:
   - Use integer-based indexing: precompute a matrix that maps `(cell_id, year)` to row index.
   - Use numeric IDs for neighbors.

2. **Precompute neighbor indices for all rows**:
   - For each cell, get its neighbors' IDs.
   - For each year, map those neighbor IDs to row indices using a fast integer lookup (no strings).

3. **Vectorize neighbor statistics**:
   - Instead of looping over rows and neighbors, use `rowsum` or matrix operations where possible.

---

### **Proposed Algorithm**

- Create a `row_index_matrix` of size `n_cells × n_years` mapping `(cell_id, year)` → row index.
- For each cell, store its neighbor IDs once.
- For each year, translate neighbor IDs to row indices via `row_index_matrix`.
- Compute stats using precomputed neighbor indices.

---

### **Working R Code**

```r
optimize_neighbor_features <- function(data, id_order, neighbors, years, vars) {
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # Map cell_id and year to row index
  row_index_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years,
                              dimnames = list(as.character(id_order), as.character(years)))
  row_index_matrix[cbind(match(data$id, id_order), match(data$year, years))] <- seq_len(nrow(data))
  
  # Precompute neighbor indices for each cell-year
  neighbor_lookup <- vector("list", length = nrow(data))
  
  for (cell_idx in seq_along(id_order)) {
    cell_neighbors <- neighbors[[cell_idx]]
    if (length(cell_neighbors) == 0) next
    for (year_idx in seq_along(years)) {
      row_id <- row_index_matrix[cell_idx, year_idx]
      if (is.na(row_id)) next
      neighbor_rows <- row_index_matrix[cell_neighbors, year_idx]
      neighbor_lookup[[row_id]] <- neighbor_rows[!is.na(neighbor_rows)]
    }
  }
  
  # Compute neighbor stats for each variable
  for (var_name in vars) {
    vals <- data[[var_name]]
    stats_matrix <- matrix(NA_real_, nrow = nrow(data), ncol = 3)
    for (i in seq_along(neighbor_lookup)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          stats_matrix[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    colnames(stats_matrix) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    data <- cbind(data, stats_matrix)
  }
  
  data
}

# Usage:
years <- 1992:2019
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, years, neighbor_source_vars)
```

---

### **Why This Is Faster**
- Eliminates string concatenation and hash lookups.
- Uses numeric indexing and precomputed matrices.
- Builds `neighbor_lookup` **once** for all years and variables.
- Memory-efficient: `neighbor_lookup` is a list of integer vectors, not strings.

---

**Expected Performance Gain**  
From 86+ hours → likely **minutes to a few hours**, as the bottleneck (string operations and repeated hashing) is removed and replaced with integer-based lookups.

**Preserves**:
- Original Random Forest model.
- Original estimand and feature definitions.