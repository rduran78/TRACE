 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:

- For each of ~6.46M rows, the code:
  - Builds `neighbor_keys` by pasting neighbor IDs with the current year.
  - Looks up indices in `idx_lookup` using these keys.
- This is repeated for every row, even though the neighbor structure is static across years.
- Complexity:  
  `O(N * avg_neighbors)` where `N ≈ 6.46M` and `avg_neighbors ≈ 4–8`.  
  The string operations dominate runtime and memory churn.

**Root cause:** The neighbor relationships are recomputed per row-year instead of leveraging the fact that:
- The neighbor graph is fixed across years.
- The panel is a Cartesian product of `id × year`.

---

### **Optimization Strategy**

1. **Precompute a numeric matrix of neighbor indices for all IDs** (not strings).
2. **Exploit panel structure**: For each year, shift the neighbor indices by an offset and reuse.
3. **Avoid string concatenation and hash lookups entirely**.
4. **Vectorize neighbor stats computation** using matrix operations or `vapply`.

This reduces complexity to:
- Precompute: `O(#ids * avg_neighbors)`
- Lookup: `O(N * avg_neighbors)` but with pure integer indexing (fast).

---

### **Working R Code**

```r
# data: data.frame with columns id, year, and variables
# id_order: vector of unique IDs in desired order
# neighbors: spdep::nb object (list of neighbor indices per ID)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  n_rows  <- nrow(data)
  
  # Map id -> position
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute neighbor positions for each ID
  max_deg <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, nrow = n_ids, ncol = max_deg)
  for (i in seq_len(n_ids)) {
    nb <- neighbors[[i]]
    if (length(nb) > 0) {
      neighbor_mat[i, seq_along(nb)] <- nb
    }
  }
  
  # Sort data by id, then year (assumed)
  # Compute offsets for each year block
  years <- sort(unique(data$year))
  rows_per_year <- n_ids
  stopifnot(n_rows == n_ids * n_years)
  
  # Build lookup as integer indices
  neighbor_lookup <- vector("list", n_rows)
  for (y_idx in seq_along(years)) {
    offset <- (y_idx - 1) * n_ids
    row_start <- offset + 1
    row_end   <- offset + n_ids
    for (i in seq_len(n_ids)) {
      nb_ids <- neighbor_mat[i, ]
      nb_ids <- nb_ids[!is.na(nb_ids)]
      if (length(nb_ids) > 0) {
        neighbor_lookup[[offset + i]] <- nb_ids + offset
      } else {
        neighbor_lookup[[offset + i]] <- integer(0)
      }
    }
  }
  
  neighbor_lookup
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
  colnames(result) <- c(paste0(var_name, "_nb_max"),
                        paste0(var_name, "_nb_min"),
                        paste0(var_name, "_nb_mean"))
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**

- **No string concatenation**: All lookups are integer-based.
- **No repeated hash map lookups**: Precomputed neighbor indices.
- **Memory efficient**: Stores only integer lists.
- **Preserves estimand**: Same neighbor relationships, same stats.
- **Compatible with existing model**: Output structure unchanged.

---

**Expected speedup**: From 86+ hours to a few hours (or less) on a 16 GB laptop, because the bottleneck (string operations in 6.46M iterations) is eliminated.