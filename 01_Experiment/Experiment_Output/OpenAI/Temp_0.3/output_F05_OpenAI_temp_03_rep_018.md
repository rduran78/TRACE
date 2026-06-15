 **Diagnosis**  
The repeated string-key work (`paste(data$id, data$year, sep = "_")` and repeated lookups in `idx_lookup`) inside `lapply` is only a symptom of a **larger algorithmic inefficiency**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs string-based lookups repeatedly.  
- This is extremely costly because string concatenation and hash lookups dominate runtime.  
- The outer loop repeats this process for each variable, compounding the inefficiency.  
- The neighbor relationships are **static across years**, so recomputing neighbor indices per row is unnecessary.  

**Optimization Strategy**  
- Precompute a full **numeric neighbor index matrix** once, avoiding string operations entirely.  
- Use integer indices for direct lookup instead of string keys.  
- Compute neighbor statistics in a **vectorized or batched way** rather than row-by-row.  
- Avoid recomputing for each variable: reuse the same neighbor index structure.  

**Algorithmic Reformulation**  
1. Sort `data` by `id` and `year`.  
2. Create a matrix `neighbor_idx` where each row corresponds to a cell-year observation, and columns store neighbor row indices (or `NA` if missing).  
3. Compute neighbor stats for all variables using this matrix.  

---

### **Working R Code**

```r
# Assumes: data has columns id, year, and is sorted by id then year
# id_order: vector of unique cell IDs in desired order
# neighbors: list of neighbor indices (as in spdep nb object)

build_neighbor_index_matrix <- function(data, id_order, neighbors) {
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  n_rows  <- nrow(data)
  
  # Map id -> position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Precompute neighbor ids for each id
  max_neighbors <- max(lengths(neighbors))
  neighbor_ids_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = max_neighbors)
  for (i in seq_along(neighbors)) {
    if (length(neighbors[[i]]) > 0) {
      neighbor_ids_matrix[i, seq_along(neighbors[[i]])] <- neighbors[[i]]
    }
  }
  
  # Build neighbor index matrix for all cell-years
  # Each block of n_ids rows corresponds to a year
  neighbor_idx <- matrix(NA_integer_, nrow = n_rows, ncol = max_neighbors)
  for (year_idx in seq_len(n_years)) {
    row_offset <- (year_idx - 1) * n_ids
    for (id_idx in seq_len(n_ids)) {
      row_id <- row_offset + id_idx
      nbrs   <- neighbor_ids_matrix[id_idx, ]
      nbrs   <- nbrs[!is.na(nbrs)]
      if (length(nbrs) > 0) {
        neighbor_idx[row_id, seq_along(nbrs)] <- row_offset + nbrs
      }
    }
  }
  
  neighbor_idx
}

compute_neighbor_stats_matrix <- function(data, neighbor_idx, var_name) {
  vals <- data[[var_name]]
  n    <- nrow(neighbor_idx)
  max_neighbors <- ncol(neighbor_idx)
  
  # Initialize result
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    nbr_idx <- neighbor_idx[i, ]
    nbr_idx <- nbr_idx[!is.na(nbr_idx)]
    if (length(nbr_idx) > 0) {
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        res[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  
  colnames(res) <- c(paste0(var_name, "_nbr_max"),
                     paste0(var_name, "_nbr_min"),
                     paste0(var_name, "_nbr_mean"))
  res
}

# ---- Main pipeline ----
# Sort data by id then year
cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

neighbor_idx <- build_neighbor_index_matrix(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_matrix(cell_data, neighbor_idx, var_name)
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Why This Is Faster**
- Eliminates per-row string concatenation and hash lookups.
- Builds neighbor index matrix once and reuses it.
- Reduces complexity from repeated string operations to pure integer indexing.
- Memory-efficient: `neighbor_idx` is `nrow(data) × max_neighbors` integers (manageable on 16GB RAM).

**Expected Impact:**  
From 86+ hours to a few hours or less, depending on I/O and CPU speed, because the main bottleneck (string operations in tight loops) is removed.  

**Preserves:**  
- Original estimand  
- Trained Random Forest model  
- All neighbor-based features, computed identically.