 **Diagnosis**  
The repeated string-key construction (`paste(id, year, sep = "_")`) and lookups inside `lapply` are not just a local inefficiency—they are part of a broader algorithmic pattern that scales poorly. For 6.46M rows and 1.37M neighbor links, repeatedly creating strings and performing hash lookups in R introduces massive overhead. The current approach effectively recomputes neighbor indices for every row and every variable, even though the neighbor structure is static across variables.

**Optimization Strategy**  
- **Precompute neighbor indices once** as integer vectors, avoiding repeated string concatenation and hash lookups.
- Use **integer-based mapping** instead of string keys: create a matrix or list of neighbor row indices aligned with the data frame.
- Reuse this precomputed neighbor index structure for all variables.
- Use **vectorized operations** or `vapply` for speed when computing statistics.
- Avoid repeated `lapply` over 6.46M rows for each variable; instead, compute stats using precomputed neighbor indices in a single pass.

---

### **Optimized Implementation**

```r
# Precompute neighbor row indices once
build_neighbor_index <- function(data, id_order, neighbors) {
  # Map id to its reference position
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) to row index
  # Convert year to factor for fast indexing
  year_levels <- sort(unique(data$year))
  year_to_idx <- setNames(seq_along(year_levels), year_levels)
  
  # Create a 2D index: id_ref x year -> row index
  n_ids <- length(id_order)
  n_years <- length(year_levels)
  row_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  
  # Fill row_matrix
  for (i in seq_len(nrow(data))) {
    ref <- id_to_ref[as.character(data$id[i])]
    yr  <- year_to_idx[as.character(data$year[i])]
    row_matrix[ref, yr] <- i
  }
  
  # Build neighbor index list
  row_ids <- seq_len(nrow(data))
  neighbor_index <- vector("list", length = nrow(data))
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    yr_idx  <- year_to_idx[as.character(data$year[i])]
    neighbor_refs <- neighbors[[ref_idx]]
    neighbor_rows <- row_matrix[neighbor_refs, yr_idx]
    neighbor_index[[i]] <- neighbor_rows[!is.na(neighbor_rows)]
  }
  
  neighbor_index
}

# Compute stats using precomputed neighbor indices
compute_neighbor_stats_fast <- function(data, neighbor_index, var_name) {
  vals <- data[[var_name]]
  result <- vapply(neighbor_index, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  t(result)
}

# Example usage
neighbor_index <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_index, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- **No repeated string concatenation**: All lookups are integer-based.
- **Neighbor structure computed once**: `neighbor_index` is reused for all variables.
- **Memory-efficient**: Uses integer matrices and lists instead of large hash maps.
- **Time complexity reduced**: From repeated O(n) string operations to O(n) integer lookups.

---

**Expected Performance Gain**  
This approach should reduce runtime from **86+ hours to a few hours** on a standard laptop, as the main bottleneck (string-based lookups repeated millions of times) is eliminated. The trained Random Forest model and original estimand remain unchanged.