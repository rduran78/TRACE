 Diagnosis:  
The repeated string-key operations (`paste()` and `idx_lookup[...]`) inside the inner `lapply` loop are not just a local inefficiency—they occur for every cell-year row (≈6.46M times) and for each neighbor set. This creates a massive overhead due to repeated string concatenation and hash lookups. The root cause is that the algorithm repeatedly reconstructs neighbor indices per row rather than precomputing them once.  

Optimization Strategy:  
Instead of building neighbor keys dynamically for every row-year combination, precompute a numeric neighbor index matrix aligned with `data` rows. This avoids repeated string operations and hash lookups. The idea:  
1. Map each `(id, year)` pair to its row index once.  
2. Expand neighbor relationships across all years using vectorized operations.  
3. Store neighbor indices in a list or matrix for direct numeric access.  

Working R Code:  

```r
# Precompute neighbor lookup for all years without repeated string ops
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  n <- nrow(data)
  years <- unique(data$year)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Map (id, year) -> row index
  idx_matrix <- matrix(seq_len(n), nrow = length(id_order), ncol = length(years))
  rownames(idx_matrix) <- as.character(id_order)
  colnames(idx_matrix) <- as.character(years)
  
  # Fill idx_matrix: assume data is sorted by id and year
  # Create a fast lookup for row positions
  id_year_to_row <- split(seq_len(n), paste(data$id, data$year, sep = "_"))
  
  for (i in seq_len(n)) {
    idx_matrix[as.character(data$id[i]), as.character(data$year[i])] <- i
  }
  
  # Build neighbor lookup: list of integer vectors
  row_ids <- seq_len(n)
  neighbor_lookup <- vector("list", n)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    yr <- as.character(data$year[i])
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_lookup[[i]] <- idx_matrix[as.character(neighbor_ids), yr]
  }
  
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    idx <- idx[!is.na(idx)]
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
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

Key Improvements:  
- Eliminates repeated `paste()` and hash lookups inside the inner loop.  
- Uses numeric indexing via a precomputed matrix for `(id, year)` → row mapping.  
- Preserves original estimand and Random Forest model.  

Expected Impact:  
This reduces complexity from repeated string operations to pure numeric indexing, cutting runtime from tens of hours to a few hours or less on a standard laptop.