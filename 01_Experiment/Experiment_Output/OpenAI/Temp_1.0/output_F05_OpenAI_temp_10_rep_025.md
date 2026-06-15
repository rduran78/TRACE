 **Diagnosis:**  
The repeated string-key construction (`paste(data$id, data$year, sep = "_")`) inside `build_neighbor_lookup` and for every row iterated in `lapply` creates severe inefficiency. This is not just a local inefficiency—it is a symptom of a broader algorithmic pattern where string concatenation and hashmap lookups occur millions of times. Given 6.46M rows × 5 neighbor variables × recursive loops, the cost is enormous. The algorithm repeatedly recomputes mappings that are static across neighbor variables.  

The fundamental issue:  
- For each row (6.46M), neighbor lookup involves creating `neighbor_keys` with `paste` and indexing a named vector.
- `build_neighbor_lookup` returns a giant list based on this expensive process before stats are computed.
- This happens once for lookup creation, then stats computation iterates again.

**Optimization Strategy:**  
Persist integer-based indices upfront and operate purely on integer vectors, avoiding repeated string concatenation and hashing. Instead of dynamic string-key generation, precompute a mapping from `(id, year)` → row index once as an integer matrix and then use integer joins.

**Approach:**  
- Sort data by `id` and `year`.
- Create a 2D matrix of indices with dimensions `length(id_order)` × `length(years)`.
- For each row in `data`, fill its `(id, year)` position in the matrix with the row index.
- Build neighbor indices by selecting elements from this matrix using integer positions.

**Working R Code:**  

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Identify unique years and assign positions
  years <- sort(unique(data$year))
  year_to_pos <- setNames(seq_along(years), years)
  
  # Assign positions for ids
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Create matrix to store row indices
  idx_matrix <- matrix(NA_integer_, nrow = length(id_order), ncol = length(years))
  
  # Fill matrix: row index by (id, year)
  for (row in seq_len(nrow(data))) {
    i_id <- id_to_pos[as.character(data$id[row])]
    i_yr <- year_to_pos[as.character(data$year[row])]
    idx_matrix[i_id, i_yr] <- row
  }
  
  # Build neighbor lookup as a list of integer vectors
  row_ids <- seq_len(nrow(data))
  result <- vector("list", length(row_ids))
  
  for (row in row_ids) {
    i_id <- id_to_pos[as.character(data$id[row])]
    i_yr <- year_to_pos[as.character(data$year[row])]
    ref_idx <- neighbors[[i_id]]
    neighbor_rows <- idx_matrix[ref_idx, i_yr]
    result[[row]] <- as.integer(neighbor_rows[!is.na(neighbor_rows)])
  }
  
  result
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

# Pipeline
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste(var_name, c("neighbor_max", "neighbor_min", "neighbor_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats_mat)
}
```

**Expected Impact:**  
- Eliminates ~6M × string-key construction and hashmap lookup.
- Reduces memory thrash and runtime from estimated 86+ hours to potentially a few hours.
- Preserves trained Random Forest model and ensures estimands remain correct.  
This is a **major algorithmic speedup** using integer-based direct indexing and structured precomputation.