 **Diagnosis**  
The repeated construction of string keys (`paste(id, year, sep = "_")`) in `build_neighbor_lookup` is not just a local inefficiency; it reflects a broader algorithmic pattern where the code performs repeated string concatenation and hash lookups for each cell-year record. This is costly because:

- There are ~6.46 million rows × multiple neighbors → tens of millions of string operations.
- The computation is repeated for each variable in `neighbor_source_vars`.
- The outer loop and `lapply` structure make the complexity roughly O(N × K), where N is rows and K is neighbors, with a heavy constant due to string handling.

**Optimization Strategy**  
The inefficiency stems from treating the panel as a flat key-value map instead of leveraging the inherent structure:  
- IDs repeat across years.  
- Neighbors are static across years.  

We can precompute neighbor indices once for all rows without string keys by using integer-based indexing. Specifically:  
1. Sort `data` by `(id, year)`.  
2. Create an integer matrix mapping each `(id, year)` to row positions.  
3. For each row, look up neighbors by direct integer mapping instead of string concatenation.

Then reuse the same `neighbor_lookup` for all variables.

**Optimized R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure data is sorted by id, year
  data <- data[order(data$id, data$year), ]
  row_ids <- seq_len(nrow(data))
  
  # Map id -> position in id_order
  id_to_ref <- match(data$id, id_order)
  
  # Number of years and unique ids
  ids <- unique(data$id)
  years <- unique(data$year)
  n_ids <- length(ids)
  n_years <- length(years)
  
  # Create matrix: row index by [id_idx, year_idx]
  idx_matrix <- matrix(NA_integer_, n_ids, n_years)
  # Map data$id and data$year to positions
  id_idx <- match(data$id, ids)
  year_idx <- match(data$year, years)
  idx_matrix[cbind(id_idx, year_idx)] <- row_ids
  
  # Precompute neighbor indices for each id
  neighbor_lookup <- vector("list", nrow(data))
  
  for (i in seq_len(nrow(data))) {
    this_id_idx <- id_idx[i]
    this_year_idx <- year_idx[i]
    ref_idx <- id_to_ref[i]
    neighbor_ids <- neighbors[[ref_idx]]
    if (length(neighbor_ids) == 0) {
      neighbor_lookup[[i]] <- integer(0)
    } else {
      # Map neighbor_ids to our ids vector
      neighbor_id_idx <- match(id_order[neighbor_ids], ids)
      # Look up row indices for same year
      nbr_rows <- idx_matrix[cbind(neighbor_id_idx, rep(this_year_idx, length(neighbor_id_idx)))]
      neighbor_lookup[[i]] <- nbr_rows[!is.na(nbr_rows)]
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
  
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats)
}
```

**Why This Works**  
- Completely eliminates repeated string concatenation and hashing.
- Single pass to build a numeric lookup structure.
- Same neighbor lookup reused across all variables.
- Complexity reduced to O(N × K) with minimal overhead.

**Expected Impact**  
This approach should cut runtime from 86+ hours to a few hours or less on a 16 GB laptop, as the bottleneck was string overhead and repeated recomputation.