 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeated lookups in R lists.  
- Neighbor lookups are recomputed for every row and every variable, causing redundant work.  
- Pure R loops and list operations are inefficient for this scale.  
- Memory overhead is high due to repeated object creation.  

**Optimization Strategy**  
- Precompute a **flat integer index matrix** for neighbors (fixed width, padded with `NA`) to avoid repeated list traversals.  
- Use **vectorized operations** or **matrix-based aggregation** instead of per-row `lapply`.  
- Leverage **data.table** for fast joins and grouping.  
- Compute all neighbor stats in a single pass per variable using efficient C-level operations (e.g., `matrixStats`).  
- Avoid recomputing neighbor relationships for each year by building a global lookup keyed by `(id, year)` → row index.  

**Working R Code**  

```r
library(data.table)
library(matrixStats)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object

# 1. Precompute neighbor lookup as a flat matrix
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  max_nbrs <- max(lengths(neighbors))
  n <- nrow(data)
  neighbor_mat <- matrix(NA_integer_, nrow = n, ncol = max_nbrs)
  
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    nbr_ids <- id_order[neighbors[[ref_idx]]]
    nbr_keys <- paste(nbr_ids, data$year[i], sep = "_")
    nbr_idx <- idx_lookup[nbr_keys]
    nbr_idx <- as.integer(nbr_idx[!is.na(nbr_idx)])
    if (length(nbr_idx) > 0) {
      neighbor_mat[i, seq_along(nbr_idx)] <- nbr_idx
    }
  }
  neighbor_mat
}

neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# 2. Compute neighbor stats efficiently
compute_neighbor_stats_matrix <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  # Extract neighbor values as matrix
  nbr_vals <- matrix(vals[neighbor_mat], nrow = nrow(neighbor_mat))
  # Compute row-wise stats ignoring NAs
  max_vals <- rowMaxs(nbr_vals, na.rm = TRUE)
  min_vals <- rowMins(nbr_vals, na.rm = TRUE)
  mean_vals <- rowMeans(nbr_vals, na.rm = TRUE)
  # Replace Inf/-Inf with NA for rows with all NAs
  max_vals[is.infinite(max_vals)] <- NA
  min_vals[is.infinite(min_vals)] <- NA
  mean_vals[is.nan(mean_vals)] <- NA
  cbind(max_vals, min_vals, mean_vals)
}

# 3. Apply for all variables and bind to cell_data
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_matrix(cell_data, neighbor_mat, var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}

# cell_data now has neighbor features added
```

**Why this is faster**  
- Single neighbor matrix built once (O(N * avg_neighbors)).  
- Vectorized row-wise operations via `matrixStats` (C-level speed).  
- No repeated list traversals or millions of small object allocations.  

**Expected Performance**  
- Precomputation: a few minutes (depends on neighbor density).  
- Stats computation: seconds per variable (instead of hours).  
- Memory: manageable (~6.46M × max_neighbors integers + working matrices).  

This preserves the original rook-neighbor relationships, computes exact max/min/mean, and avoids retraining the Random Forest model.