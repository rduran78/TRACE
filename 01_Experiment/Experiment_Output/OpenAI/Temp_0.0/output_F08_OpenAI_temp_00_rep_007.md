 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every cell-year row (≈6.46M rows) by iterating through a large list of neighbor indices. This results in repeated lookups and redundant computations across years, even though the neighbor structure is static. The complexity is roughly `O(N * k)` per variable, where `N` is 6.46M and `k` is average neighbor count, multiplied by 5 variables. This is memory- and CPU-intensive for a 16 GB laptop.

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute neighbor relationships once at the cell level (344,208 cells), not per cell-year.
- **Vectorize across years**: For each variable, reshape data into a matrix of size `cells × years` and compute neighbor stats using matrix operations.
- **Avoid repeated list traversals**: Use `rowsum` or `apply` on matrices instead of looping through 6.46M rows.
- **Memory efficiency**: Work in chunks if needed, but 344k × 28 matrix (~9.6M entries) is manageable in memory.

**Working R Code**

```r
# Precompute neighbor lookup at cell level (static)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_ref[as.character(neighbor_ids)])
  })
}

# Compute neighbor stats using matrix operations
compute_neighbor_stats_matrix <- function(var_matrix, neighbor_lookup) {
  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)
  
  max_mat <- matrix(NA_real_, n_cells, n_years)
  min_mat <- matrix(NA_real_, n_cells, n_years)
  mean_mat <- matrix(NA_real_, n_cells, n_years)
  
  for (i in seq_len(n_cells)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) next
    neighbor_vals <- var_matrix[idx, , drop = FALSE]
    max_mat[i, ]  <- apply(neighbor_vals, 2, max, na.rm = TRUE)
    min_mat[i, ]  <- apply(neighbor_vals, 2, min, na.rm = TRUE)
    mean_mat[i, ] <- apply(neighbor_vals, 2, mean, na.rm = TRUE)
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# Main pipeline
# Assume cell_data has columns: id, year, and variables
id_order <- sort(unique(cell_data$id))
year_order <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(year_order)

# Reshape data into wide matrix per variable
reshape_to_matrix <- function(data, var_name, id_order, year_order) {
  mat <- matrix(NA_real_, nrow = length(id_order), ncol = length(year_order),
                dimnames = list(id_order, year_order))
  idx <- match(paste(data$id, data$year), 
               paste(rep(id_order, each = n_years), rep(year_order, times = n_cells)))
  mat[idx] <- data[[var_name]]
  mat
}

neighbor_lookup <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge back
for (var_name in neighbor_source_vars) {
  var_matrix <- reshape_to_matrix(cell_data, var_name, id_order, year_order)
  stats <- compute_neighbor_stats_matrix(var_matrix, neighbor_lookup)
  
  # Melt back to long format
  long_idx <- match(paste(cell_data$id, cell_data$year),
                    paste(rep(id_order, each = n_years), rep(year_order, times = n_cells)))
  
  cell_data[[paste0(var_name, "_nbr_max")]]  <- stats$max[long_idx]
  cell_data[[paste0(var_name, "_nbr_min")]]  <- stats$min[long_idx]
  cell_data[[paste0(var_name, "_nbr_mean")]] <- stats$mean[long_idx]
}

# Now cell_data has neighbor features ready for Random Forest prediction
```

**Why this works**  
- Reduces complexity from 6.46M row-wise loops to 344k cell-wise loops with vectorized year operations.
- Handles all years in one pass per variable.
- Preserves original estimand and Random Forest model.
- Fits in memory: 344k × 28 ≈ 9.6M doubles (~77 MB per matrix).

**Expected runtime**: From 86+ hours to a few minutes on a standard laptop.