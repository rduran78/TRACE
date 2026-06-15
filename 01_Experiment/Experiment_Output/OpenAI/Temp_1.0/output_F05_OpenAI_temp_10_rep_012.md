 **Diagnosis**  
The repeated `paste()` + string-based lookup in the inner loop is a major inefficiency and a symptom of a broader pattern: for every cell-year row (≈ 6.46M), neighbor indices are derived via string keys. These heavy operations (`paste()`, `setNames`, repeated dictionary access) drive time complexity up drastically because they re-build composite keys instead of leveraging numeric indexing. The algorithm repeatedly maps (id, year) to row indices, which can be precomputed once using numeric references rather than repeated string-matching.

**Optimization Strategy**  
Reformulate the lookup layer so:
- Precompute a fast integer mapping from `id` and `year` to row index using a matrix or a hash-based list indexed by numeric id and year.
- Avoid repeated `paste()` and string manipulation entirely.
- Replace `lapply(row_ids, ...)` with a vectorized approach or at least avoid per-row key construction.
- Use integer-based neighbor references in flat arrays or matrices for quick direct indexing.
  
**Working R Code**

```r
# Precomputation: integer-based lookup table
build_neighbor_lookup_fast <- function(data, id_order, neighbors, years) {
  # Map cell_id to 1:length(id_order)
  id_to_ref <- match(data$id, id_order)
  
  # Create a matrix for year offset
  year_to_offset <- match(data$year, years)
  
  n_ids <- length(id_order)
  n_years <- length(years)
  
  # Compute global index: row position = id_index + (year_index-1)*n_ids
  global_index <- (year_to_offset - 1L) * n_ids + id_to_ref
  
  # Prepare neighbors as integer list indexed by row position
  row_count <- nrow(data)
  
  neighbor_lookup <- vector("list", row_count)
  
  for (i in seq_len(row_count)) {
    ref_idx <- id_to_ref[i]
    neighbor_ref <- neighbors[[ref_idx]]  # numeric positions of neighbors in id_order
    if (length(neighbor_ref) == 0L) {
      neighbor_lookup[[i]] <- integer(0)
    } else {
      # Compute neighbor global indices for same year
      neighbor_global <- (year_to_offset[i] - 1L) * n_ids + neighbor_ref
      neighbor_lookup[[i]] <- neighbor_global
    }
  }
  
  neighbor_lookup
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0L) next
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  
  result
}

# Outer loop
years <- sort(unique(cell_data$year))
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique, years)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste0(var_name, c("_max_nbr", "_min_nbr", "_mean_nbr"))
  cell_data <- cbind(cell_data, stats)
}
```

**Key Improvements**
- Eliminated string-based composite keys (`paste()` and named vectors).
- Leveraged numeric offsets for (id, year).
- Maintained row order integrity and estimands.
- Compatible with existing Random Forest model since feature output is unchanged.

Expected runtime drops from multi-hour to manageable since integer operations and precomputed offsets are far cheaper than repeated string manipulation over millions of rows.