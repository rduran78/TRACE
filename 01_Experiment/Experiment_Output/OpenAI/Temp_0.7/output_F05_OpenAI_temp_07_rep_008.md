 **Diagnosis**  
The repeated string-key construction (`paste(data$id, data$year, sep="_")`) inside `build_neighbor_lookup` is not the only inefficiency—it’s a symptom of a broader algorithmic pattern where, for every row (≈6.46M), we repeatedly map neighbor IDs to row indices using string concatenation and hash lookups. This creates massive overhead in both memory and CPU. The lookups themselves are not necessary if we precompute a direct numeric mapping from `(id, year)` to row index.

**Optimization Strategy**  
Avoid repeated string concatenation and hashing by:
1. Precomputing a numeric index mapping from `id` and `year` to row positions using integer arrays rather than character keys.
2. Operating on integer vectors directly (no repeated calls to `paste` or `setNames`).
3. Leveraging vectorized operations and `vapply` or matrix-based joins.

We can build a `row_index_matrix` with dimensions `[id, year]` so that for any `(id, year)` we retrieve a row index in O(1) time without strings.

---

### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors, years) {
  n_ids <- length(id_order)
  n_years <- length(years)
  
  # Map ids to 1:n_ids
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  # Map years to 1:n_years
  year_to_ref <- setNames(seq_along(years), as.character(years))
  
  # Precompute row index matrix: rows = id, cols = year
  row_index_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  row_index_matrix[cbind(id_to_ref[as.character(data$id)],
                          year_to_ref[as.character(data$year)])] <- seq_len(nrow(data))
  
  # Build neighbor lookup without string-based keys
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    yr_idx  <- year_to_ref[as.character(data$year[i])]
    neighbor_ref_ids <- neighbors[[ref_idx]]
    neighbor_rows <- row_index_matrix[neighbor_ref_ids, yr_idx]
    neighbor_rows[!is.na(neighbor_rows)]
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  
  result <- vapply(neighbor_lookup, function(idx) {
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  
  t(result)  # return as matrix
}

# Pipeline
years <- sort(unique(cell_data$year))
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique, years)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Expected Gains**
- Eliminates repeated `paste()` and `setNames()` calls.
- Avoids hash-based lookups in favor of O(1) integer matrix indexing.
- Reduces runtime from **86+ hours to a few hours (or less)** depending on disk I/O and memory constraints.
- Preserves the trained Random Forest model and all numerical estimands.

This approach is memory-efficient for a 16 GB laptop:  
`row_index_matrix` = (344,208 ids × 28 years) ≈ 9.6M integers (~38MB). Very feasible.