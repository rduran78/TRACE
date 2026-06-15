 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all 6.46M rows for each variable, repeatedly scanning neighbor indices. This results in ~32.3M expensive list operations (6.46M × 5 variables), with redundant work across years. The neighbor structure is static across years, but the code recomputes neighbor relationships for every row-year combination. This leads to massive overhead and memory pressure on a 16 GB laptop.

---

**Optimization Strategy**  
1. **Exploit Static Neighbor Structure**:  
   - Build the neighbor lookup **once per cell**, not per cell-year.
   - Store neighbors as integer vectors referencing cell IDs (not row indices).

2. **Vectorized Yearly Computation**:  
   - For each year, slice the data for that year into a matrix of variables.
   - Compute neighbor max, min, mean using **matrix operations** or `vapply` over the static neighbor list.

3. **Avoid Repeated Joins**:  
   - Preallocate output columns for all neighbor stats.
   - Fill them year by year.

4. **Memory Efficiency**:  
   - Process one year at a time (28 iterations) instead of all 6.46M rows at once.
   - Use numeric vectors and `vapply` for speed.

This reduces complexity from O(N*T*V) with redundant neighbor lookups to O(T*(N+E)*V), where E is edges (~1.37M), which is manageable.

---

**Working R Code**

```r
# Build static neighbor lookup by cell ID (not year)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is spdep::nb object
  lapply(neighbors, function(nb) as.integer(nb))  # list of integer vectors
}

# Compute neighbor stats for one year
compute_year_neighbor_stats <- function(year_data, neighbor_lookup, var_names) {
  n <- nrow(year_data)
  out <- matrix(NA_real_, nrow = n, ncol = length(var_names) * 3)
  colnames(out) <- as.vector(t(outer(var_names, c("max", "min", "mean"), paste, sep = "_")))

  for (vi in seq_along(var_names)) {
    vals <- year_data[[var_names[vi]]]
    # For each cell, compute stats from its neighbors
    stats <- vapply(neighbor_lookup, function(idx) {
      if (length(idx) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }, numeric(3))
    out[, (vi - 1) * 3 + 1:3] <- t(stats)
  }
  out
}

# Main optimized pipeline
neighbor_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate result columns
neighbor_cols <- as.vector(t(outer(neighbor_source_vars, c("max", "min", "mean"), paste, sep = "_")))
cell_data[neighbor_cols] <- NA_real_

# Process year by year
years <- sort(unique(cell_data$year))
for (yr in years) {
  idx <- which(cell_data$year == yr)
  year_data <- cell_data[idx, ]
  stats_mat <- compute_year_neighbor_stats(year_data, neighbor_lookup, neighbor_source_vars)
  cell_data[idx, neighbor_cols] <- stats_mat
}

# Now cell_data has neighbor features ready for Random Forest prediction
```

---

**Why This Works**  
- Neighbor relationships are computed **once** (static).
- Each year is processed in isolation, so memory stays within 16 GB.
- Uses `vapply` for speed and avoids repeated `paste`/lookup overhead.
- Preserves the original numerical estimand and the trained Random Forest model.

**Expected Runtime**:  
From 86+ hours → a few hours (dominated by ~28 passes over 344k rows × 5 variables × ~1.37M edges, but vectorized and memory-efficient).