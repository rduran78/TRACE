 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over **6.46 million rows** and repeatedly computes neighbor statistics by scanning neighbors for each row. This results in redundant work across years since the neighbor structure is static, but the code recomputes neighbor indices for every row-year combination. Additionally, the `lapply` approach creates large intermediate objects and does not leverage vectorization or grouping by year.

---

**Optimization Strategy**  
1. **Precompute neighbor lookup once per cell (not per row-year)** since neighbors do not change across years.
2. **Process data year by year**: For each year, extract the relevant slice of data and compute neighbor stats using the static neighbor lookup.
3. Use **vectorized operations** (e.g., `vapply`, `matrix` operations) instead of deeply nested `lapply`.
4. Avoid creating huge lists; instead, build a matrix or data frame directly.
5. Keep memory usage low by working in chunks (year-level) rather than the full 6.46M rows at once.

---

**Optimized R Code**

```r
# Precompute neighbor lookup at cell level (not row-year)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # Returns a list where each element is the vector of neighbor indices (cell-level)
  lapply(seq_along(id_order), function(i) {
    id_order[neighbors[[i]]]
  })
}

compute_neighbor_stats_year <- function(year_data, neighbor_lookup_static, var_name) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  
  # Compute stats for each cell
  result <- vapply(neighbor_lookup_static, function(neigh_ids) {
    neighbor_vals <- vals[as.character(neigh_ids)]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }, numeric(3))
  
  t(result)  # transpose to rows = cells
}

# Main optimized pipeline
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output container
neighbor_feature_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))
}))

# Process year by year
years <- sort(unique(cell_data$year))
cell_data_out <- list()

for (yr in years) {
  year_slice <- cell_data[cell_data$year == yr, ]
  year_slice <- year_slice[match(id_order, year_slice$id), ]  # align with id_order
  
  # Compute neighbor stats for each variable
  stats_matrix <- do.call(cbind, lapply(neighbor_source_vars, function(v) {
    compute_neighbor_stats_year(year_slice, neighbor_lookup_static, v)
  }))
  
  colnames(stats_matrix) <- neighbor_feature_names
  year_slice <- cbind(year_slice, stats_matrix)
  cell_data_out[[as.character(yr)]] <- year_slice
}

cell_data <- do.call(rbind, cell_data_out)
```

---

**Why This Is Faster**  
- **Neighbor lookup computed once** for 344,208 cells instead of 6.46M rows.
- **Year-based processing** reduces memory pressure and avoids redundant computations.
- **Vectorized operations** replace millions of small list operations.
- Expected runtime drops from 86+ hours to **a few hours** (depending on disk I/O and CPU).

---

**Preserves**  
- Original Random Forest model (unchanged).
- Original numerical estimand (neighbor max, min, mean per cell-year).
- Data integrity (same rows, same features, just computed efficiently).