 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics (`max`, `min`, `mean`) for every cell-year row (≈6.46M rows) by iterating through `neighbor_lookup` each time. This is O(N × K) with large N (6.46M) and K (avg. neighbors), causing 86+ hours runtime. The neighbor structure is static across years, but the code does not exploit this: it rebuilds and applies neighbor lookups repeatedly for all rows, even though the neighbor relationships are constant.

---

**Optimization Strategy**  
1. **Precompute static neighbor indices once per cell** (already done by `build_neighbor_lookup`).
2. **Process year by year**: For each year, slice the data for that year, compute neighbor stats using the static neighbor structure (cell-level), then merge back. This avoids iterating over all 6.46M rows at once.
3. **Vectorize computations**: Use matrix operations or `vapply` instead of repeated `lapply` over all rows.
4. **Memory efficiency**: Work in chunks (per year) to keep RAM usage low.
5. **Preserve estimand**: The computed neighbor max/min/mean must remain identical to the original logic.

---

**Optimized R Code**

```r
# Precompute neighbor lookup at cell level (static)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # neighbors: spdep::nb object
  lapply(seq_along(id_order), function(i) neighbors[[i]])
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one year
compute_neighbor_stats_year <- function(year_data, var_name, neighbor_lookup_static, id_order) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  # Preallocate result matrix
  res <- matrix(NA_real_, nrow = nrow(year_data), ncol = 3)
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  
  for (i in seq_along(id_order)) {
    cell_id <- id_order[i]
    nbr_ids <- id_order[neighbor_lookup_static[[i]]]
    nbr_vals <- vals[as.character(nbr_ids)]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) > 0) {
      res[year_data$id == cell_id, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }
  }
  res
}

# Main loop: process by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  idx <- which(cell_data$year == yr)
  year_data <- cell_data[idx, ]
  
  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_neighbor_stats_year(year_data, var_name, neighbor_lookup_static, id_order)
    cell_data[idx, colnames(stats_mat)] <- stats_mat
  }
}

# At this point, cell_data has neighbor features added.
# Proceed with Random Forest prediction as before.
```

---

**Why This is Faster**  
- The neighbor relationships are computed once, not per row-year.
- Computation is done year by year (28 iterations), reducing memory and avoiding 6.46M-row lapply.
- Each year processes only 344,208 rows with a simple loop over cells and vector lookups.
- Expected runtime drops from 86+ hours to a few hours or less on a standard laptop.

**Preserves**  
- Trained Random Forest model.
- Original numerical estimand (same max/min/mean logic).
- Static neighbor structure assumption.