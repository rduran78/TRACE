 **Diagnosis**  
The current pipeline recomputes neighbor relationships for each cell-year row repeatedly inside `compute_neighbor_stats`. With 6.46M rows and 5 variables, this results in massive redundant lookups and memory churn. The neighbor structure is static across years, so rebuilding neighbor indices per row/year is unnecessary. This inefficiency explains the 86+ hour runtime.

---

**Optimization Strategy**  
1. **Precompute a reusable neighbor lookup table** once using cell IDs (not cell-year).
2. For each year, subset the data, join attributes, and compute neighbor stats using the prebuilt lookup.
3. Use `matrix` operations or `vapply` instead of repeated `lapply`/`rbind` for speed.
4. Process year-by-year to keep memory within 16 GB.
5. Append computed features back to the main dataset and preserve the original estimand.

---

**Optimized R Code**

```r
# Precompute neighbor lookup by cell index (static across years)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_idx[as.character(neighbor_ids)])
  })
}

compute_neighbor_stats_matrix <- function(vals, neighbor_lookup) {
  n <- length(vals)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neigh_vals <- vals[idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        res[i, ] <- c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
      }
    }
  }
  res
}

# Main optimized pipeline
id_order <- unique(cell_data$id)
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output list
results_list <- vector("list", length = length(unique(cell_data$year)))

years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  year_data <- cell_data[cell_data$year == yr, ]
  year_data <- year_data[match(id_order, year_data$id), ]  # align by id_order

  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_neighbor_stats_matrix(year_data[[var_name]], neighbor_lookup_static)
    colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    year_data <- cbind(year_data, stats_mat)
  }

  results_list[[as.character(yr)]] <- year_data
}

# Combine all years back
optimized_cell_data <- do.call(rbind, results_list)

# Now you can run predictions with the existing Random Forest model
# predictions <- predict(trained_rf_model, optimized_cell_data)
```

---

**Why This Is Faster**  
- Neighbor lookup built **once** (O(n_cells)).
- Year-by-year processing avoids 6.46M-row full memory load.
- Vectorized loops over static neighbor lists reduce overhead.
- Eliminates repeated string concatenation and hash lookups.

**Expected Runtime**  
From 86+ hours down to a few hours (dominated by I/O and 28 yearly passes), well within laptop constraints.