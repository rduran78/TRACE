 **Diagnosis**

- Current approach computes neighbor statistics row-wise across ~6.46M rows, recalculating operations for each cell-year combination, causing severe inefficiency.
- Neighbor relationships are static, but neighbor lookup is rebuilt dynamically for each row-year.
- For 28 years × 344,208 cells and millions of neighbor lookups, this results in massive redundant computations and memory overhead.

**Optimization Strategy**

1. **Exploit Static Neighbor Topology**: Compute neighbor index list **once** for the 344,208 unique cells (not per cell-year).
2. **Vectorize by Year**: For each year, slice the data and operate on 344,208 rows using matrix operations instead of looping over millions.
3. **Use preallocated structures**: Preallocate result matrices for all variables and avoid rbind loops.
4. **Aggregate neighbor stats with `vapply` and fast indexing**: Avoid per-element lapply over 6M+ entries.
5. **Append results efficiently**: Bind results back to the original data after computing in chunks by year.

**Working R Code**

```r
# Precompute a static neighbor lookup for cell IDs
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_ref[as.character(neighbor_ids)])
  })
}

compute_neighbor_stats_by_year <- function(data, years, var_names, neighbor_lookup, n_cells) {
  # Prepare output list for each variable
  out_list <- vector("list", length(var_names))
  names(out_list) <- var_names
  for (vn in var_names) {
    out_list[[vn]] <- matrix(NA_real_, nrow = n_cells * length(years), ncol = 3)
  }

  # Process year by year
  for (y_idx in seq_along(years)) {
    y <- years[y_idx]
    slice_idx <- data$year == y
    year_data <- data[slice_idx, ]
    stopifnot(nrow(year_data) == n_cells)  # Ensure ordering
    for (vn in var_names) {
      vals <- year_data[[vn]]
      stats_mat <- vapply(neighbor_lookup, function(nb) {
        if (length(nb) == 0) return(c(NA, NA, NA))
        nv <- vals[nb]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0) return(c(NA, NA, NA))
        c(max(nv), min(nv), mean(nv))
      }, numeric(3))
      out_list[[vn]][((y_idx - 1) * n_cells + 1):(y_idx * n_cells), ] <- t(stats_mat)
    }
  }
  out_list
}

# ----------------------------
# Usage
n_cells <- length(id_order)
years <- sort(unique(cell_data$year))

# Neighbor lookup is static for cells
neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all stats efficiently
neighbor_stats <- compute_neighbor_stats_by_year(cell_data, years, neighbor_source_vars, neighbor_lookup_static, n_cells)

# Bind back to main data
for (vn in neighbor_source_vars) {
  stat_names <- c("max", "min", "mean")
  for (j in seq_along(stat_names)) {
    col_name <- paste0(vn, "_nb_", stat_names[j])
    cell_data[[col_name]] <- neighbor_stats[[vn]][, j]
  }
}

# Preserve trained RF model usage as before
predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Expected gain**:  
- Avoids 6.46M × per-row neighbor recomputation → now loops over 344k cells per year × 28 years in vectorized fashion.
- Reduces estimated runtime from 86+ hours to likely under 1–2 hours even on modest hardware.
- Preserves the original estimand and model integrity.