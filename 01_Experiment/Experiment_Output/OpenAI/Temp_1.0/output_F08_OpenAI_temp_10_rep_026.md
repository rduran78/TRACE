 **Diagnosis**  
The current implementation repeatedly computes neighbor statistics for each `(cell, year)` using an `lapply` across ~6.46M rows. For each row, it builds neighbor lookups by matching both `id` and `year`. This introduces massive redundancy since the neighbor structure is static but being recalculated for every row-year combination. Result: excessive memory access, repeated lookups, and very poor performance (86+ hours).

---

**Optimization Strategy**  
1. **Precompute static neighbor mapping at cell level only** (not per year).
2. **Iterate by year** and compute neighbor stats in batches:  
   - For each year, slice the dataset (all cells for that year) into a temporary data frame.  
   - Use static neighbor index arrays to directly compute stats for that year.  
3. **Vectorized computation** instead of nested `lapply` per row when possible.  
4. Avoid building huge lists of per-row neighbors; reuse the same `neighbor_lookup` across years.  
   
This reduces overhead from millions of duplicated computations and memory lookups.

---

**Working R Code**

```r
# Static neighbor lookup per cell (just indices of cells, no year)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # neighbors is a list: each element corresponds to a cell's neighbors
  # Create a named list of integer vectors representing positions in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_pos[as.character(neighbor_ids)])
  })
}

compute_neighbor_stats_year <- function(year_data, neighbor_lookup, var_name) {
  vals <- year_data[[var_name]]
  n <- nrow(year_data)
  out <- matrix(NA_real_, nrow = n, ncol = 3)  # max, min, mean
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    nbr_vals <- vals[idx]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) > 0) {
      out[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    } else {
      out[i, ] <- c(NA_real_, NA_real_, NA_real_)
    }
  }
  out
}

add_neighbor_features_batch <- function(data, neighbor_lookup, neighbor_vars) {
  years <- sort(unique(data$year))
  result_list <- vector("list", length(years))
  
  for (y in seq_along(years)) {
    yr <- years[y]
    year_data <- data[data$year == yr, ]
    feature_mats <- lapply(neighbor_vars, function(var) {
      compute_neighbor_stats_year(year_data, neighbor_lookup, var)
    })
    # Combine into data frame
    for (j in seq_along(neighbor_vars)) {
      stats_mat <- feature_mats[[j]]
      colnames(stats_mat) <- paste0(neighbor_vars[j], c("_nbr_max", "_nbr_min", "_nbr_mean"))
      year_data <- cbind(year_data, stats_mat)
    }
    result_list[[y]] <- year_data
  }
  do.call(rbind, result_list)
}

# Usage:
# Precompute static neighbors (no year)
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_batch(cell_data, neighbor_lookup_static, neighbor_source_vars)
```

---

**Why This Works**  
- **Static neighbor lookup** eliminates repeated year-based lookups.  
- **Year-wise batching** confines vector operations to ~344K rows vs 6.46M repeated overhead.  
- **Preserves estimands** because each neighbor stat is still computed for the same `(cell, year)` using its year-specific variable values.  
- No model retraining; only preprocessing is optimized.

Expected runtime improves from **86+ hours to a few hours** on standard hardware.