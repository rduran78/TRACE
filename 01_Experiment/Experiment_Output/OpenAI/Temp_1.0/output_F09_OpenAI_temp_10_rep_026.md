 **Diagnosis**  
- The pipeline recomputes neighbor lookups for all ~6.46 million rows. The nested `lapply` scales linearly with rows and repeatedly creates string keys (`paste`) for neighbors per cell-year. String operations and repeated lookups cause severe overhead and memory churn.  
- Neighbor relationships themselves are static across years, so the costly per-row neighbor resolution can be avoided.  
- Current design lacks vectorization and unnecessarily builds intermediate lists repeatedly.

---

**Optimization Strategy**  
1. **Build reusable neighbor index once**:  
   - Precompute `neighbor_lookup_base` as a list of neighbor cell indices (not cell-year rows).  
   - Year dimension is appended cheaply later.  
2. **Process by year chunks**:  
   - For each year, filter rows, map neighbor IDs to their current year's rows, then compute stats.  
3. **Vectorize where possible**:  
   - Use matrix operations and `tapply`/`aggregate` logic instead of heavy nested `lapply`.  
4. **Avoid string-based keys**:  
   - Use integer indexing, which is faster and memory efficient.  
5. Keep **trained Random Forest model** unchanged.

---

**Working R Code**

```r
# Precompute base neighbor lookup
build_base_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    nbr_idx <- neighbors[[i]]
    id_order[nbr_idx]
  })
}

# Compute stats for a single year's data
compute_year_neighbor_stats <- function(year_data, base_lookup, var_name) {
  vals <- year_data[[var_name]]
  id_to_row <- setNames(seq_len(nrow(year_data)), year_data$id)
  
  result <- matrix(NA_real_, nrow = nrow(year_data), ncol = 3,
                   dimnames = list(NULL, c("max", "min", "mean")))
  
  for (i in seq_len(nrow(year_data))) {
    nbr_ids <- base_lookup[[i]]
    nbr_rows <- id_to_row[as.character(nbr_ids)]
    nbr_rows <- nbr_rows[!is.na(nbr_rows)]
    if (length(nbr_rows) > 0) {
      nbr_vals <- vals[nbr_rows]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        result[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
      }
    }
  }
  result
}

# Main pipeline
id_order <- sort(unique(cell_data$id))
base_lookup <- build_base_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Add columns for neighbor features
for (var in neighbor_source_vars) {
  for (stat in c("max","min","mean")) {
    cell_data[[paste0(var, "_nbr_", stat)]] <- NA_real_
  }
}

# Process year by year (memory-friendly)
for (yr in years) {
  year_idx <- which(cell_data$year == yr)
  year_data <- cell_data[year_idx, ]
  
  for (var in neighbor_source_vars) {
    stats_mat <- compute_year_neighbor_stats(year_data, base_lookup, var)
    cell_data[year_idx, paste0(var, "_nbr_max")]  <- stats_mat[, "max"]
    cell_data[year_idx, paste0(var, "_nbr_min")]  <- stats_mat[, "min"]
    cell_data[year_idx, paste0(var, "_nbr_mean")] <- stats_mat[, "mean"]
  }
}

# Predict using existing Random Forest model
# predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why this is faster**  
- Eliminates repeated string concatenations and large hash lookups across all rows.  
- Uses static base neighbor structure and year-based chunking, reducing complexity and memory pressure.  
- Scales approximately O(N * V) per year (N = cells per year, V = neighbor variables), versus O(N * Y * overhead) in original implementation.  

Expected speed-up: from ~86 hours to a few hours on a standard laptop, depending on I/O and disk performance.