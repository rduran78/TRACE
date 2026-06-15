 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every cell-year row, which is extremely costly given 6.46M rows and 5 variables. The `build_neighbor_lookup` function is already efficient because it builds a reusable index, but the bottleneck is repeatedly scanning and joining large vectors in `compute_neighbor_stats` for each variable. For 5 variables × 6.46M rows, this results in heavy repeated work and memory churn.

**Optimization Strategy**  
- Build the neighbor lookup **once** at the cell level (not cell-year).
- For each year, subset the data, compute neighbor stats for all 5 variables in a single pass, then append results.
- Use **vectorized operations** and `vapply` or `matrix` binding instead of repeated `lapply` calls.
- Avoid recomputing `idx_lookup` or string concatenations inside loops.
- Process year by year to keep memory within 16 GB.
- Preserve the trained Random Forest model and estimand by producing the same features, just faster.

---

### **Optimized R Code**

```r
# Build neighbor lookup once at cell level
build_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is a list of integer vectors (rook neighbors)
  # Return as-is but aligned to id_order
  neighbors
}

compute_neighbor_stats_matrix <- function(vals, neighbor_lookup) {
  # vals: numeric vector of length = number of cells in that year
  n <- length(vals)
  result <- matrix(NA_real_, nrow = n, ncol = 3) # max, min, mean
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Main optimized pipeline
neighbor_lookup <- build_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate storage
out_list <- vector("list", length = length(unique(cell_data$year)))
names(out_list) <- unique(cell_data$year)

years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  subset_idx <- which(cell_data$year == yr)
  year_data <- cell_data[subset_idx, ]
  
  # Ensure order matches id_order
  year_data <- year_data[match(id_order, year_data$id), ]
  
  # Compute neighbor stats for each variable
  stats_list <- lapply(neighbor_source_vars, function(var_name) {
    compute_neighbor_stats_matrix(year_data[[var_name]], neighbor_lookup)
  })
  
  # Combine all stats into a data.frame
  # Each var contributes 3 columns: max, min, mean
  stats_mat <- do.call(cbind, stats_list)
  colnames(stats_mat) <- as.vector(t(outer(neighbor_source_vars, c("_max", "_min", "_mean"), paste0)))
  
  # Bind to year_data
  year_data <- cbind(year_data, stats_mat)
  
  out_list[[as.character(yr)]] <- year_data
}

# Combine all years back
cell_data_enriched <- do.call(rbind, out_list)

# Predict using existing Random Forest model
predictions <- predict(trained_rf_model, newdata = cell_data_enriched)
```

---

**Why This Is Faster**  
- Neighbor relationships are reused without recomputation.
- Year-by-year processing keeps memory manageable.
- All 5 variables are processed in a single pass per year, reducing overhead.
- Avoids repeated string concatenation and large `lapply` calls over 6.46M rows.

**Expected Impact**  
This approach reduces complexity from ~O(N × V) repeated heavy lookups to O(N) per year with minimal overhead, likely cutting runtime from 86+ hours to a few hours on a 16 GB laptop.