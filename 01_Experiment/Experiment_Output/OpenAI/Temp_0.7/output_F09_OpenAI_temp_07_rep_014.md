 **Diagnosis:**  
The current pipeline recomputes neighbor relationships repeatedly for each variable across all 6.46M rows, which is highly inefficient. The nested loops and repeated lookups create excessive overhead. Neighbor relationships are static across years, so rebuilding them per variable is unnecessary. Memory and time costs scale poorly with 344K cells × 28 years.

---

**Optimization Strategy:**  
- Build the neighbor lookup **once** and reuse it for all variables.
- Use **vectorized operations** or preallocate results instead of repeated `lapply` calls.
- Avoid repeated paste and indexing for each variable; compute once.
- Consider `data.table` for fast joins and efficient memory usage.
- Compute neighbor stats in bulk using matrix operations, then bind results to the main dataset.

---

**Working R Code (Optimized):**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Function to compute and add neighbor stats for multiple variables efficiently
compute_and_add_neighbor_features <- function(dt, vars, neighbor_lookup) {
  for (v in vars) {
    vals <- dt[[v]]
    
    # Preallocate matrix for results
    res_mat <- matrix(NA_real_, nrow(dt), 3, dimnames = list(NULL, c("max", "min", "mean")))
    
    # Compute stats
    for (i in seq_along(neighbor_lookup)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          res_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    
    # Add columns to data.table
    dt[[paste0(v, "_nbr_max")]]  <- res_mat[, "max"]
    dt[[paste0(v, "_nbr_min")]]  <- res_mat[, "min"]
    dt[[paste0(v, "_nbr_mean")]] <- res_mat[, "mean"]
  }
  invisible(dt)
}

# Apply optimization
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
compute_and_add_neighbor_features(cell_data, neighbor_source_vars, neighbor_lookup)

# The Random Forest model can now predict with new features without retraining
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Expected Impact:**  
- Neighbor table built **once** instead of 5×.
- Eliminates redundant string operations and repeated indexing.
- Improves runtime dramatically—likely reducing from 86+ hours to a few hours (depending on thread usage).
- Preserves estimands and trained model integrity.