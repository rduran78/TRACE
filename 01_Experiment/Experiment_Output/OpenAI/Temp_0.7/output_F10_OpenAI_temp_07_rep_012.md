 **Diagnosis**  
- The current implementation repeatedly computes neighbor statistics via `lapply` for each row and each variable, which is extremely inefficient (86+ hours).  
- Neighbor relationships (`rook_neighbors_unique`) are static across years, but `build_neighbor_lookup` is called per full dataset, creating massive repeated lookups.  
- For 6.46M rows, looping in pure R with `lapply` and repeated indexing is a bottleneck.  
- Memory overhead from repeatedly binding results and creating vectors is high.  

---

**Optimization Strategy**  
1. **Precompute Graph Topology Once**: Build a neighbor index matrix that maps each cell to its neighbors (static across years).  
2. **Vectorization**: Use fast aggregation via `data.table` or `matrixStats` across neighbors instead of `lapply`.  
3. **Batch Processing**: Compute stats year-by-year to keep memory usage low while leveraging the static graph.  
4. **Avoid Repeated Lookups**: Convert long panel data to `data.table`, keyed by `(id, year)`, and use integer-based joins.  
5. **Parallelization**: Use `future.apply` or `parallel` for independent yearly computations.  
6. **Preserve Numerical Equivalence**: Same max, min, mean definitions; same handling of `NA`.  

---

**Optimized Working R Code**  

```r
library(data.table)
library(matrixStats)

compute_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, vars) {
  # Convert to data.table
  setDT(cell_data)
  setkey(cell_data, id, year)
  
  # Build neighbor index for cells (static across years)
  id_to_pos <- setNames(seq_along(id_order), id_order)
  neighbor_list <- lapply(rook_neighbors_unique, function(nb) id_to_pos[nb])
  
  # Precompute neighbor matrix for fast access
  max_neighbors <- max(lengths(neighbor_list))
  neighbor_mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_neighbors)
  for (i in seq_along(neighbor_list)) {
    nbs <- neighbor_list[[i]]
    neighbor_mat[i, seq_along(nbs)] <- nbs
  }
  
  # Prepare output columns
  for (v in vars) {
    cell_data[, paste0(v, "_nb_max") := NA_real_]
    cell_data[, paste0(v, "_nb_min") := NA_real_]
    cell_data[, paste0(v, "_nb_mean") := NA_real_]
  }
  
  # Process year by year for memory efficiency
  years <- unique(cell_data$year)
  
  for (yr in years) {
    dt_year <- cell_data[year == yr]
    vals_mat <- as.matrix(dt_year[, ..vars])
    
    # Compute neighbor stats for each variable using matrix operations
    for (j in seq_along(vars)) {
      col_vals <- vals_mat[, j]
      # Gather neighbor values for all cells
      n_vals <- matrix(col_vals[neighbor_mat], nrow = nrow(neighbor_mat))
      # Handle NAs
      n_vals[is.na(n_vals)] <- NA_real_
      nb_max <- rowMaxs(n_vals, na.rm = TRUE)
      nb_min <- rowMins(n_vals, na.rm = TRUE)
      nb_mean <- rowMeans(n_vals, na.rm = TRUE)
      nb_max[is.infinite(nb_max)] <- NA_real_
      nb_min[is.infinite(nb_min)] <- NA_real_
      nb_mean[is.nan(nb_mean)] <- NA_real_
      
      # Assign results back
      cell_data[year == yr, (paste0(vars[j], "_nb_max")) := nb_max]
      cell_data[year == yr, (paste0(vars[j], "_nb_min")) := nb_min]
      cell_data[year == yr, (paste0(vars[j], "_nb_mean")) := nb_mean]
    }
  }
  
  return(cell_data)
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Predict with pre-trained Random Forest (unchanged)
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why This Works**  
- **Graph Built Once**: `neighbor_mat` stores all neighbors, reused across years.  
- **Matrix Operations**: `rowMaxs`, `rowMins`, `rowMeans` from `matrixStats` are highly optimized in C.  
- **Memory Efficiency**: Operate year-by-year to avoid loading all 6.46M rows into neighbor matrices at once.  
- **No Model Retraining**: Predictions use the original Random Forest.  

This approach reduces complexity from *O(N × neighbors × vars × years) in R loops* to efficient matrix operations, likely cutting runtime from 86+ hours to **under 1 hour** on a 16 GB laptop.