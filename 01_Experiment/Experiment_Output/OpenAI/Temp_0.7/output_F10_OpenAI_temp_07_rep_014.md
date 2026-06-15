 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) due to:  
- Repeatedly scanning the entire dataset for each row and each variable.  
- Heavy use of `lapply` and list operations rather than vectorized or matrix-based computations.  
- Rebuilding neighbor relationships in a row-wise manner instead of leveraging efficient graph structures.  

The main performance bottleneck: `compute_neighbor_stats` loops through 6.46 million rows × 5 variables, repeatedly extracting subsets from large vectors.  

**Optimization Strategy**  
- Represent the panel as a sparse graph adjacency structure (using `Matrix` or `igraph`).  
- Build a single adjacency matrix for spatial cells (344,208 nodes).  
- For each year, filter rows and compute neighbor stats using **matrix multiplication with sparse matrices**, which is highly optimized in R.  
- Process variable blocks in memory-efficient chunks.  
- Preallocate output matrices and bind results once.  
- Keep the Random Forest model unchanged, preserving numerical equivalence of computed neighbor features.  

**Efficient Implementation in R**  

```r
library(Matrix)
library(data.table)

# Assume: cell_data (id, year, variables), id_order, rook_neighbors_unique (spdep nb object)

# Convert nb object to adjacency matrix (cells only, no time)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_len(n_cells), sapply(adj_list, length))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# Convert data to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Parameters
years <- sort(unique(cell_data$year))
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate result storage
for (var in neighbor_vars) {
  cell_data[, paste0(var, "_nb_max") := NA_real_]
  cell_data[, paste0(var, "_nb_min") := NA_real_]
  cell_data[, paste0(var, "_nb_mean") := NA_real_]
}

# Process year by year
for (yr in years) {
  # Subset current year
  year_data <- cell_data[year == yr]
  
  # Map rows to adjacency matrix order
  vals_by_id <- year_data[match(id_order, id), ..neighbor_vars]
  
  # Convert to dense matrix for fast row ops
  vals_matrix <- as.matrix(vals_by_id)
  
  # For each variable, compute neighbor stats
  for (j in seq_along(neighbor_vars)) {
    vcol <- vals_matrix[, j]
    
    # Multiply adjacency by values to get sum of neighbors (for mean)
    sum_vals <- adj_mat %*% vcol
    count_vals <- adj_mat %*% !is.na(vcol)
    
    # Compute element-wise
    nb_mean <- as.numeric(sum_vals) / as.numeric(count_vals)
    
    # For max and min, use aggregation on adjacency indices
    # Build index-based aggregation
    # Extract adjacency list once to avoid overhead
    nlist <- adj_list
    nb_max <- nb_min <- numeric(n_cells)
    for (i in seq_len(n_cells)) {
      neigh_idx <- nlist[[i]]
      if (length(neigh_idx) == 0) {
        nb_max[i] <- nb_min[i] <- NA
      } else {
        neigh_vals <- vcol[neigh_idx]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) == 0) {
          nb_max[i] <- nb_min[i] <- NA
        } else {
          nb_max[i] <- max(neigh_vals)
          nb_min[i] <- min(neigh_vals)
        }
      }
    }
    
    # Assign back
    cell_data[year == yr, paste0(neighbor_vars[j], "_nb_max") := nb_max[match(id, id_order)]]
    cell_data[year == yr, paste0(neighbor_vars[j], "_nb_min") := nb_min[match(id, id_order)]]
    cell_data[year == yr, paste0(neighbor_vars[j], "_nb_mean") := nb_mean[match(id, id_order)]]
  }
}

# At this point, cell_data has all neighbor features computed.
# Apply pre-trained Random Forest model without retraining:
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

**Key Improvements**  
- Build adjacency once, reuse across years.  
- Use sparse matrix multiplication for neighbor sums (fast for mean).  
- Loop only over 28 years instead of 6.46M rows per variable.  
- Vectorized assignments via `data.table`.  
- Preserves original estimand and Random Forest integrity.  

**Expected Speed Gain**  
From 86+ hours to a few hours or less (mostly limited by max/min loops, which can be further parallelized with `parallel::mclapply`).  

**Further Optimization**  
- Replace max/min loop with `pmax`/`pmin` on chunked adjacency slices or use `Rcpp` for vectorized C++ aggregation.  
- Parallelize per-year or per-variable computations.  
- Memory footprint remains manageable since adjacency is stored once as sparse structure.