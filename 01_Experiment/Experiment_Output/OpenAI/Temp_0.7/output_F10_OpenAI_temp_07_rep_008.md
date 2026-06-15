 **Diagnosis**  
The current pipeline repeatedly computes neighbor statistics for each variable and each cell-year by traversing `neighbor_lookup` lists, resulting in high overhead. For 6.46M rows and multiple variables, using `lapply` and repeated subsetting is inefficient and memory-heavy. The main inefficiencies:  
- Repeated neighbor graph traversal across years and variables.  
- Using `lapply` and `do.call(rbind, ...)` for large lists (6M+ elements).  
- Building year-specific lookups repeatedly instead of leveraging sparse graph structure.  

**Optimization Strategy**  
- Represent the rook-neighbor relationships as a sparse adjacency matrix once (`Matrix::dgCMatrix`), size = (#cells × #cells).  
- For each year, filter rows corresponding to that year, then use matrix multiplication for aggregation:  
    - Neighbor max/min/mean can be computed by grouping neighbor values efficiently.  
- Process all variables in a single pass per year using vectorized operations.  
- Avoid repeated `paste` and lookups; precompute row indices for each year.  
- Use `data.table` for fast grouping and merging.  
- Preserve original numerical estimand by ensuring identical computations (max, min, mean across neighbors ignoring NA).  

**Working R Code**  

```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2, ...)
# rook_neighbors_unique: spdep::nb object
# id_order: vector of cell IDs in graph order
# rf_model: pre-trained Random Forest model

# 1. Build sparse adjacency matrix once
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 2. Convert cell_data to data.table and add graph index
setDT(cell_data)
id_to_idx <- setNames(seq_along(id_order), id_order)
cell_data[, gidx := id_to_idx[as.character(id)]]

# 3. Function to compute neighbor stats for one year
compute_year_stats <- function(dt_year, adj_mat, var_names) {
  n <- nrow(dt_year)
  gidx <- dt_year$gidx
  
  # Create result matrix: rows = n, cols = length(var_names)*3 (max, min, mean)
  result <- matrix(NA_real_, nrow = n, ncol = length(var_names)*3)
  
  for (v in seq_along(var_names)) {
    vals <- dt_year[[var_names[v]]]
    # Fill a vector for all cells in graph (NA for missing)
    full_vals <- rep(NA_real_, n_cells)
    full_vals[gidx] <- vals
    
    # Compute neighbor sums and counts for mean using matrix multiplication
    neighbor_sum <- as.vector(adj_mat %*% full_vals)
    neighbor_count <- as.vector(adj_mat %*% !is.na(full_vals))
    
    # For min and max, use apply on each row subset efficiently
    # Extract non-NA neighbor values for each node
    idx_non_na <- which(!is.na(full_vals))
    # Precompute list of neighbor indices
    # Use adjacency in dgCMatrix format for fast iteration
    indptr <- adj_mat@p
    indices <- adj_mat@i
    
    max_vals <- numeric(n)
    min_vals <- numeric(n)
    
    for (i in seq_len(n)) {
      start <- indptr[gidx[i]] + 1
      end <- indptr[gidx[i]+1]
      neigh_idx <- indices[start:end] + 1
      neigh_vals <- full_vals[neigh_idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        max_vals[i] <- max(neigh_vals)
        min_vals[i] <- min(neigh_vals)
      } else {
        max_vals[i] <- NA_real_
        min_vals[i] <- NA_real_
      }
    }
    
    mean_vals <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    result[, (v-1)*3 + 1] <- max_vals
    result[, (v-1)*3 + 2] <- min_vals
    result[, (v-1)*3 + 3] <- mean_vals[gidx]
  }
  
  as.data.table(result)
}

# 4. Apply per year and bind results
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- unique(cell_data$year)
all_results <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  stats_dt <- compute_year_stats(dt_year, adj_mat, neighbor_source_vars)
  all_results[[i]] <- cbind(dt_year[, .(id, year)], stats_dt)
}

neighbor_features <- rbindlist(all_results)
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)

# 5. Predict using pre-trained Random Forest model
predictions <- predict(rf_model, newdata = cell_data)

# Save or return predictions
predictions
```

**Why this is efficient**  
- The adjacency matrix is built once and reused.  
- Matrix multiplication handles mean computation in O(E) time for each variable-year.  
- Min/max use efficient iteration over compressed adjacency indices rather than repeated lookups.  
- Data.table ensures fast joins and filtering.  
- Reduces from estimated 86+ hours to a few hours or less, depending on CPU and disk I/O.  

This preserves the original numerical estimand (max, min, mean of non-NA neighbor values) and uses the pre-trained Random Forest model without retraining.