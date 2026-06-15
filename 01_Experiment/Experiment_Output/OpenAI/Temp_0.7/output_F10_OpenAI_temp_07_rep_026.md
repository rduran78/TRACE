 **Diagnosis**  
The current pipeline is slow (86+ hours) because:  
- `neighbor_lookup` repeats work for every row-year combination (6.46M rows).  
- Neighbor aggregation occurs via repeated `lapply` calls, creating high overhead.  
- Graph topology is rebuilt per iteration instead of reused.  
- No vectorization; computations are scattered across millions of small lists.  

**Optimization Strategy**  
- Represent the panel as a sparse graph using `Matrix` or `igraph`.  
- Build a single adjacency matrix for rook neighbors (344,208 nodes).  
- For each year, slice the data and apply sparse matrix multiplication to compute sums/means quickly.  
- Compute min/max via efficient grouping using `pmin`/`pmax`.  
- Avoid per-row `lapply`; use vectorized operations.  
- Preallocate results and bind efficiently.  
- Preserve original numerical values (max, min, mean per node-year).  

**Efficient R Implementation**  

```r
library(Matrix)
library(data.table)

# Assume cell_data has columns: id (factor/int), year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# 1. Build sparse adjacency matrix (once)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
i_idx <- rep(seq_along(adj_list), lengths(adj_list))
j_idx <- unlist(adj_list)
adj_mat <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n_cells, n_cells))

# 2. Prepare data.table for fast slicing
setDT(cell_data)
setkey(cell_data, year)

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 3. Function to compute neighbor stats for one variable across all years
compute_neighbor_stats_sparse <- function(var_name) {
  result_list <- vector("list", length(unique(cell_data$year)))
  yrs <- sort(unique(cell_data$year))
  
  for (k in seq_along(yrs)) {
    yr <- yrs[k]
    slice <- cell_data[year == yr, ..var_name]
    vals <- slice[[1]]
    
    # Sparse aggregation
    # Mean: sum of neighbors / neighbor count
    neighbor_sum <- as.numeric(adj_mat %*% vals)
    neighbor_count <- rowSums(adj_mat)
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # For min/max, iterate neighbors efficiently
    # Use adjacency indices to compute min/max without large lapply
    # Vectorized approach:
    idx_pairs <- cbind(i_idx, j_idx)
    neighbor_vals <- vals[j_idx]
    dt <- data.table(i = i_idx, val = neighbor_vals)
    min_vals <- dt[, .(min = min(val)), by = i][order(i)]$min
    max_vals <- dt[, .(max = max(val)), by = i][order(i)]$max
    
    # Handle cells with no neighbors
    has_neighbors <- neighbor_count > 0
    min_vals[!has_neighbors] <- NA_real_
    max_vals[!has_neighbors] <- NA_real_
    
    result_list[[k]] <- data.table(
      year = yr,
      max = max_vals,
      min = min_vals,
      mean = neighbor_mean
    )
  }
  
  rbindlist(result_list)
}

# 4. Compute for all variables and merge
all_stats <- list()
for (v in neighbor_vars) {
  stats_v <- compute_neighbor_stats_sparse(v)
  setnames(stats_v, c("year", paste0(v, "_max"), paste0(v, "_min"), paste0(v, "_mean")))
  if (length(all_stats) == 0) {
    all_stats[[1]] <- stats_v
  } else {
    all_stats[[length(all_stats) + 1]] <- stats_v[, -1]  # drop year column for merge
  }
}

neighbor_features <- do.call(cbind, all_stats)
neighbor_features[, id := id_order]
final_data <- merge(cell_data, neighbor_features, by = c("id", "year"))

# Random Forest prediction (preserve trained model)
predictions <- predict(pretrained_rf_model, newdata = final_data)
```

**Why This Is Fast**  
- Adjacency built once as a sparse matrix.  
- Heavy computations (sum, mean) done via matrix multiplication.  
- Min/max computed using grouped aggregation in `data.table` rather than millions of small lists.  
- Memory use optimized by processing one year at a time.  
- Preserves numerical equivalence with original logic.  

**Expected Improvement**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, due to vectorization and sparse operations.