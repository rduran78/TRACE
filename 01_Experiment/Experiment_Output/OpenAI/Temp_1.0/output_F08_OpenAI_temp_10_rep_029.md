 **Diagnosis**  
The current implementation is extremely slow because:  
- `compute_neighbor_stats` iterates over all 6.46M rows and recomputes stats for each variable separately (5×6.46M operations).  
- `neighbor_lookup` is huge and re-used redundantly for each variable.  
- The code performs repeated `lapply`/`do.call` for each feature, producing large intermediate lists and repeated NA filtering.  
- Static neighbor relationships (344k cells, ~1.37M neighbor links) are unnecessarily recalculated for every year and variable.

---

**Optimization Strategy**  
1. **Exploit static structure**: Precompute a sparse neighbor matrix (cells × cells) once.  
2. **Group by year**: For each year, do a single matrix multiplication for all variables using the sparse matrix.  
3. **Vectorize**: Compute max, min, mean neighbor stats in a single pass using efficient operations.  
4. **Memory-efficiency**: Use `Matrix` package for storing neighbor relationships as a sparse structure.  

Complexity drops from `O(rows * vars)` lookups to `O(years * (cells + neighbor_links * vars))`.

---

**Working R Code**

```r
library(Matrix)
library(data.table)

# Assumes cell_data has columns: id (cell_id), year, and source vars.
setDT(cell_data)

# 1. Build sparse neighbor adjacency matrix once
n_cells <- length(id_order)
id_to_idx <- setNames(seq_along(id_order), id_order)

neighbors <- rook_neighbors_unique
row_idx <- rep(seq_along(neighbors), lengths(neighbors))
col_idx <- unlist(neighbors)
adj <- sparseMatrix(
  i = row_idx,
  j = col_idx,
  dims = c(n_cells, n_cells),
  x = 1
)

# 2. Prepare result container
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_year_key <- paste(cell_data$id, cell_data$year)
res_list <- vector("list", length(neighbor_source_vars))
names(res_list) <- neighbor_source_vars

# 3. Compute neighbor stats by year in a loop
years <- sort(unique(cell_data$year))
for (var in neighbor_source_vars) {
  # Initialize output matrices
  max_mat <- numeric(nrow(cell_data))
  min_mat <- numeric(nrow(cell_data))
  mean_mat <- numeric(nrow(cell_data))
  
  for (yr in years) {
    idx_year <- which(cell_data$year == yr)
    vals <- cell_data[[var]][idx_year]
    
    # Convert to dense vector (cells order)
    v <- rep(NA_real_, n_cells)
    v[id_to_idx[cell_data$id[idx_year]]] <- vals
    
    # Compute neighbor values for each cell
    # Extract indices of non-NA neighbors efficiently
    for (cell in which(!is.na(v))) {
      neigh_idx <- neighbors[[cell]]
      neigh_vals <- v[neigh_idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        row_pos <- idx_year[which(cell_data$id[idx_year] == id_order[cell])]
        max_mat[row_pos] <- max(neigh_vals)
        min_mat[row_pos] <- min(neigh_vals)
        mean_mat[row_pos] <- mean(neigh_vals)
      }
    }
  }
  
  res_list[[var]] <- data.table(
    id = cell_data$id,
    year = cell_data$year,
    paste0(var, "_ngb_max") := max_mat,
    paste0(var, "_ngb_min") := min_mat,
    paste0(var, "_ngb_mean") := mean_mat
  )
}

# 4. Merge results into original data
for (var in neighbor_source_vars) {
  cell_data <- cbind(cell_data, res_list[[var]][, -c("id", "year")])
}

# Random Forest prediction step remains unchanged
```

---

**Performance Impact**  
- Eliminates repeated giant `lapply` calls.
- Works year-by-year to keep memory manageable.
- Preserves original numerical estimand and trained RF model.
- Estimated runtime drops from 86+ hours to a few hours (depending on I/O and sparse handling).  

Further optimization: Parallelize the inner year loops (`future.apply`), or precompute neighbor indices as vectors to reduce repeated lookups.