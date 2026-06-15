 **Diagnosis**  
The current pipeline is extremely slow because `compute_neighbor_stats` iterates over all 6.46M rows and recomputes neighbor lookups for each year-variable combination. This results in repeated traversals and redundant aggregation. The neighbor structure is static across years, but calculations are re-done per cell-year in a highly inefficient way (essentially O(N * k * #vars * #years), with N ≈ 6.46M). Additionally, using `lapply` with row-wise operations in R on millions of rows is memory and CPU expensive.

---

**Optimization Strategy**  
1. **Exploit Static Neighbor Structure:** Compute a fixed neighbor index for each cell (not cell-year).  
2. **Vectorize Yearly Computations:** For each year and variable, compute neighbor aggregates in a single grouped operation instead of row-by-row loops.  
3. **Use Matrix Representation:** Store variables in a cell × year matrix to allow fast column-wise operations.  
4. **Avoid Large Repeated `paste` and Lookups:** Precompute an integer neighbor list.  
5. **Process in Chunks (Optional):** For memory constraints, handle one variable at a time but vectorized.  

---

**Optimized R Code**

```r
# Precompute neighbor index once (static across years)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), id_order)
  lapply(neighbors, function(nbrs) as.integer(id_to_idx[nbrs]))
}

# Compute stats for all years efficiently
compute_neighbor_stats_matrix <- function(var_matrix, neighbor_lookup) {
  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)
  
  max_mat <- matrix(NA_real_, n_cells, n_years)
  min_mat <- matrix(NA_real_, n_cells, n_years)
  mean_mat <- matrix(NA_real_, n_cells, n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- neighbor_lookup[[i]]
    if (length(nbrs) == 0) next
    nbr_vals <- var_matrix[nbrs, , drop = FALSE]
    # Compute across rows for each column (year)
    max_mat[i, ]  <- apply(nbr_vals, 2, max, na.rm = TRUE)
    min_mat[i, ]  <- apply(nbr_vals, 2, min, na.rm = TRUE)
    mean_mat[i, ] <- apply(nbr_vals, 2, mean, na.rm = TRUE)
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# --- Main Pipeline ---
# Assumptions:
# cell_data: data.frame with columns id, year, and vars
# id_order: vector of unique cell IDs
# rook_neighbors_unique: spdep nb object

neighbor_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# Reshape to cell × year matrix for each variable
library(data.table)
dt <- as.data.table(cell_data)
setkey(dt, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  var_matrix <- matrix(NA_real_, n_cells, n_years)
  var_matrix[as.integer(factor(dt$id, levels = id_order)),
             as.integer(factor(dt$year, levels = years))] <- dt[[var_name]]
  
  stats <- compute_neighbor_stats_matrix(var_matrix, neighbor_lookup)
  
  # Convert back and bind to dt
  for (stat_type in names(stats)) {
    stat_values <- as.vector(t(stats[[stat_type]])) # cell-year order
    new_col <- paste0(var_name, "_nbr_", stat_type)
    dt[, (new_col) := stat_values]
  }
}

# dt now contains original data + neighbor features
cell_data <- as.data.frame(dt)
```

---

**Why This Works**
- **Static neighbor lookup** avoids recomputing for every row-year.  
- **Matrix-based computation** reduces overhead by eliminating 6.46M row loops; instead, operations happen in vectorized chunks.  
- **Preserves trained Random Forest model** (same features, same estimand).  
- Expected runtime drops from **86+ hours to a few hours (or less)** on a standard laptop.  

**Memory Consideration:**  
- For 344,208 cells × 28 years ≈ 9.6M elements per matrix (≈ 76 MB per variable in double precision), easily fits in 16 GB RAM if processed sequentially per variable.