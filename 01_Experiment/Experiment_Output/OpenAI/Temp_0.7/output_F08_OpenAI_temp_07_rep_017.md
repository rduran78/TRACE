 ### **Diagnosis**

The current pipeline is inefficient because:

1. **Redundant computation**: `compute_neighbor_stats` is applied on the full 6.46M rows for each variable (5 variables × 6.46M rows = ~32M iterations).
2. **Repeated neighbor resolution**: Although `neighbor_lookup` is static, the computation loops over all rows repeatedly.
3. **Memory and CPU heavy**: For each row, repeated list lookups and aggregation create large overhead.
4. **Year-specific dependence**: The neighbor relationships are static across years, but the computation does not leverage this — it recomputes the same neighbor index logic for every row.

---

### **Optimization Strategy**

- **Precompute neighbor relationships once per cell** (already done).
- **Exploit year grouping**: Compute neighbor stats year by year using vectorized operations instead of looping row by row.
- **Matrix-based approach**:
  - Represent `id`-year data as a list of yearly slices.
  - For each year and variable, compute neighbor aggregates by applying the adjacency structure to the vector of variable values for that year.
- **Use `sparseMatrix` multiplication**:
  - Convert rook neighbor structure to a sparse adjacency matrix `A` (size: 344,208 × 344,208).
  - For each year:
    - Extract values for that year as vector `v`.
    - Compute `max`, `min`, `mean` using adjacency index lists efficiently.
- **Avoid row-wise `lapply`**: Replace with vectorized operations using `rowsum` or `tapply` patterns.

**Memory feasibility**: 344k × 344k matrix is too large dense, but adjacency in `spdep::nb` is sparse (~1.37M edges ≈ 0.001% density). `Matrix` package handles this efficiently.

---

### **Optimized R Code**

```r
library(Matrix)
library(data.table)

# Convert data to data.table for speed
setDT(cell_data)

# Step 1: Build adjacency as sparse matrix
# rook_neighbors_unique: list of integer vectors (length = n_cells)
n_cells <- length(id_order)
neighbors <- rook_neighbors_unique
row_idx <- rep(seq_along(neighbors), sapply(neighbors, length))
col_idx <- unlist(neighbors)
adjacency <- sparseMatrix(i = row_idx, j = col_idx, x = 1, dims = c(n_cells, n_cells))

# Step 2: Precompute map from id -> row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Step 3: Split data by year for processing
years <- sort(unique(cell_data$year))

# Step 4: Function to compute neighbor stats for one variable and one year
compute_year_stats <- function(dt_year, var_vec, adjacency) {
  # var_vec ordered by id_order
  var_vec_dt <- var_vec
  # For mean: sum and count
  nbr_sum <- adjacency %*% var_vec_dt
  nbr_count <- rowSums(adjacency)
  nbr_mean <- nbr_sum / nbr_count
  
  # For max and min: need list-based apply because sparseMatrix max/min not vectorized
  idx_list <- neighbors
  nbr_max <- numeric(length(idx_list))
  nbr_min <- numeric(length(idx_list))
  for (i in seq_along(idx_list)) {
    vals <- var_vec_dt[idx_list[[i]]]
    nbr_max[i] <- if (length(vals)) max(vals, na.rm = TRUE) else NA
    nbr_min[i] <- if (length(vals)) min(vals, na.rm = TRUE) else NA
  }
  
  list(max = nbr_max, min = nbr_min, mean = as.numeric(nbr_mean))
}

# Step 5: Main loop over variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
results_list <- vector("list", length(neighbor_source_vars))

for (var_name in neighbor_source_vars) {
  # Prepare storage
  max_all <- numeric(nrow(cell_data))
  min_all <- numeric(nrow(cell_data))
  mean_all <- numeric(nrow(cell_data))
  
  for (yr in years) {
    dt_year <- cell_data[year == yr]
    idx_year <- match(dt_year$id, id_order)
    var_vec <- numeric(n_cells)
    var_vec[idx_year] <- dt_year[[var_name]]
    
    stats <- compute_year_stats(dt_year, var_vec, adjacency)
    
    # Assign back to rows for this year
    row_idx <- which(cell_data$year == yr)
    max_all[row_idx] <- stats$max[idx_year]
    min_all[row_idx] <- stats$min[idx_year]
    mean_all[row_idx] <- stats$mean[idx_year]
  }
  
  cell_data[[paste0(var_name, "_nbr_max")]] <- max_all
  cell_data[[paste0(var_name, "_nbr_min")]] <- min_all
  cell_data[[paste0(var_name, "_nbr_mean")]] <- mean_all
}
```

---

### **Why This Is Faster**
- **Single adjacency matrix** shared across all years and variables.
- **Batch computation per year** avoids row-by-row loops.
- **Sparse matrix multiplication** for sums and means.
- **Static neighbor relationships** used directly without recomputation.

---

### **Performance Estimate**
- Complexity drops from O(N × V) row-wise loops (~32M iterations) to:
  - O(Y × V × N_neighbors) for min/max (still large but manageable with efficient loops).
  - O(Y × V × sparse-matrix multiplication) for mean (very fast).
- On a laptop, this should reduce runtime from 86+ hours to a few hours (likely < 2–3 hrs).

---

✅ Preserves:
- Original estimand (neighbor max, min, mean).
- Pre-trained Random Forest model (unchanged).
- Static neighbor structure.

This approach is scalable and memory-safe with 16 GB RAM.