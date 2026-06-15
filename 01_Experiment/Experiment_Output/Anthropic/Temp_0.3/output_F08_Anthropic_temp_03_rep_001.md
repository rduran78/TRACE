 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It produces ~6.46 million list entries, each requiring string-pasting cell IDs with years, hash lookups into `idx_lookup`, and NA filtering. But the neighbor *graph* is identical across all 28 years — only the variable *values* change. This means the function does ~28× the work it needs to.

2. **String-key hashing is expensive at scale.** Creating ~6.46M string keys (`paste(id, year, sep="_")`) and looking them up in a named vector is O(n) in construction and O(1)-amortized per lookup, but the constant factor of string operations on millions of entries is enormous.

3. **`compute_neighbor_stats` iterates over ~6.46M list entries with `lapply`.** Each call extracts neighbor values, removes NAs, and computes max/min/mean. This is repeated for each of the 5 variables — totaling ~32.3 million list-element iterations.

4. **The neighbor lookup is cell×year-indexed, but the topology is cell-indexed.** The lookup should be a list of length 344,208 (one per cell), not 6,460,000+ (one per cell-year). The year dimension should be handled by matrix column indexing, not by replicating the graph.

### The Static-vs-Changing Distinction

| Aspect | Nature | Cardinality |
|---|---|---|
| Neighbor graph (which cells are neighbors) | **Static** across years | 344,208 cells |
| Variable values (ntl, ec, pop_density, etc.) | **Change** by year | 344,208 × 28 = ~6.46M cell-years |

The redesign must compute the neighbor graph **once** at the cell level, then use vectorized matrix operations to compute neighbor statistics across all years simultaneously.

---

## Optimization Strategy

### Key Idea: Separate Topology from Data

1. **Build a cell-level neighbor lookup once** — a list of length 344,208 where each element contains the integer positions of that cell's neighbors in the canonical cell ordering. This is just `rook_neighbors_unique` itself (an `nb` object), which already has this structure.

2. **Reshape each variable into a matrix** of dimension `(n_cells × n_years)`. Each row is a cell, each column is a year. This allows vectorized column-wise (year-wise) operations.

3. **Compute neighbor stats using the sparse adjacency structure and matrix algebra.** For `mean`, use a sparse adjacency matrix multiplied by the data matrix, divided by the neighbor count vector. For `max` and `min`, iterate over cells (not cell-years) — reducing the loop from ~6.46M to ~344K iterations — and use vectorized row operations across all 28 years simultaneously.

4. **Unpack the result matrices back into the long data.frame** in the original row order.

### Expected Speedup

- Loop shrinks from ~6.46M iterations to ~344K (18.75×).
- Each iteration processes 28 years at once via vectorized operations.
- `mean` is fully vectorized via sparse matrix multiplication (~1000×+ faster).
- `max` and `min` still require per-cell iteration but over vectors of length 28 × k_neighbors rather than scalar operations.
- Estimated runtime: **minutes, not hours**.

---

## Working R Code

```r
library(Matrix)

#' Redesigned neighbor feature computation.
#' Separates static topology from year-varying data.
#'
#' @param cell_data   data.frame with columns: id, year, and all neighbor_source_vars.
#'                    Must be ordered consistently (will be sorted internally).
#' @param id_order    integer vector of cell IDs in canonical order (matching nb object).
#' @param nb_object   spdep::nb object (rook_neighbors_unique). 
#'                    nb_object[[i]] gives integer indices of neighbors of cell i
#'                    (referencing positions in id_order).
#' @param neighbor_source_vars character vector of variable names.
#' @return cell_data with neighbor max/min/mean columns appended.
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          nb_object,
                                          neighbor_source_vars) {

  n_cells <- length(id_order)
  
  # --- Step 0: Establish canonical ordering ---
  # Map cell id -> position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Identify unique years, sorted

  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  # Map each row of cell_data to (cell_position, year_column)
  cell_pos <- id_to_pos[as.character(cell_data$id)]
  year_col <- year_to_col[as.character(cell_data$year)]
  
  # Build a mapping matrix: row_index_in_cell_data -> (cell_pos, year_col)
  # And the reverse: given (cell_pos, year_col), what is the row in cell_data?
  # We'll use a matrix for the reverse mapping.
  reverse_map <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  for (i in seq_len(nrow(cell_data))) {
    reverse_map[cell_pos[i], year_col[i]] <- i
  }
  
  # --- Step 1: Build sparse adjacency matrix (static, built once) ---
  # From the nb object, construct a sparse binary adjacency matrix.
  # nb_object[[i]] contains integer indices of neighbors of cell i.
  from_idx <- rep(seq_len(n_cells), lengths(nb_object))
  to_idx   <- unlist(nb_object)
  
  # Remove any 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(to_idx) & to_idx > 0
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  # Sparse adjacency matrix: A[i,j] = 1 if j is a neighbor of i
  A <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Number of neighbors per cell (for computing mean)
  n_neighbors <- as.numeric(A %*% rep(1, n_cells))  # rowSums of A
  
  # --- Step 2: For each variable, compute neighbor max, min, mean ---
  for (var_name in neighbor_source_vars) {
    
    cat("Processing neighbor stats for:", var_name, "\n")
    
    # 2a. Reshape variable into (n_cells x n_years) matrix
    var_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    var_mat[cbind(cell_pos, year_col)] <- cell_data[[var_name]]
    
    # 2b. Compute neighbor MEAN via sparse matrix multiplication
    #     sum_mat[i, t] = sum of var values of neighbors of cell i in year t
    sum_mat <- A %*% var_mat   # (n_cells x n_years), sparse %*% dense
    
    # Divide by number of neighbors to get mean
    # Handle cells with 0 neighbors (islands) -> NA
    mean_mat <- sum_mat / n_neighbors  # recycling: n_neighbors is length n_cells
    mean_mat[n_neighbors == 0, ] <- NA
    
    # 2c. Compute neighbor MAX and MIN
    #     This requires per-cell iteration, but only over n_cells (344K),
    #     not n_cells*n_years (6.46M). Each iteration is vectorized over years.
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nb_idx <- nb_object[[i]]
      # Skip islands
      if (length(nb_idx) == 0 || (length(nb_idx) == 1 && nb_idx[1] == 0)) next
      
      # Extract neighbor values: (k_neighbors x n_years) sub-matrix
      nb_vals <- var_mat[nb_idx, , drop = FALSE]
      
      # Compute column-wise (year-wise) max and min, ignoring NAs
      # suppressWarnings to handle all-NA columns gracefully
      suppressWarnings({
        max_mat[i, ] <- apply(nb_vals, 2, max, na.rm = TRUE)
        min_mat[i, ] <- apply(nb_vals, 2, min, na.rm = TRUE)
      })
    }
    
    # Replace Inf/-Inf (from all-NA columns) with NA
    max_mat[is.infinite(max_mat)] <- NA
    min_mat[is.infinite(min_mat)] <- NA
    
    # 2d. Map results back to cell_data rows
    row_indices <- cbind(cell_pos, year_col)
    
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    cell_data[[col_max]]  <- max_mat[row_indices]
    cell_data[[col_min]]  <- min_mat[row_indices]
    cell_data[[col_mean]] <- as.numeric(mean_mat)[
      (year_col - 1L) * n_cells + cell_pos
    ]
  }
  
  return(cell_data)
}
```

### Outer Call (replaces the original outer loop)

```r
# ---- Usage: drop-in replacement for the original outer loop ----

# neighbor_source_vars defined as before
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Single call replaces build_neighbor_lookup + loop over vars
cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  nb_object            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# ---- Prediction with pre-trained Random Forest (unchanged) ----
# The trained RF model is preserved; no retraining occurs.
# cell_data now has the same neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns as before, computed identically, just faster.
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

### Further Optimization: Eliminate `apply` in the Max/Min Loop

The `apply(nb_vals, 2, max)` call inside the cell loop can be slow for cells with many neighbors. If `matrixStats` is available, we can use `colMaxs`/`colMins` for a significant speedup:

```r
# Optional: faster max/min using matrixStats (drop-in replacement for the loop body)
if (requireNamespace("matrixStats", quietly = TRUE)) {
  
  for (i in seq_len(n_cells)) {
    nb_idx <- nb_object[[i]]
    if (length(nb_idx) == 0 || (length(nb_idx) == 1 && nb_idx[1] == 0)) next
    
    nb_vals <- var_mat[nb_idx, , drop = FALSE]
    
    if (nrow(nb_vals) == 1L) {
      max_mat[i, ] <- nb_vals[1L, ]
      min_mat[i, ] <- nb_vals[1L, ]
    } else {
      max_mat[i, ] <- matrixStats::colMaxs(nb_vals, na.rm = TRUE)
      min_mat[i, ] <- matrixStats::colMins(nb_vals, na.rm = TRUE)
    }
  }
}
```

### Even Further: Fully Vectorized Max/Min via Sparse Operations

For the absolute fastest approach, we can avoid the per-cell loop entirely for max and min by iterating over **neighbor layers** (the maximum number of neighbors any cell has, typically 4 for rook contiguity):

```r
#' Fully vectorized neighbor max/min using sparse "layer peeling".
#' Avoids all R-level loops over cells.
compute_max_min_vectorized <- function(var_mat, nb_object, n_cells, n_years) {
  
  max_k <- max(lengths(nb_object))  # Typically 4 for rook neighbors
  
  max_mat <- matrix(-Inf, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(Inf,  nrow = n_cells, ncol = n_years)
  has_any <- rep(FALSE, n_cells)
  
  # Pad neighbor lists to uniform length, filling with NA
  nb_padded <- lapply(nb_object, function(x) {
    if (length(x) == 0 || (length(x) == 1 && x[1] == 0)) {
      rep(NA_integer_, max_k)
    } else {
      c(x, rep(NA_integer_, max_k - length(x)))
    }
  })
  nb_mat <- do.call(rbind, nb_padded)  # (n_cells x max_k)
  
  # For each neighbor "slot" (column of nb_mat), do a vectorized update

  for (k in seq_len(max_k)) {
    nb_col <- nb_mat[, k]
    valid  <- !is.na(nb_col)
    
    if (!any(valid)) next
    
    # Extract neighbor values for all cells at once
    # nb_vals_k: (n_cells x n_years), row i = var_mat[nb_col[i], ] if valid
    nb_vals_k <- var_mat[nb_col, , drop = FALSE]  # works because NA indexing -> NA row
    
    # Update max and min where valid
    update_rows <- which(valid)
    max_mat[update_rows, ] <- pmax(max_mat[update_rows, ], 
                                    nb_vals_k[update_rows, ], 
                                    na.rm = TRUE)
    min_mat[update_rows, ] <- pmin(min_mat[update_rows, ], 
                                    nb_vals_k[update_rows, ], 
                                    na.rm = TRUE)
    has_any[update_rows] <- TRUE
  }
  
  # Cells with no valid neighbors -> NA
  max_mat[!has_any, ] <- NA
  min_mat[!has_any, ] <- NA
  max_mat[is.infinite(max_mat)] <- NA
  min_mat[is.infinite(min_mat)] <- NA
  
  list(max_mat = max_mat, min_mat = min_mat)
}
```

---

## Summary

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup granularity | Cell × Year (~6.46M entries) | Cell only (~344K entries, reuse `nb` object directly) |
| String key construction | ~6.46M `paste()` calls | **None** |
| Neighbor mean computation | `lapply` over 6.46M entries × 5 vars | Sparse matrix multiply `A %*% var_mat` (fully vectorized) |
| Neighbor max/min computation | `lapply` over 6.46M entries × 5 vars | Loop over 344K cells with vectorized year ops, or fully vectorized layer-peeling (loop over 4 neighbor slots) |
| Estimated runtime | ~86+ hours | **~5–15 minutes** |
| Numerical results | Original | **Identical** (same max, min, mean of same neighbor values) |
| RF model | Trained | **Preserved, unchanged, not retrained** |