 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It iterates over all ~6.46 million rows, performing string-pasting and hash lookups for every row. But the neighbor *relationships* among cells are **time-invariant** — cell A is a rook-neighbor of cell B regardless of year. The lookup should be built once over 344,208 cells, not over 6.46 million cell-year rows.

2. **`compute_neighbor_stats` operates on the full stacked data frame row-by-row.** It calls an `lapply` over 6.46M elements, each of which subsets a vector by index. While each individual operation is small, the sheer volume of R-level loop iterations is crushing.

3. **String key construction (`paste(id, year, sep="_")`) and named-vector lookup (`idx_lookup[neighbor_keys]`)** are O(n) in memory and slow in R for millions of entries. This is entirely unnecessary since the neighbor topology doesn't change.

4. **The data layout is row-stacked (long panel).** To exploit the static topology, we should reshape to a **cell × year matrix** for each variable, apply neighbor aggregation as matrix operations over the cell dimension (using the static neighbor list), and then slot results back.

### Why it matters

| Aspect | Current | Proposed |
|---|---|---|
| Neighbor lookup size | 6.46M entries | 344,208 entries (one per cell) |
| Stats computation loops | 6.46M × 5 vars = 32.3M iterations | 344,208 × 5 vars = 1.72M iterations (vectorized across years) |
| String hashing | ~19M `paste` + named lookups | Zero |
| Memory pattern | Random access into 6.46M-row frame | Column-wise matrix ops (cache-friendly) |

---

## Optimization Strategy

**Core insight:** Separate the *static graph structure* (which cells are neighbors) from the *dynamic variable values* (which change by year). Then vectorize the year dimension.

### Steps

1. **Build a cell-level neighbor lookup once** — a simple list of length 344,208, where element `i` contains the integer indices of cell `i`'s neighbors. This is just `rook_neighbors_unique` itself (an `nb` object), possibly re-indexed.

2. **For each variable, reshape to a matrix of dimension `n_cells × n_years`.** Each column is one year's values for all cells.

3. **Compute neighbor max/min/mean as matrix operations.** For each cell `i`, gather its neighbors' rows from the matrix, and compute columnwise (i.e., year-wise) max, min, mean. This turns 6.46M R-level iterations into 344,208 iterations, each of which is a small vectorized matrix operation across 28 years simultaneously.

4. **Melt the result matrices back to long format** and join onto the original data frame.

5. **Pass to the pre-trained Random Forest unchanged** — the numerical results are identical (same max, min, mean of the same neighbor values), preserving the original estimand.

### Expected speedup

- Loop iterations reduced by factor of ~18.8× (28 years collapsed).
- Each iteration is vectorized (matrix column ops) instead of scalar.
- No string operations. No hash lookups.
- Estimated runtime: **~2–10 minutes** instead of 86+ hours.

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the static-topology / dynamic-variable distinction
# ==============================================================================

compute_all_neighbor_features <- function(cell_data, 
                                           id_order, 
                                           rook_neighbors_unique, 
                                           neighbor_source_vars) {
  # ------------------------------------------------------------------
  # STEP 1: Establish stable cell ordering and year ordering

  # ------------------------------------------------------------------
  
  # id_order is the vector of cell IDs in the same order as rook_neighbors_unique.
  # We create a mapping from cell ID -> position index in id_order.
  n_cells <- length(id_order)
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Identify unique years and sort them
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))
  
  # ------------------------------------------------------------------
  # STEP 2: Compute row position of each cell_data row in the 
  #         (cell_pos, year_col) matrix layout
  #
  #   We need this to (a) fill matrices from long data, and 
  #   (b) write results back to the correct rows.
  # ------------------------------------------------------------------
  
  cell_data_pos <- id_to_pos[as.character(cell_data$id)]   # cell position for each row
  cell_data_col <- year_to_col[as.character(cell_data$year)] # year column for each row
  
  # Linear index into an n_cells x n_years matrix (column-major)
  linear_idx <- cell_data_pos + (cell_data_col - 1L) * n_cells
  
  # ------------------------------------------------------------------
  # STEP 3: The static neighbor list is rook_neighbors_unique itself.
  #         It is already a list of length n_cells where element [[i]]

  #         gives the indices (into id_order) of cell i's neighbors.
  #         spdep::nb objects use integer index vectors. We just use it
  #         directly, filtering out the 0-neighbor sentinel if present.
  # ------------------------------------------------------------------
  
  # spdep nb objects encode "no neighbors" as a single integer 0.
  # We convert to a clean list of integer vectors.
  nb_list <- lapply(rook_neighbors_unique, function(x) {
    x <- as.integer(x)
    x[x > 0L]
  })
  
  # ------------------------------------------------------------------
  # STEP 4: For each variable, build matrix, compute neighbor stats,
  #         and write results back to cell_data
  # ------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing neighbor stats for: %s\n", var_name))
    
    # 4a. Build n_cells x n_years matrix from long data
    vals_vec <- cell_data[[var_name]]
    mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat[linear_idx] <- vals_vec
    
    # 4b. Compute neighbor max, min, mean — loop over cells only (344K),
    #     vectorized across years (28 columns at once)
    
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nbrs <- nb_list[[i]]
      if (length(nbrs) == 0L) next
      
      # Submatrix: rows = neighbors, cols = years
      # For a typical rook neighborhood, this is 2-4 rows × 28 cols — tiny.
      nb_mat <- mat[nbrs, , drop = FALSE]
      
      if (nrow(nb_mat) == 1L) {
        # Single neighbor: stats are trivial
        nb_max[i, ]  <- nb_mat[1L, ]
        nb_min[i, ]  <- nb_mat[1L, ]
        nb_mean[i, ] <- nb_mat[1L, ]
      } else {
        # Multiple neighbors: column-wise operations
        # Handle NAs the same way as the original code (remove NAs per group)
        # colMeans/colMaxs etc. — we use matrixStats if available, else base
        
        # Using base R for portability:
        nb_max[i, ]  <- apply(nb_mat, 2L, function(v) {
          v <- v[!is.na(v)]
          if (length(v) == 0L) NA_real_ else max(v)
        })
        nb_min[i, ]  <- apply(nb_mat, 2L, function(v) {
          v <- v[!is.na(v)]
          if (length(v) == 0L) NA_real_ else min(v)
        })
        nb_mean[i, ] <- apply(nb_mat, 2L, function(v) {
          v <- v[!is.na(v)]
          if (length(v) == 0L) NA_real_ else mean(v)
        })
      }
    }
    
    # 4c. Write results back to cell_data in the correct row positions
    max_col_name  <- paste0(var_name, "_neighbor_max")
    min_col_name  <- paste0(var_name, "_neighbor_min")
    mean_col_name <- paste0(var_name, "_neighbor_mean")
    
    cell_data[[max_col_name]]  <- nb_max[linear_idx]
    cell_data[[min_col_name]]  <- nb_min[linear_idx]
    cell_data[[mean_col_name]] <- nb_mean[linear_idx]
    
    cat(sprintf("    -> Added %s, %s, %s\n", max_col_name, min_col_name, mean_col_name))
  }
  
  return(cell_data)
}

# ==============================================================================
# USAGE — drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The pre-trained Random Forest model is used unchanged downstream:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Acceleration (Optional): Use `matrixStats` to Eliminate Inner `apply`

The `apply(..., 2, ...)` inside the cell loop processes only 2–4 rows × 28 columns, so it's already fast. But if you want to squeeze out more performance, replace the inner apply calls with `matrixStats::colMaxs`, `colMins`, `colMeans2`:

```r
# Install once: install.packages("matrixStats")
library(matrixStats)

# Replace the multi-neighbor branch with:
if (nrow(nb_mat) == 1L) {
  nb_max[i, ]  <- nb_mat[1L, ]
  nb_min[i, ]  <- nb_mat[1L, ]
  nb_mean[i, ] <- nb_mat[1L, ]
} else {
  nb_max[i, ]  <- colMaxs(nb_mat,  na.rm = TRUE)
  nb_min[i, ]  <- colMins(nb_mat,  na.rm = TRUE)
  nb_mean[i, ] <- colMeans2(nb_mat, na.rm = TRUE)
}
# Note: colMaxs/colMins return -Inf/Inf for all-NA columns.
# Post-process if needed:
# nb_max[i, ][is.infinite(nb_max[i, ])] <- NA_real_
# nb_min[i, ][is.infinite(nb_min[i, ])] <- NA_real_
```

This version with `matrixStats` is the fastest and should bring total runtime to **under 5 minutes** on a 16 GB laptop.

---

## Summary

| Dimension | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key entries | 344K integer-index entries (reuse `nb` directly) |
| **Main loop iterations** | 6.46M × 5 = 32.3M | 344K × 5 = 1.72M |
| **Per-iteration work** | Scalar (1 year) | Vectorized (28 years simultaneously) |
| **String operations** | ~19M `paste` calls | Zero |
| **Hash table lookups** | ~19M named-vector lookups | Zero |
| **Estimated runtime** | 86+ hours | ~2–10 minutes |
| **Numerical output** | Original estimand | **Preserved identically** |
| **RF model** | Pre-trained | **Unchanged, not retrained** |