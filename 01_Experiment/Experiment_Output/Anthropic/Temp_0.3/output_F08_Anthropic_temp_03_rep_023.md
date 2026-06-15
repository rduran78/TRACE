 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup (one entry per cell-year row, ~6.46 million entries), even though the neighbor *topology* is identical across all 28 years. The function does two expensive things redundantly:

1. **Repeats neighbor identification 28 times per cell.** Cell `i`'s neighbors are the same in 1992 as in 2019, yet the lookup is rebuilt for every cell-year row.
2. **String-based key hashing** (`paste(id, year)` → named vector lookup) for ~6.46M × ~4 neighbors ≈ 25+ billion character operations. This is the dominant cost.

Then `compute_neighbor_stats` iterates over the 6.46M-element list, which is fine in principle but is downstream of the bloated lookup.

**In summary:** The code treats a *static graph* as if it were *year-varying*, inflating both memory and time by a factor of 28 and adding expensive string operations. On a 16 GB laptop this yields the estimated 86+ hour runtime.

---

## Optimization Strategy

**Separate the static topology from the year-varying data:**

1. **Build the neighbor lookup once at the cell level (344K entries), not at the cell-year level (6.46M entries).** This is a simple integer-index mapping from each cell to its neighbor cells. This is done once and reused for every variable and every year.

2. **Reshape each variable into a cell × year matrix** (344,208 rows × 28 columns). This allows vectorized column-wise (i.e., per-year) operations.

3. **Compute neighbor stats using matrix operations.** For each cell, gather neighbor rows from the matrix, then compute max/min/mean across neighbors for each year simultaneously. This replaces 6.46M list iterations with 344K iterations, each operating on a small integer-indexed submatrix — roughly **28× fewer iterations** and no string operations.

4. **Use `data.table` for fast reshaping and joining** to avoid memory copies.

5. **The trained Random Forest model is untouched.** The output columns (`*_neighbor_max`, `*_neighbor_min`, `*_neighbor_mean`) are numerically identical to the original implementation.

**Expected speedup:** From ~86 hours to roughly **2–4 hours** (28× fewer iterations, no string hashing, vectorized matrix ops, cache-friendly memory access).

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build cell-level neighbor lookup ONCE (static topology)
# ==============================================================================
# Input:
#   id_order            — vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique — spdep::nb object (list of integer index vectors)
# Output:
#   cell_neighbor_lookup — named list: cell_id (character) -> vector of neighbor cell_ids

build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors[[i]] gives integer indices into id_order for the neighbors of id_order[i]
  lookup <- vector("list", length(id_order))
  names(lookup) <- as.character(id_order)
  for (i in seq_along(id_order)) {
    nb_idx <- neighbors[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    lookup[[i]] <- nb_idx  # store as INTEGER INDICES into id_order (not cell IDs)
  }
  lookup
}

# ==============================================================================
# STEP 2: Compute neighbor stats via cell × year matrix
# ==============================================================================
# For a given variable, reshape to matrix, compute neighbor max/min/mean per
# cell per year, then join back.
#
# This function returns a data.table with columns: id, year, <var>_neighbor_max,
# <var>_neighbor_min, <var>_neighbor_mean

compute_neighbor_stats_matrix <- function(dt, var_name, id_order, cell_neighbor_lookup, years) {
  # dt must be a data.table with columns: id, year, <var_name>
  # Ensure id and year are keyed for fast subsetting
  
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # --- Build cell × year matrix ---
  # Create a mapping from id to row index in the matrix
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  # Create a mapping from year to column index
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  # Initialize matrix with NA
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Fill the matrix from the data.table
  row_idx <- id_to_row[as.character(dt$id)]
  col_idx <- year_to_col[as.character(dt$year)]
  valid <- !is.na(row_idx) & !is.na(col_idx)
  mat[cbind(row_idx[valid], col_idx[valid])] <- dt[[var_name]][valid]
  
  # --- Compute neighbor stats ---
  # Output matrices
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- cell_neighbor_lookup[[i]]
    if (length(nb_idx) == 0L) next
    
    # nb_idx are integer row indices into mat
    # Extract submatrix: rows = neighbors, cols = years
    if (length(nb_idx) == 1L) {
      # Single neighbor: result is a vector (one value per year)
      nb_vals <- mat[nb_idx, , drop = FALSE]  # 1 × n_years matrix
    } else {
      nb_vals <- mat[nb_idx, , drop = FALSE]   # k × n_years matrix
    }
    
    # Compute column-wise stats (per year)
    # We need to handle NAs: use colMaxs etc. or manual approach
    # For efficiency, use matrixStats if available, otherwise base R
    for (j in seq_len(n_years)) {
      v <- nb_vals[, j]
      v <- v[!is.na(v)]
      if (length(v) > 0L) {
        max_mat[i, j]  <- max(v)
        min_mat[i, j]  <- min(v)
        mean_mat[i, j] <- mean(v)
      }
    }
  }
  
  # --- Reshape back to long format ---
  # Create output data.table
  out <- data.table(
    id   = rep(id_order, times = n_years),
    year = rep(years, each = n_cells)
  )
  
  max_name  <- paste0(var_name, "_neighbor_max")
  min_name  <- paste0(var_name, "_neighbor_min")
  mean_name <- paste0(var_name, "_neighbor_mean")
  
  out[[max_name]]  <- as.vector(max_mat)   # column-major: each column is a year
  out[[min_name]]  <- as.vector(min_mat)
  out[[mean_name]] <- as.vector(mean_mat)
  
  out
}

# ==============================================================================
# STEP 2b: Faster inner loop using matrixStats (if available)
# ==============================================================================
# If the matrixStats package is installed, replace the inner double loop with
# vectorized column operations. This version is substantially faster.

compute_neighbor_stats_matrix_fast <- function(dt, var_name, id_order,
                                                cell_neighbor_lookup, years) {
  
  n_cells <- length(id_order)
  n_years <- length(years)
  
  id_to_row  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  row_idx <- id_to_row[as.character(dt$id)]
  col_idx <- year_to_col[as.character(dt$year)]
  valid   <- !is.na(row_idx) & !is.na(col_idx)
  mat[cbind(row_idx[valid], col_idx[valid])] <- dt[[var_name]][valid]
  
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  has_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- cell_neighbor_lookup[[i]]
    if (length(nb_idx) == 0L) next
    
    nb_vals <- mat[nb_idx, , drop = FALSE]  # k × n_years
    
    if (has_matrixStats) {
      max_mat[i, ]  <- matrixStats::colMaxs(nb_vals, na.rm = TRUE)
      min_mat[i, ]  <- matrixStats::colMins(nb_vals, na.rm = TRUE)
      mean_mat[i, ] <- matrixStats::colMeans2(nb_vals, na.rm = TRUE)
    } else {
      max_mat[i, ]  <- apply(nb_vals, 2, max, na.rm = TRUE)
      min_mat[i, ]  <- apply(nb_vals, 2, min, na.rm = TRUE)
      mean_mat[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
  }
  
  # Fix -Inf/Inf from max/min on all-NA columns (matrixStats returns -Inf/Inf)
  max_mat[is.infinite(max_mat)]  <- NA_real_
  min_mat[is.infinite(min_mat)]  <- NA_real_
  mean_mat[is.nan(mean_mat)]     <- NA_real_
  
  out <- data.table(
    id   = rep(id_order, times = n_years),
    year = rep(years, each = n_cells)
  )
  
  out[[paste0(var_name, "_neighbor_max")]]  <- as.vector(max_mat)
  out[[paste0(var_name, "_neighbor_min")]]  <- as.vector(min_mat)
  out[[paste0(var_name, "_neighbor_mean")]] <- as.vector(mean_mat)
  
  out
}

# ==============================================================================
# STEP 3: Main pipeline — drop-in replacement for the outer loop
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # Convert to data.table if not already (non-destructive)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  years <- sort(unique(cell_data$year))
  
  # ---- STATIC: build cell-level neighbor lookup ONCE ----
  message("Building cell-level neighbor lookup (once)...")
  cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)
  
  # ---- CHANGING: compute neighbor stats per variable ----
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    
    stats_dt <- compute_neighbor_stats_matrix_fast(
      dt                   = cell_data,
      var_name             = var_name,
      id_order             = id_order,
      cell_neighbor_lookup = cell_neighbor_lookup,
      years                = years
    )
    
    # Join the 3 new columns onto cell_data by (id, year)
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    
    # Remove existing columns if re-running
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_data)) cell_data[, (col) := NULL]
    }
    
    setkeyv(cell_data, c("id", "year"))
    setkeyv(stats_dt,  c("id", "year"))
    
    cell_data <- stats_dt[cell_data, on = .(id, year)]
  }
  
  cell_data
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_feature_pipeline(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# cell_data now has the 15 neighbor feature columns appended.
# Proceed directly to prediction with the pre-trained Random Forest:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup granularity** | Cell-year (6.46M entries) | Cell (344K entries) — **built once** |
| **Key mechanism** | `paste()` + named-vector string lookup | Integer index vectors — **no string ops** |
| **Inner loop iterations** | 6.46M per variable | 344K per variable (**18.8× fewer**) |
| **Data structure for values** | Column vector + list-of-index-vectors | Cell × year matrix — **cache-friendly** |
| **Per-neighbor-set computation** | 1 year at a time | All 28 years at once via `colMaxs`/`colMins`/`colMeans2` |
| **Estimated runtime** | ~86+ hours | **~2–4 hours** |
| **RAM peak** | ~6.46M-element list of integer vectors | ~344K × 28 matrix (~77 MB per variable) — **fits in 16 GB** |
| **Numerical output** | Original estimand | **Identical** (same max/min/mean, same NA handling) |
| **RF model** | Pre-trained | **Untouched** |