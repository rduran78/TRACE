 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row neighbor indices for all ~6.46 million cell-year rows.** It does this by pasting `id_year` keys and looking them up in a named vector. But the neighbor *relationships* are purely spatial — cell A is a neighbor of cell B regardless of year. The topology is static across all 28 years. The function needlessly recomputes 6.46M lists when only 344,208 cell-level lists are needed.

2. **`compute_neighbor_stats` indexes into the full 6.46M-row data frame using the bloated per-row lookup.** Because the lookup was built at the cell-year level, every stats computation carries the overhead of the inflated structure.

3. **String-based key construction (`paste(id, year, sep="_")`) and named-vector lookups (`setNames`, `idx_lookup[neighbor_keys]`)** are extremely slow at this scale — millions of string allocations, hashing, and named-vector searches.

4. **`lapply` over 6.46M rows** with per-element R function calls creates massive interpreter overhead.

### Summary

| Aspect | Current | Optimal |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344,208 cells (once) |
| Lookup rebuild per year? | Implicitly yes (embedded) | No — static, built once |
| Key mechanism | String paste + named vector | Integer index matrix |
| Stats loop | `lapply` over 6.46M rows | Vectorized matrix ops per year |
| Estimated time | 86+ hours | ~2–5 minutes |

---

## Optimization Strategy

**Principle: Separate the static neighbor graph from the dynamic yearly variable values.**

1. **Build a cell-level neighbor lookup once** — a list of length 344,208 where each element contains integer indices into the *cell-order* vector (not the data frame). This is topology and never changes.

2. **For each year, extract the variable column as a vector indexed by cell order.** Use the static cell-level neighbor lookup to compute max/min/mean via vectorized operations over that vector.

3. **Use `data.table` for fast split-by-year and column assignment**, avoiding copies.

4. **Pre-build a sparse neighbor matrix (or padded neighbor matrix) to fully vectorize** the neighbor aggregation, eliminating all `lapply` calls over millions of rows.

The trained Random Forest model is never touched. The output columns (`*_neighbor_max`, `*_neighbor_min`, `*_neighbor_mean`) are numerically identical to the original implementation.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 1: Build the STATIC cell-level neighbor lookup (done ONCE)
# ==============================================================================
# Inputs:
#   id_order            — vector of 344,208 cell IDs in the canonical order
#   rook_neighbors_unique — spdep nb object (list of length 344,208)
#
# This maps each cell (by its position in id_order) to the positions of its
# neighbors in id_order. This is pure topology — no year dependency.
# ==============================================================================

build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is an nb object: neighbors[[i]] gives integer indices into
  # id_order for the neighbors of id_order[i].
  # We just need to ensure 0-neighbor cells return integer(0).
  n <- length(id_order)
  lookup <- vector("list", n)
  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(nb_idx) == 1L && nb_idx[1] == 0L) {
      lookup[[i]] <- integer(0)
    } else {
      lookup[[i]] <- as.integer(nb_idx)
    }
  }
  lookup
}

# Build it once
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# ==============================================================================
# STEP 2: Pre-build a padded neighbor matrix for fully vectorized operations
# ==============================================================================
# Convert the ragged list into a fixed-width integer matrix (n_cells x max_neighbors).
# Cells with fewer neighbors get NA padding.
# This enables vectorized matrix-column extraction instead of lapply.
# ==============================================================================

build_padded_neighbor_matrix <- function(cell_neighbor_lookup) {
  max_nb <- max(vapply(cell_neighbor_lookup, length, integer(1)))
  n <- length(cell_neighbor_lookup)
  mat <- matrix(NA_integer_, nrow = n, ncol = max_nb)
  for (i in seq_len(n)) {
    nb <- cell_neighbor_lookup[[i]]
    if (length(nb) > 0L) {
      mat[i, seq_along(nb)] <- nb
    }
  }
  mat
}

nb_matrix <- build_padded_neighbor_matrix(cell_neighbor_lookup)
# nb_matrix: 344,208 rows x max_neighbors cols (typically 4 for rook)

# ==============================================================================
# STEP 3: Vectorized neighbor stats computation for one variable, one year
# ==============================================================================
# Given a numeric vector of values (length = n_cells, ordered by id_order)
# and the padded neighbor matrix, compute max/min/mean across neighbors
# entirely with vectorized matrix operations.
# ==============================================================================

compute_neighbor_stats_vectorized <- function(vals, nb_matrix) {
  # vals: numeric vector of length n_cells (one value per cell for one year)
  # nb_matrix: integer matrix (n_cells x max_neighbors), indices into vals
  
  n <- length(vals)
  k <- ncol(nb_matrix)
  
  # Build a matrix of neighbor values: n_cells x max_neighbors
  # Use vals[nb_matrix], which vectorizes the lookup.
  # NA indices (padding) will produce NA values — correct behavior.
  nb_vals <- matrix(vals[nb_matrix], nrow = n, ncol = k)
  
  # Compute row-wise stats, ignoring NAs
  # For cells with ALL neighbors NA (no neighbors or all neighbor vals NA),
  # these will return appropriate NA/NaN — we fix below.
  nb_max  <- apply(nb_vals, 1, max,  na.rm = TRUE)
  nb_min  <- apply(nb_vals, 1, min,  na.rm = TRUE)
  nb_mean <- rowMeans(nb_vals, na.rm = TRUE)  # fast C-level
  
  # Fix Inf/-Inf from max/min on all-NA rows
  nb_max[is.infinite(nb_max)] <- NA_real_
  nb_min[is.infinite(nb_min)] <- NA_real_
  # rowMeans already returns NaN for all-NA rows; convert to NA
  nb_mean[is.nan(nb_mean)] <- NA_real_
  
  data.table(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# ==============================================================================
# STEP 4: Optimized alternative using matrixStats for even faster row ops
# ==============================================================================
# If matrixStats is available, use rowMaxs/rowMins for C-level speed.
# Falls back to apply() otherwise.
# ==============================================================================

if (requireNamespace("matrixStats", quietly = TRUE)) {
  compute_neighbor_stats_fast <- function(vals, nb_matrix) {
    n <- length(vals)
    k <- ncol(nb_matrix)
    nb_vals <- matrix(vals[nb_matrix], nrow = n, ncol = k)
    
    nb_max  <- matrixStats::rowMaxs(nb_vals, na.rm = TRUE)
    nb_min  <- matrixStats::rowMins(nb_vals, na.rm = TRUE)
    nb_mean <- rowMeans(nb_vals, na.rm = TRUE)
    
    nb_max[is.infinite(nb_max)] <- NA_real_
    nb_min[is.infinite(nb_min)] <- NA_real_
    nb_mean[is.nan(nb_mean)]    <- NA_real_
    
    data.table(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
  }
} else {
  compute_neighbor_stats_fast <- compute_neighbor_stats_vectorized
}

# ==============================================================================
# STEP 5: Main loop — iterate over variables, iterate over years
# ==============================================================================
# Convert cell_data to data.table for fast operations.
# For each variable and each year:
#   1. Extract the variable values in id_order for that year.
#   2. Compute vectorized neighbor stats (on 344K cells, not 6.46M rows).
#   3. Write results back into the corresponding rows of cell_data.
# ==============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build a mapping from cell ID to position in id_order (static)
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Ensure cell_data has a column mapping each row to its cell position in id_order
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Get sorted unique years
years <- sort(unique(cell_data$year))

# Neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  if (is.null(cell_data[[col_max]]))  set(cell_data, j = col_max,  value = NA_real_)
  if (is.null(cell_data[[col_min]]))  set(cell_data, j = col_min,  value = NA_real_)
  if (is.null(cell_data[[col_mean]])) set(cell_data, j = col_mean, value = NA_real_)
}

# Key the data.table for fast subsetting by year
setkey(cell_data, year)

# Main computation loop
n_cells <- length(id_order)

for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  t0 <- proc.time()
  
  for (yr in years) {
    # Get row indices for this year (fast via key)
    yr_rows <- cell_data[.(yr), which = TRUE]
    
    # Extract cell positions and variable values for this year
    yr_cell_pos <- cell_data$cell_pos[yr_rows]
    yr_vals_raw <- cell_data[[var_name]][yr_rows]
    
    # Build a full-length vector indexed by cell position
    # (handles case where not all cells appear in every year)
    vals_by_cell <- rep(NA_real_, n_cells)
    vals_by_cell[yr_cell_pos] <- yr_vals_raw
    
    # Compute neighbor stats — vectorized over all 344,208 cells
    stats <- compute_neighbor_stats_fast(vals_by_cell, nb_matrix)
    
    # Write results back — only for cells present in this year
    set(cell_data, i = yr_rows, j = col_max,  value = stats$nb_max[yr_cell_pos])
    set(cell_data, i = yr_rows, j = col_min,  value = stats$nb_min[yr_cell_pos])
    set(cell_data, i = yr_rows, j = col_mean, value = stats$nb_mean[yr_cell_pos])
  }
  
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("  Done in %.1f seconds\n", elapsed))
}

# Clean up helper column
cell_data[, cell_pos := NULL]

# ==============================================================================
# STEP 6: Proceed to prediction with the pre-trained Random Forest
# ==============================================================================
# The cell_data now has all *_neighbor_max, *_neighbor_min, *_neighbor_mean
# columns populated with numerically identical values to the original code.
# The trained RF model is unchanged — proceed directly to predict().
#
# Example:
#   predictions <- predict(trained_rf_model, newdata = cell_data)
# ==============================================================================

cat("Neighbor feature computation complete. Ready for Random Forest prediction.\n")
```

---

## Performance Analysis

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-key lists | 344K integer lists (once) + padded matrix |
| Per-variable-year computation | `lapply` over 6.46M rows | Vectorized matrix ops on 344K cells |
| Total iterations (inner) | 6.46M × 5 vars = 32.3M | 344K × 5 vars × 28 yrs = 48.2M cells, but vectorized |
| String operations | ~19.4M `paste()` calls | Zero |
| Estimated wall time | 86+ hours | **2–5 minutes** |
| Peak memory | Bloated lookup lists | ~344K × 4 matrix (5.5 MB) + data.table in-place |
| Numerical result | Reference | **Identical** |

### Why it's correct (preserves the estimand)

- For each cell-year row, the neighbor max/min/mean is computed over exactly the same set of rook-neighbor cells for the same year, using the same variable values.
- The padded matrix `nb_matrix` encodes exactly the same adjacency as `rook_neighbors_unique`.
- `vals[nb_matrix]` with NA padding + `na.rm = TRUE` produces identical results to the original's `neighbor_vals[!is.na(neighbor_vals)]` filtering.
- The pre-trained Random Forest model is never modified — only `predict()` is called on the enriched data.