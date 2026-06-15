 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination, producing ~6.46M list entries instead of ~344K.

2. **String-key hashing is expensive.** `paste(id, year)` is called millions of times, and named-vector lookups via character keys over a 6.46M-element vector are O(n) or at best O(1) with large hash overhead.

3. **The `lapply` over 6.46M rows in `compute_neighbor_stats`** iterates row-by-row in R, extracting and summarizing small vectors — this is the classic "R loop" anti-pattern.

4. **The entire pipeline repeats this for 5 variables**, compounding the cost.

**In summary:** The code fails to exploit the fact that neighbor structure is year-invariant. It conflates the static graph with the dynamic panel, leading to ~6.46M list lookups instead of ~344K, and row-level R loops instead of vectorized matrix operations.

## Optimization Strategy

**Separate static topology from dynamic data:**

1. **Build the neighbor lookup once, over cells only (~344K entries).** Each entry maps a cell index to its neighbor cell indices. This is year-independent.

2. **Reshape each variable into a matrix: cells × years.** With ~344K rows and 28 columns, this is small (~77 MB per double variable).

3. **Compute neighbor stats via vectorized matrix operations.** For each cell, gather neighbor rows from the matrix, then compute `max`, `min`, `mean` across neighbors for each year — all vectorized across the year dimension.

4. **Use `data.table` for efficient reshaping and joining** back to the panel.

This reduces the problem from 6.46M row-level operations to ~344K cell-level operations, each working on a 28-element year vector, with heavy use of vectorized C-level R functions.

**Expected speedup:** From ~86+ hours to roughly 10–30 minutes.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert to data.table if not already
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Preserve original row order for final reassembly
cell_data[, .row_order := .I]

# ============================================================
# STEP 1: Build STATIC neighbor lookup (cells only, year-free)
#
# rook_neighbors_unique: spdep nb object, indexed by position
#   in id_order (a vector of all unique cell IDs).
# We convert it to a simple list: cell_position -> neighbor_positions
# This is already what rook_neighbors_unique is, so we just
# ensure it's a clean integer list.
# ============================================================
# id_order is the vector of unique cell IDs matching the nb object
n_cells <- length(id_order)
stopifnot(n_cells == length(rook_neighbors_unique))

# Create a mapping from cell ID to positional index
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# The nb object already gives neighbor positions; just sanitize
# (spdep nb objects use 0L to mean "no neighbors")
neighbor_positions <- lapply(rook_neighbors_unique, function(nb) {
  nb <- as.integer(nb)
  nb[nb > 0L]
})

# ============================================================
# STEP 2: Get sorted unique years
# ============================================================
years_all <- sort(unique(cell_data$year))
n_years   <- length(years_all)
year_to_col <- setNames(seq_along(years_all), as.character(years_all))

# ============================================================
# STEP 3: Assign each cell its positional index
# ============================================================
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Verify completeness (balanced panel assumed)
stopifnot(all(!is.na(cell_data$cell_pos)))

# ============================================================
# STEP 4: Function to compute neighbor stats for one variable
#
# Strategy:
#   - Reshape variable into a matrix [n_cells x n_years]
#   - For each cell, pull neighbor rows, compute col-wise
#     max/min/mean across neighbors
#   - Return a matrix [n_cells x n_years] for each stat
#   - Melt back and join to cell_data
# ============================================================
compute_neighbor_features_fast <- function(dt, var_name, id_order,
                                           neighbor_positions,
                                           years_all, year_to_col) {
  n_cells <- length(id_order)
  n_years <- length(years_all)

  # --- Build the variable matrix [cell_pos, year_col] ---
  # Use keyed data.table for fast extraction
  setkeyv(dt, c("cell_pos", "year"))
  
  var_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Vectorized fill: get cell_pos, year_col, and value
  pos_vec  <- dt$cell_pos
  ycol_vec <- year_to_col[as.character(dt$year)]
  val_vec  <- dt[[var_name]]
  
  # Fill matrix (this is vectorized via linear indexing)
  linear_idx <- (ycol_vec - 1L) * n_cells + pos_vec
  var_mat[linear_idx] <- val_vec

  # --- Compute neighbor stats ---
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb <- neighbor_positions[[i]]
    if (length(nb) == 0L) next
    
    # Extract neighbor rows: a sub-matrix [length(nb) x n_years]
    nb_vals <- var_mat[nb, , drop = FALSE]
    
    if (length(nb) == 1L) {
      # Single neighbor: stats are trivial
      max_mat[i, ]  <- nb_vals[1L, ]
      min_mat[i, ]  <- nb_vals[1L, ]
      mean_mat[i, ] <- nb_vals[1L, ]
    } else {
      # colMins/colMaxs/colMeans — use matrixStats if available,
      # otherwise base R
      max_mat[i, ]  <- apply(nb_vals, 2L, max, na.rm = TRUE)
      min_mat[i, ]  <- apply(nb_vals, 2L, min, na.rm = TRUE)
      mean_mat[i, ] <- colMeans(nb_vals, na.rm = TRUE)
      
      # Fix columns where all neighbors were NA
      all_na <- colSums(!is.na(nb_vals)) == 0L
      if (any(all_na)) {
        max_mat[i, all_na]  <- NA_real_
        min_mat[i, all_na]  <- NA_real_
        mean_mat[i, all_na] <- NA_real_
      }
    }
  }

  # --- Flatten back to panel format ---
  # Extract values using the same linear indexing
  max_col_name  <- paste0(var_name, "_neighbor_max")
  min_col_name  <- paste0(var_name, "_neighbor_min")
  mean_col_name <- paste0(var_name, "_neighbor_mean")

  dt[, (max_col_name)  := max_mat[linear_idx]]
  dt[, (min_col_name)  := min_mat[linear_idx]]
  dt[, (mean_col_name) := mean_mat[linear_idx]]

  invisible(dt)
}

# ============================================================
# STEP 5 (OPTIONAL): Use matrixStats for much faster col ops
# ============================================================
# If matrixStats is available, replace the inner loop body:
use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)

if (use_matrixStats) {
  library(matrixStats)
  
  compute_neighbor_features_fast <- function(dt, var_name, id_order,
                                             neighbor_positions,
                                             years_all, year_to_col) {
    n_cells <- length(id_order)
    n_years <- length(years_all)
    
    # Build variable matrix
    var_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    pos_vec  <- dt$cell_pos
    ycol_vec <- year_to_col[as.character(dt$year)]
    val_vec  <- dt[[var_name]]
    linear_idx <- (ycol_vec - 1L) * n_cells + pos_vec
    var_mat[linear_idx] <- val_vec
    
    # Compute neighbor stats
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nb <- neighbor_positions[[i]]
      if (length(nb) == 0L) next
      
      nb_vals <- var_mat[nb, , drop = FALSE]
      
      if (length(nb) == 1L) {
        max_mat[i, ]  <- nb_vals[1L, ]
        min_mat[i, ]  <- nb_vals[1L, ]
        mean_mat[i, ] <- nb_vals[1L, ]
      } else {
        max_mat[i, ]  <- colMaxs(nb_vals, na.rm = TRUE)
        min_mat[i, ]  <- colMins(nb_vals, na.rm = TRUE)
        mean_mat[i, ] <- colMeans2(nb_vals, na.rm = TRUE)
        
        all_na <- colAlls(is.na(nb_vals))
        if (any(all_na)) {
          max_mat[i, all_na]  <- NA_real_
          min_mat[i, all_na]  <- NA_real_
          mean_mat[i, all_na] <- NA_real_
        }
      }
    }
    
    max_col_name  <- paste0(var_name, "_neighbor_max")
    min_col_name  <- paste0(var_name, "_neighbor_min")
    mean_col_name <- paste0(var_name, "_neighbor_mean")
    
    dt[, (max_col_name)  := max_mat[linear_idx]]
    dt[, (min_col_name)  := min_mat[linear_idx]]
    dt[, (mean_col_name) := mean_mat[linear_idx]]
    
    invisible(dt)
  }
}

# ============================================================
# STEP 6: Run for all neighbor source variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(Sys.time(), " — Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_fast(
    dt                 = cell_data,
    var_name           = var_name,
    id_order           = id_order,
    neighbor_positions = neighbor_positions,
    years_all          = years_all,
    year_to_col        = year_to_col
  )
}

# ============================================================
# STEP 7: Restore original row order and clean up helper column
# ============================================================
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]
# Optionally remove helper column before prediction:
# cell_data[, cell_pos := NULL]

# ============================================================
# STEP 8: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# cell_data now contains the same 15 neighbor feature columns
# (5 vars × {max, min, mean}) with identical numerical values.
#
# Example:
# cell_data$predicted <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup size** | ~6.46M entries (cell×year) | ~344K entries (cell only) |
| **Key mechanism** | Character paste + named-vector hash | Integer positional indexing |
| **Stats computation** | R `lapply` over 6.46M rows, per variable | Loop over 344K cells, each doing vectorized column ops on a [neighbors × 28] matrix |
| **Total inner iterations** | ~32.3M (6.46M × 5 vars) | ~1.72M (344K × 5 vars), each doing vectorized 28-year ops |
| **Memory** | Large character key vectors | 5 matrices of ~77 MB each (~385 MB total) |
| **Estimated time** | 86+ hours | ~10–30 minutes |
| **Numerical output** | — | Identical (same max/min/mean over same neighbor values) |
| **RF model** | — | Untouched; used only for prediction |