 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

1. **O(N) string construction repeated N times inside `lapply`**: `build_neighbor_lookup` creates `idx_lookup` (a named character vector of ~6.46M paste keys) once, but then for each of the 6.46M rows, it calls `paste()` to build `neighbor_keys` and performs named vector lookup against that 6.46M-element vector. Named vector lookup in R is **O(k)** per query (hash-based, but with overhead per call), and you're doing it ~6.46M × avg_neighbors times. The string allocation and hashing alone is enormous.

2. **The neighbor lookup is year-invariant but rebuilt per cell-year row**: Rook neighbors are a spatial relationship — they don't change across years. Yet the inner function iterates over every *cell-year* row (6.46M), rediscovering the same set of neighbors for the same cell 28 times (once per year). The lookup structure should be built at the **cell level** (344K cells), not the cell-year level.

3. **`compute_neighbor_stats` is called sequentially per variable**: Each of the 5 variables triggers a full pass over 6.46M rows. This is fine algorithmically (O(N×k) per variable), but the data access pattern can be consolidated.

**Root cause**: The algorithm conflates *spatial* neighbor resolution (which is year-invariant) with *panel* row resolution. This turns a 344K-cell spatial problem into a 6.46M-row string-matching problem, multiplied by 28 years of redundancy.

## Optimization Strategy

| Aspect | Current | Proposed |
|---|---|---|
| Neighbor resolution level | Per cell-year row (6.46M) | Per cell (344K), then broadcast to years |
| Key mechanism | String paste + named vector lookup | Integer index arithmetic (zero string ops) |
| Year mapping | Implicit in string keys | Explicit integer offset matrix |
| Row iteration | `lapply` over 6.46M rows in R | Vectorized matrix operations, no row-level R loop |
| Stats computation | `lapply` row-by-row with subsetting | Vectorized column operations on pre-built index matrices |
| Estimated time | 86+ hours | Minutes |

**Core insight**: If the data is sorted by `(id, year)` and every cell has a complete 28-year panel, then the row index for cell `c` in year `y` is simply `(c - 1) * 28 + (y - 1992 + 1)`. Neighbor indices for *all* years of a cell can be computed by integer arithmetic — no strings, no hash lookups.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Preserves original numerical estimand; no model retraining needed.
# =============================================================================

build_neighbor_features_optimized <- function(cell_data,
                                               rook_neighbors_unique,
                                               neighbor_source_vars,
                                               id_order) {
  # ------------------------------------------------------------------
  # 0. Validate and prepare: ensure data is sorted by (id, year)
  # ------------------------------------------------------------------
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]
  
  unique_ids   <- sort(unique(cell_data$id))
  unique_years <- sort(unique(cell_data$year))
  n_cells      <- length(unique_ids)
  n_years      <- length(unique_years)
  N             <- nrow(cell_data)
  
  stopifnot(N == n_cells * n_years)  # balanced panel required
  
  cat(sprintf("Panel: %d cells × %d years = %d rows\n", n_cells, n_years, N))
  
  # ------------------------------------------------------------------
  # 1. Build integer cell-index mapping (no strings)
  # ------------------------------------------------------------------
  # id_to_cell_idx: maps cell id -> sequential cell index 1..n_cells
  # This must align with the sorted unique_ids AND with the nb object.
  
  # id_order is the vector of cell IDs in the order matching rook_neighbors_unique.
  # So rook_neighbors_unique[[j]] gives neighbor indices into id_order.
  
  # We need: for each cell in our sorted data, which index in id_order is it?
  id_order_to_nb_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each unique_id (sorted), find its nb-object index
  # Also, row offset: cell with sorted index `s` occupies rows
  #   ((s-1)*n_years + 1) : (s * n_years)
  # Within that block, year `y` is at local offset (y - min_year + 1).
  
  # Map from unique_ids (sorted) -> position in sorted order
  sorted_id_to_sidx <- setNames(seq_along(unique_ids), as.character(unique_ids))
  
  # ------------------------------------------------------------------
  # 2. Build neighbor row-index matrix (integer arithmetic, no strings)
  #    For each cell, find all neighbor cells, then expand to all years.
  #    Store as a list of integer vectors (row indices into cell_data).
  # ------------------------------------------------------------------
  cat("Building integer neighbor index lists per cell-year...\n")
  
  # Pre-compute: for each sorted-cell-index, the list of neighbor sorted-cell-indices
  # Step A: sorted_id -> nb_idx -> neighbor nb_idxs -> neighbor id_order ids -> neighbor sorted_idxs
  
  # Vectorized mapping: for each unique_id, get its nb index
  nb_idx_per_sorted <- id_order_to_nb_idx[as.character(unique_ids)]
  # nb_idx_per_sorted[s] = index into rook_neighbors_unique for sorted cell s
  
  # Pre-compute neighbor sorted indices for each cell (year-invariant)
  # This is the only list operation, over 344K cells, not 6.46M rows.
  cell_neighbor_sorted_idx <- vector("list", n_cells)
  
  for (s in seq_len(n_cells)) {
    nb_i <- nb_idx_per_sorted[s]
    if (is.na(nb_i)) {
      cell_neighbor_sorted_idx[[s]] <- integer(0)
      next
    }
    nb_cell_nb_idxs <- rook_neighbors_unique[[nb_i]]
    if (length(nb_cell_nb_idxs) == 0L || (length(nb_cell_nb_idxs) == 1L && nb_cell_nb_idxs[1] == 0L)) {
      cell_neighbor_sorted_idx[[s]] <- integer(0)
      next
    }
    # Convert nb-object indices -> cell ids -> sorted indices
    nb_ids <- id_order[nb_cell_nb_idxs]
    nb_sorted <- sorted_id_to_sidx[as.character(nb_ids)]
    cell_neighbor_sorted_idx[[s]] <- as.integer(nb_sorted[!is.na(nb_sorted)])
  }
  
  cat("Cell-level neighbor lists built.\n")
  
  # ------------------------------------------------------------------
  # 3. Compute neighbor stats variable-by-variable using matrix reshaping
  #    Reshape each variable into a (n_cells × n_years) matrix.
  #    For each cell, gather neighbor rows from the matrix, compute stats.
  #    This avoids per-row R iteration over 6.46M rows.
  # ------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))
    
    # Reshape to matrix: rows = cells (sorted), cols = years
    vals_vec <- cell_data[[var_name]]
    val_mat  <- matrix(vals_vec, nrow = n_cells, ncol = n_years, byrow = TRUE)
    # val_mat[s, t] = value for sorted-cell s in year t
    
    # Initialize result matrices
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # For each cell, extract neighbor sub-matrix and compute column-wise stats
    # This iterates 344K times (not 6.46M), and each iteration is vectorized
    # over years via column operations on small matrices.
    
    for (s in seq_len(n_cells)) {
      nb_idx <- cell_neighbor_sorted_idx[[s]]
      if (length(nb_idx) == 0L) next
      
      if (length(nb_idx) == 1L) {
        # Single neighbor: just a vector (one row)
        nb_vals <- val_mat[nb_idx, , drop = FALSE]  # 1 × n_years matrix
        max_mat[s, ]  <- nb_vals[1, ]
        min_mat[s, ]  <- nb_vals[1, ]
        mean_mat[s, ] <- nb_vals[1, ]
      } else {
        # Multiple neighbors: sub-matrix of (k_neighbors × n_years)
        nb_vals <- val_mat[nb_idx, , drop = FALSE]
        
        # Column-wise stats (each column = one year, vectorized across neighbors)
        # Using colMeans / apply for max/min
        # For handling NAs properly:
        max_mat[s, ]  <- apply(nb_vals, 2, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x)
        })
        min_mat[s, ]  <- apply(nb_vals, 2, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x)
        })
        mean_mat[s, ] <- apply(nb_vals, 2, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x)
        })
      }
    }
    
    # Flatten matrices back to vectors (row-major to match sorted cell_data)
    cell_data[[paste0("max_nbr_",  var_name)]]  <- as.vector(t(max_mat))
    cell_data[[paste0("min_nbr_",  var_name)]]  <- as.vector(t(min_mat))
    cell_data[[paste0("mean_nbr_", var_name)]]  <- as.vector(t(mean_mat))
    
    cat(sprintf("  Done: %s\n", var_name))
  }
  
  cat("All neighbor features built.\n")
  return(cell_data)
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

# cell_data <- build_neighbor_features_optimized(
#   cell_data              = cell_data,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   id_order               = id_order
# )
#
# # The trained Random Forest model is unchanged — just feed the updated cell_data.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Further Optimization: Eliminating `apply` with `matrixStats`

The inner `apply(..., 2, ...)` for each cell is still an R-level loop over columns. For cells with many neighbors, we can use the **`matrixStats`** package for true vectorized column operations:

```r
# install.packages("matrixStats") if needed
library(matrixStats)

# Replace the inner-loop body with:
for (s in seq_len(n_cells)) {
  nb_idx <- cell_neighbor_sorted_idx[[s]]
  if (length(nb_idx) == 0L) next
  
  nb_vals <- val_mat[nb_idx, , drop = FALSE]  # k × 28 matrix
  
  max_mat[s, ]  <- colMaxs(nb_vals,  na.rm = TRUE)
  min_mat[s, ]  <- colMins(nb_vals,  na.rm = TRUE)
  mean_mat[s, ] <- colMeans2(nb_vals, na.rm = TRUE)
}

# Note: colMaxs/colMins return -Inf/Inf for all-NA columns.
# Post-process:
max_mat[is.infinite(max_mat)]   <- NA_real_
min_mat[is.infinite(min_mat)]   <- NA_real_
mean_mat[is.nan(mean_mat)]      <- NA_real_
```

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| String keys created | ~6.46M (idx_lookup) + ~6.46M × k (neighbor_keys) | **0** |
| Hash lookups | ~6.46M × k per variable | **0** |
| R-level loop iterations | 6.46M × 5 vars = 32.3M | 344K × 5 = 1.72M |
| Work per iteration | String paste + hash + subset + stats | Integer subset of 28-col matrix + column stats |
| **Estimated wall time** | **86+ hours** | **5–20 minutes** |
| RAM delta | Moderate (string allocations) | ~6.46M × 3 doubles per var ≈ 0.15 GB per var, peaks ~1 GB |

The numerical results are **identical** because the same (cell, year) → neighbor (cell, year) mapping is resolved, the same values are gathered, and the same `max`/`min`/`mean` with `na.rm` logic is applied. The trained Random Forest model requires no changes.