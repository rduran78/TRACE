 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows
For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(id, year)`) for each neighbor-year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which are hash-based but repeated millions of times with string allocation overhead.

This is fundamentally **year-redundant**: the neighbor topology is purely spatial and identical across all 28 years, yet the lookup is rebuilt per row (per cell-year), creating ~6.46M list entries instead of ~344K.

### Bottleneck 2: `compute_neighbor_stats` — R-level `lapply` over 6.46M elements
For each of the ~6.46 million rows, an R-level anonymous function extracts neighbor values, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable), totaling ~32.3M R function invocations with per-element vector subsetting. The R interpreter overhead here is enormous.

### Why 86+ hours?
- ~6.46M list constructions in `build_neighbor_lookup` with string operations.
- ~32.3M R-level loop iterations in `compute_neighbor_stats` (6.46M × 5 vars).
- All computation is single-threaded, interpreted R, with heavy allocation of temporary character/numeric vectors.

### Why raster focal/kernel operations don't directly apply
Focal operations assume a regular grid with a fixed rectangular window. The data here is a panel (cell × year) stored as a long data frame, the neighbor structure is an irregular `spdep::nb` object (not a rectangular kernel), and the computation must be done **within-year** across spatial neighbors. While the analogy is instructive (focal = neighborhood summary), a direct `terra::focal` approach would require reshaping each variable into a raster stack per year and reconstructing the nb topology as a custom weight matrix — adding complexity without guaranteeing correctness for irregular neighbor structures at boundaries. The better strategy is to vectorize the existing logic.

---

## Optimization Strategy

### Strategy 1: Separate spatial topology from temporal indexing
Build the neighbor lookup **once per cell** (~344K entries), not once per cell-year (~6.46M entries). Then, for each year, use integer matrix indexing to gather neighbor values.

### Strategy 2: Vectorize with `data.table` and matrix operations
- Reshape data so that each year's values for a variable can be accessed as a column or matrix slice.
- Use `data.table` for fast grouped operations or pre-build an integer index matrix of neighbor row positions per year.
- Replace the `lapply` with vectorized `rowMaxs`, `rowMins`, `rowMeans` from the `matrixStats` package over a gathered neighbor-value matrix.

### Strategy 3: Pre-build a sparse neighbor-row-index matrix
For each cell-year row, we know which rows are its neighbors (same year, neighbor cell). We can encode this as a fixed-width integer matrix (each row has at most 4 rook neighbors). Then `max/min/mean` can be computed via `matrixStats::rowMaxs` etc., which are C-level vectorized.

### Expected speedup
- Eliminating per-row string operations: ~100–500×  
- Vectorized C-level `rowMaxs/rowMins/rowMeans`: ~50–200×  
- Estimated new runtime: **1–5 minutes** total (from 86+ hours).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, matrixStats
# install.packages(c("data.table", "matrixStats"))

library(data.table)
library(matrixStats)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -------------------------------------------------------------------------
  # STEP 1: Convert to data.table for fast indexed operations
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Preserve original row order for faithful output
  dt[, .roworder := .I]

  # -------------------------------------------------------------------------
  # STEP 2: Build spatial-only neighbor edge list (cell-level, not row-level)
  #
  # rook_neighbors_unique is an nb object: a list of length = # cells,
  # where element i contains integer indices (into id_order) of neighbors.
  # We expand this into an edge list of (cell_index, neighbor_cell_index).
  # -------------------------------------------------------------------------
  n_cells <- length(id_order)

  # Map cell IDs to their position in id_order (1-based index)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list: from_id -> to_id (using original cell IDs)
  edge_from <- integer(0)
  edge_to   <- integer(0)

  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) > 0L) {
      edge_from <- c(edge_from, rep(id_order[i], length(nb_idx)))
      edge_to   <- c(edge_to,   id_order[nb_idx])
    }
  }

  edges <- data.table(id = edge_from, neighbor_id = edge_to)

  # -------------------------------------------------------------------------
  # STEP 3: Determine max number of rook neighbors (should be <= 4)
  # -------------------------------------------------------------------------
  max_neighbors <- edges[, .N, by = id][, max(N)]
  cat("Max rook neighbors per cell:", max_neighbors, "\n")

  # -------------------------------------------------------------------------
  # STEP 4: Build a fixed-width neighbor ID matrix (n_cells x max_neighbors)
  #
  # For each cell, store the IDs of its neighbors padded with NA.
  # -------------------------------------------------------------------------
  # Assign a within-cell neighbor index
  edges[, nb_seq := seq_len(.N), by = id]

  # Create a lookup: for each unique cell ID, a row of neighbor IDs
  # Using dcast for a wide matrix
  neighbor_wide <- dcast(edges, id ~ nb_seq, value.var = "neighbor_id")
  # Columns: id, 1, 2, ..., max_neighbors
  nb_cols <- setdiff(names(neighbor_wide), "id")

  # Merge this into dt so every cell-year row knows its neighbor cell IDs
  # But this would replicate — instead, we'll take a more direct approach.

  # -------------------------------------------------------------------------
  # STEP 5: For each (cell-year) row, find neighbor rows via integer indexing
  #
  # Key insight: sort dt by (id, year) and build a row-lookup matrix.
  # All cells share the same set of years, so for a given cell at position p
  # in the cell list and year at position t in the year list, the row index
  # is: (p - 1) * n_years + t  (if data is sorted by id, then year).
  # -------------------------------------------------------------------------

  # Ensure keyed sort: id, year
  setkey(dt, id, year)

  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_pos <- setNames(seq_along(years), as.character(years))

  # Verify that the panel is balanced (every cell has every year)
  cells_per_year <- dt[, .N, by = id]
  if (!all(cells_per_year$N == n_years)) {
    warning("Panel is unbalanced. Falling back to hash-based row lookup.")
    # Build a hash-based lookup for unbalanced panels
    dt[, .rowidx := .I]
    row_lookup <- dt[, .(.rowidx), keyby = .(id, year)]
    balanced <- FALSE
  } else {
    balanced <- TRUE
  }

  # Unique cell IDs in sorted order (matches keyed dt)
  cell_ids_sorted <- sort(unique(dt$id))
  cell_id_to_pos  <- setNames(seq_along(cell_ids_sorted), as.character(cell_ids_sorted))

  # -------------------------------------------------------------------------
  # STEP 6: Build neighbor POSITION matrix (n_cells x max_neighbors)
  #   neighbor_pos_mat[p, k] = position (in cell_ids_sorted) of p's k-th neighbor
  # -------------------------------------------------------------------------
  neighbor_pos_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_neighbors)

  # Re-derive from edges more efficiently
  # For each cell in cell_ids_sorted, get neighbor positions
  for (i in seq_len(n_cells)) {
    cid <- cell_ids_sorted[i]
    # original position in id_order
    orig_pos <- id_to_pos[as.character(cid)]
    nb_idx <- rook_neighbors_unique[[orig_pos]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) > 0L) {
      nb_cids <- id_order[nb_idx]
      nb_positions <- cell_id_to_pos[as.character(nb_cids)]
      nb_positions <- nb_positions[!is.na(nb_positions)]
      if (length(nb_positions) > 0L) {
        neighbor_pos_mat[i, seq_along(nb_positions)] <- nb_positions
      }
    }
  }

  # -------------------------------------------------------------------------
  # STEP 7: For each variable, compute neighbor stats vectorized
  #
  # For a balanced panel sorted by (id, year):
  #   Row index of cell at position p, year at position t = (p-1)*n_years + t
  #
  # For each neighbor slot k (1..max_neighbors):
  #   neighbor_row[i] = (neighbor_pos_mat[cell_pos[i], k] - 1) * n_years + year_pos[i]
  #
  # We gather all neighbor values into a matrix (n_rows x max_neighbors),

  # then compute rowMaxs, rowMins, rowMeans with na.rm = TRUE.
  # -------------------------------------------------------------------------

  n_rows <- nrow(dt)

  if (balanced) {
    # Pre-compute cell_pos and year_pos for each row in sorted dt
    # Since sorted by (id, year) with n_years per cell:
    cell_pos_vec <- rep(seq_len(n_cells), each = n_years)
    year_pos_vec <- rep(seq_len(n_years), times = n_cells)

    for (var_name in neighbor_source_vars) {
      cat("Processing variable:", var_name, "\n")

      vals <- dt[[var_name]]

      # Build neighbor value matrix: n_rows x max_neighbors
      nb_val_mat <- matrix(NA_real_, nrow = n_rows, ncol = max_neighbors)

      for (k in seq_len(max_neighbors)) {
        # For each row i, the k-th neighbor's cell position
        nb_cell_pos <- neighbor_pos_mat[cell_pos_vec, k]
        # Convert to row index: (nb_cell_pos - 1) * n_years + year_pos_vec
        nb_row_idx <- (nb_cell_pos - 1L) * n_years + year_pos_vec
        # nb_row_idx is NA where there is no k-th neighbor
        valid <- !is.na(nb_row_idx)
        nb_val_mat[valid, k] <- vals[nb_row_idx[valid]]
      }

      # Compute stats using matrixStats (C-level, vectorized)
      nb_max  <- rowMaxs(nb_val_mat,  na.rm = TRUE)
      nb_min  <- rowMins(nb_val_mat,  na.rm = TRUE)
      nb_mean <- rowMeans2(nb_val_mat, na.rm = TRUE)

      # rowMaxs/rowMins return -Inf/Inf when all NA; convert to NA
      nb_max[is.infinite(nb_max)] <- NA_real_
      nb_min[is.infinite(nb_min)] <- NA_real_
      # rowMeans2 returns NaN for all-NA rows
      nb_mean[is.nan(nb_mean)] <- NA_real_

      # Add columns to dt
      set(dt, j = paste0("n_max_", var_name), value = nb_max)
      set(dt, j = paste0("n_min_", var_name), value = nb_min)
      set(dt, j = paste0("n_mean_", var_name), value = nb_mean)

      cat("  Done:", var_name, "\n")
    }
  } else {
    # ----- Unbalanced panel fallback (still much faster than original) -----
    # Build a row lookup: given (cell_pos, year_pos) -> row index in dt
    dt[, cell_pos := cell_id_to_pos[as.character(id)]]
    dt[, year_pos := year_to_pos[as.character(year)]]

    # Create a lookup matrix: n_cells x n_years -> row index
    row_idx_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
    row_idx_mat[cbind(dt$cell_pos, dt$year_pos)] <- seq_len(n_rows)

    for (var_name in neighbor_source_vars) {
      cat("Processing variable:", var_name, "\n")

      vals <- dt[[var_name]]
      nb_val_mat <- matrix(NA_real_, nrow = n_rows, ncol = max_neighbors)

      for (k in seq_len(max_neighbors)) {
        nb_cell_pos <- neighbor_pos_mat[dt$cell_pos, k]
        # Look up row index via the matrix
        nb_row_idx <- row_idx_mat[cbind(nb_cell_pos, dt$year_pos)]
        valid <- !is.na(nb_row_idx)
        nb_val_mat[valid, k] <- vals[nb_row_idx[valid]]
      }

      nb_max  <- rowMaxs(nb_val_mat,  na.rm = TRUE)
      nb_min  <- rowMins(nb_val_mat,  na.rm = TRUE)
      nb_mean <- rowMeans2(nb_val_mat, na.rm = TRUE)

      nb_max[is.infinite(nb_max)] <- NA_real_
      nb_min[is.infinite(nb_min)] <- NA_real_
      nb_mean[is.nan(nb_mean)]    <- NA_real_

      set(dt, j = paste0("n_max_", var_name), value = nb_max)
      set(dt, j = paste0("n_min_", var_name), value = nb_min)
      set(dt, j = paste0("n_mean_", var_name), value = nb_mean)

      cat("  Done:", var_name, "\n")
    }

    # Clean up temporary columns
    dt[, c("cell_pos", "year_pos", ".rowidx") :=
         .(NULL, NULL, NULL)]
  }

  # -------------------------------------------------------------------------
  # STEP 8: Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, .roworder)
  dt[, .roworder := NULL]

  return(as.data.frame(dt))
}


# =============================================================================
# USAGE
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key list entries via `lapply` + `paste` | One-time 344K × 4 integer matrix; row indices derived arithmetically |
| **Stat computation** | 32.3M R-level anonymous function calls (6.46M × 5 vars) | 5 × 4 vectorized column gathers + `matrixStats::rowMaxs/rowMins/rowMeans2` (C-level) |
| **Memory** | ~6.46M-element list of integer vectors + temporaries | One 6.46M × 4 `nb_val_mat` (~207 MB) reused per variable |
| **Estimated time** | 86+ hours | **~1–5 minutes** |
| **Numerical result** | max, min, mean of rook-neighbor values | Identical — same `max`, `min`, `mean` over same neighbor sets |
| **Trained RF model** | — | Untouched; no retraining |

The key insight is that **the spatial neighbor topology is invariant across years**, so it should be encoded once as an integer matrix at the cell level (~344K rows), and then the year dimension is handled via arithmetic index computation — `(cell_pos - 1) * n_years + year_pos` — eliminating all string operations and per-row R function calls. The `matrixStats` package provides C-implemented row-wise summary functions that process the full 6.46M-row matrix in seconds.