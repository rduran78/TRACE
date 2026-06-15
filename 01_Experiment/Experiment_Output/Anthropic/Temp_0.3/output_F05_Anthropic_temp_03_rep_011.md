 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current year to form string keys — this is O(k) string allocations per row, where k ≈ 4 (rook neighbors).
4. **Looks up** those keys in `idx_lookup`, a named character vector of length 6.46M — named vector lookup is hash-based but still involves repeated hashing of freshly-allocated strings.

Across 6.46M rows × ~4 neighbors = ~25.8 million `paste` + hash-lookup operations, all inside an R-level `lapply`. This is the dominant bottleneck in `build_neighbor_lookup`.

Then, `compute_neighbor_stats` is called **5 times** (once per variable), each time iterating over 6.46M entries in `neighbor_lookup`. That's 5 × 6.46M = 32.3M R-level function calls, each extracting a small vector, subsetting, and computing three summary statistics.

### Summary of Inefficiencies

| Layer | Problem | Impact |
|-------|---------|--------|
| `build_neighbor_lookup` | Row-by-row `lapply` with `paste`/hash lookup in R | ~25.8M string ops in interpreted R |
| `build_neighbor_lookup` | String-keyed lookup instead of integer arithmetic | Hash overhead vs. O(1) arithmetic |
| `compute_neighbor_stats` | Called 5× separately, each with 6.46M R-level iterations | 32.3M R function calls |
| `compute_neighbor_stats` | Returns list-of-vectors, then `do.call(rbind, ...)` on 6.46M elements | Extremely slow `rbind` |
| Overall architecture | Neighbor lookup is row-centric (long format) when it could be cell-centric then broadcast across years | 28× redundant neighbor resolution |

## Optimization Strategy

### Key Insight: Separate the Spatial Topology from the Temporal Panel

Neighbor relationships are **purely spatial** — they don't change across years. The current code redundantly resolves the same cell→neighbor mapping 28 times (once per year per cell). Instead:

1. **Build a cell-level neighbor matrix once** (344,208 cells × max_neighbors) using integer indexing — no strings at all.
2. **Restructure data** so that for each cell, we know which rows correspond to which years — use integer arithmetic: if data is sorted by (id, year), then `row = (cell_index - 1) * 28 + (year - 1991)`.
3. **Vectorize** the neighbor statistics computation using matrix operations — extract neighbor values for all cells at once via matrix indexing, then compute `max`/`min`/`mean` with `rowMaxs`/`rowMins`/`rowMeans` from the `matrixStats` package.

This eliminates all string operations, all row-level `lapply` calls, and reduces the problem to bulk integer-indexed matrix subsetting.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites: data.table, matrixStats
# install.packages(c("data.table", "matrixStats"))  # if needed

library(data.table)
library(matrixStats)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # Step 0: Convert to data.table for fast manipulation; preserve row order
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .orig_row := .I]  # preserve original row order for final reassembly

  # -------------------------------------------------------------------------
  # Step 1: Sort by (id, year) so we can use integer arithmetic for row lookup
  # -------------------------------------------------------------------------
  # Create a dense cell index: map each unique id to 1..N_cells
  unique_ids <- as.integer(id_order)  # id_order defines the canonical ordering
  n_cells    <- length(unique_ids)
  
  # Map from cell id -> dense index (1-based, aligned with id_order)
  id_to_cellidx <- integer(max(unique_ids))
  id_to_cellidx[unique_ids] <- seq_len(n_cells)
  # If IDs are not contiguous integers, use a hash:
  # But let's be safe and use a named integer vector for arbitrary IDs
  id_to_cellidx_safe <- setNames(seq_len(n_cells), as.character(unique_ids))

  # Add cell index to dt
  dt[, cell_idx := id_to_cellidx_safe[as.character(id)]]

  # Determine years
  years     <- sort(unique(dt$year))
  n_years   <- length(years)
  year_to_yidx <- setNames(seq_len(n_years), as.character(years))
  dt[, year_idx := year_to_yidx[as.character(year)]]

  # Sort by (cell_idx, year_idx) for contiguous memory access
  setorder(dt, cell_idx, year_idx)

  # Now row for (cell_idx=c, year_idx=y) is at position: (c-1)*n_years + y
  # Verify this mapping is correct (all cells must have all years)
  expected_nrow <- n_cells * n_years
  if (nrow(dt) != expected_nrow) {
    # Panel is unbalanced — need explicit row lookup
    message("Panel is unbalanced (", nrow(dt), " vs expected ", expected_nrow,
            "). Building explicit row index.")
    # Build a matrix: row_index_mat[cell_idx, year_idx] = row in dt (or NA)
    row_index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
    row_index_mat[cbind(dt$cell_idx, dt$year_idx)] <- seq_len(nrow(dt))
    balanced <- FALSE
  } else {
    row_index_mat <- NULL
    balanced <- TRUE
    message("Panel is balanced: ", n_cells, " cells x ", n_years, " years = ",
            expected_nrow, " rows.")
  }

  # -------------------------------------------------------------------------
  # Step 2: Build cell-level neighbor index matrix (no strings!)
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length n_cells
  # where element i contains integer indices (into id_order) of neighbors of cell i.
  # We convert this to a padded integer matrix for vectorized access.

  max_neighbors <- max(lengths(rook_neighbors_unique))
  message("Max rook neighbors per cell: ", max_neighbors)

  # Build neighbor matrix: n_cells x max_neighbors
  # Entry (i, j) = cell_idx of j-th neighbor of cell i, or NA if fewer neighbors
  neighbor_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_neighbors)
  n_neighbors  <- integer(n_cells)

  for (ci in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[ci]]
    # spdep nb objects use 0 to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    k <- length(nb_i)
    n_neighbors[ci] <- k
    if (k > 0L) {
      neighbor_mat[ci, seq_len(k)] <- as.integer(nb_i)
    }
  }

  # -------------------------------------------------------------------------
  # Step 3: For each variable, compute neighbor max/min/mean vectorized
  # -------------------------------------------------------------------------
  # Strategy: for each year, extract the variable as a vector indexed by cell_idx,
  # then use neighbor_mat to gather neighbor values into a matrix, then apply
  # rowMaxs/rowMins/rowMeans.

  get_row <- function(ci, yi) {
    if (balanced) {
      return((ci - 1L) * n_years + yi)
    } else {
      return(row_index_mat[cbind(ci, yi)])
    }
  }

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    col_max  <- paste0("max_", var_name)
    col_min  <- paste0("min_", var_name)
    col_mean <- paste0("mean_", var_name)

    # Pre-allocate result columns
    res_max  <- rep(NA_real_, nrow(dt))
    res_min  <- rep(NA_real_, nrow(dt))
    res_mean <- rep(NA_real_, nrow(dt))

    vals_all <- dt[[var_name]]

    # Process year by year to keep memory bounded
    for (yi in seq_len(n_years)) {
      # Row indices in dt for all cells in this year
      if (balanced) {
        rows_this_year <- seq(from = yi, by = n_years, length.out = n_cells)
        # Actually with sort by (cell_idx, year_idx):
        # cell c, year y -> row (c-1)*n_years + y
        rows_this_year <- (seq_len(n_cells) - 1L) * n_years + yi
      } else {
        rows_this_year <- row_index_mat[, yi]
      }

      # Extract values for this year, indexed by cell_idx
      # vals_by_cell[ci] = value of var_name for cell ci in year yi
      vals_by_cell <- rep(NA_real_, n_cells)
      valid_rows <- !is.na(rows_this_year)
      vals_by_cell[valid_rows] <- vals_all[rows_this_year[valid_rows]]

      # Gather neighbor values: n_cells x max_neighbors matrix
      # neighbor_vals_mat[ci, j] = vals_by_cell[ neighbor_mat[ci, j] ]
      neighbor_vals_mat <- matrix(vals_by_cell[neighbor_mat],
                                  nrow = n_cells, ncol = max_neighbors)
      # Cells with fewer than max_neighbors already have NA in neighbor_mat,
      # which propagates to NA in neighbor_vals_mat — exactly what we want.

      # Also mask out cells that have 0 neighbors entirely
      has_neighbors <- n_neighbors > 0L

      # Compute stats using matrixStats (handles NA correctly)
      yr_max  <- rep(NA_real_, n_cells)
      yr_min  <- rep(NA_real_, n_cells)
      yr_mean <- rep(NA_real_, n_cells)

      if (any(has_neighbors)) {
        sub_mat <- neighbor_vals_mat[has_neighbors, , drop = FALSE]
        yr_max[has_neighbors]  <- rowMaxs(sub_mat,  na.rm = TRUE)
        yr_min[has_neighbors]  <- rowMins(sub_mat,  na.rm = TRUE)
        yr_mean[has_neighbors] <- rowMeans(sub_mat, na.rm = TRUE)
      }

      # Handle all-NA rows (rowMaxs returns -Inf, rowMins returns Inf)
      yr_max[is.infinite(yr_max)]  <- NA_real_
      yr_min[is.infinite(yr_min)]  <- NA_real_
      yr_mean[is.nan(yr_mean)]     <- NA_real_

      # Write results back to the correct rows in dt
      target_rows <- rows_this_year[valid_rows]
      res_max[target_rows]  <- yr_max[valid_rows]
      res_min[target_rows]  <- yr_min[valid_rows]
      res_mean[target_rows] <- yr_mean[valid_rows]
    }

    # Assign to dt
    set(dt, j = col_max,  value = res_max)
    set(dt, j = col_min,  value = res_min)
    set(dt, j = col_mean, value = res_mean)
  }

  # -------------------------------------------------------------------------
  # Step 4: Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, .orig_row)
  dt[, c("cell_idx", "year_idx", ".orig_row") := NULL]

  return(as.data.frame(dt))
}

# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_optimized(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The trained Random Forest model is untouched — just use cell_data for
# # predict(rf_model, newdata = cell_data) as before.
```

## Complexity Comparison

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Neighbor resolution** | 6.46M × `paste` + hash lookup | 344K × integer copy (once) |
| **String allocations** | ~25.8M `paste` calls | **Zero** |
| **R-level loop iterations** (neighbor lookup) | 6.46M `lapply` iterations | 0 (matrix indexing) |
| **R-level loop iterations** (stats) | 5 × 6.46M = 32.3M | 5 × 28 = **140** (year loop) |
| **Stats computation** | Scalar R per row | Vectorized `rowMaxs`/`rowMins`/`rowMeans` (C-level) |
| **`do.call(rbind, ...)` on 6.46M elements** | Yes (extremely slow) | **Eliminated** |
| **Estimated runtime** | 86+ hours | **~2–5 minutes** |
| **Peak memory** | ~same (string keys are large) | ~similar (matrix of neighbor values: 344K × 4 × 8 bytes ≈ 11 MB per year-slice) |
| **Numerical output** | max/min/mean of non-NA neighbor values | **Identical** — same estimand preserved |

## Why This Preserves the Trained Random Forest

The optimized code computes **exactly the same 15 columns** (3 stats × 5 variables) with **identical numerical values** — it's a pure algorithmic refactoring of the feature construction, not a change to the features themselves. The RF model object is never touched and can be used with `predict()` on the resulting `cell_data` exactly as before.