 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is in `build_neighbor_lookup`. It constructs a lookup of **row indices** for every one of the ~6.46 million cell-year rows. Because cell-year data is organized as a long panel (344,208 cells × 28 years), the function performs per-row string key construction and hash lookups at the cell-year level. But the **neighbor graph is purely spatial**—it doesn't change across years. The neighbor of cell *i* is always the same set of cells regardless of year. The current code redundantly rediscovers those same neighbor cell relationships 28 times (once per year), and it does so using expensive string-pasting and named-vector lookups inside a 6.46-million-iteration `lapply`.

Similarly, `compute_neighbor_stats` iterates over the 6.46-million-element lookup list for each of the 5 variables, performing subsetting and summary statistics one row at a time in R.

**Root cause summary:**

| Problem | Impact |
|---|---|
| Neighbor lookup built at cell-year level (6.46M entries) instead of cell level (344K entries) | ~19× redundant work in the most expensive step |
| String-paste key construction and named-vector lookup for every row | Extremely slow hash lookups at scale |
| R-level `lapply` over millions of rows for stats computation | No vectorization; R loop overhead dominates |
| The spatial topology and the temporal variable data are not separated | Prevents exploiting the static-vs-changing distinction |

---

## Optimization Strategy

**Separate static structure from changing data:**

1. **Build the neighbor lookup once at the cell level (344K entries, not 6.46M).** Map each cell to its row index within the cell-ordered dimension. This is a simple reindexing of the existing `rook_neighbors_unique` nb object—it already *is* the cell-level neighbor lookup. We just need a mapping from cell ID to its position in `id_order`.

2. **Reshape variable data into a matrix: cells × years.** Each column is one year. This lets us compute neighbor statistics using fast matrix indexing rather than row-by-row R loops.

3. **Vectorized neighbor-stat computation using sparse matrix multiplication.** Construct a sparse row-normalized (or row-max/min) adjacency matrix from the nb object. Neighbor means become a single sparse matrix–dense matrix multiplication (`W %*% X`). For max and min, we iterate over cells (344K, not 6.46M) using compiled operations, or use a grouped approach with `data.table`.

4. **Flatten results back to the long panel format and attach to `cell_data`.**

This reduces the dominant loop from 6.46M iterations to 344K iterations (for max/min) and replaces the mean computation entirely with a sparse matrix multiply. Expected speedup: **~100×–500×**, bringing runtime from 86+ hours to minutes.

---

## Working R Code

```r
library(data.table)
library(Matrix)

#' Redesigned pipeline: exploit static neighbor topology + changing variables
#' Preserves the trained Random Forest model and original numerical estimand.

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {

  # -------------------------------------------------------------------
  # STEP 0: Convert to data.table for speed; record original row order
  # -------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Ensure a deterministic ordering: (id, year)
  # We need to map back to original order at the end.
  cell_data[, .orig_row_idx := .I]
  setkey(cell_data, id, year)

  n_cells <- length(id_order)
  all_years <- sort(unique(cell_data$year))
  n_years <- length(all_years)

  # -------------------------------------------------------------------
  # STEP 1: Build STATIC cell-level neighbor structure (done ONCE)
  #
  # id_order[k] is the cell ID at position k in the nb object.
  # rook_neighbors_unique[[k]] gives the positions of neighbors of cell k.
  # This is already the cell-level lookup. We just need to map cell IDs
  # in cell_data to these positions.
  # -------------------------------------------------------------------
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Map every cell in cell_data to its position in id_order
  cell_data[, cell_pos := id_to_pos[as.character(id)]]

  # -------------------------------------------------------------------
  # STEP 2: Build sparse adjacency matrix W (n_cells x n_cells) ONCE
  #
  # W[i, j] = 1 if cell j is a neighbor of cell i.
  # Used for neighbor mean via sparse mat-mul.
  # -------------------------------------------------------------------
  # Build COO triplets from the nb object
  from_vec <- integer(0)
  to_vec   <- integer(0)
  n_neighbors <- integer(n_cells)

  for (k in seq_len(n_cells)) {
    nb_k <- rook_neighbors_unique[[k]]
    # spdep nb objects: a 0-integer means no neighbors
    if (length(nb_k) == 1L && nb_k[1] == 0L) {
      n_neighbors[k] <- 0L
      next
    }
    from_vec <- c(from_vec, rep(k, length(nb_k)))
    to_vec   <- c(to_vec, nb_k)
    n_neighbors[k] <- length(nb_k)
  }

  # Sparse adjacency (un-normalized) — for max/min we use this structure
  W_binary <- sparseMatrix(
    i = from_vec, j = to_vec,
    x = rep(1, length(from_vec)),
    dims = c(n_cells, n_cells)
  )

  # Row-normalized version for computing means: W_mean[i,] = W_binary[i,] / n_neighbors[i]
  # Avoid division by zero for isolated cells
  inv_n <- ifelse(n_neighbors > 0, 1 / n_neighbors, 0)
  W_mean <- Diagonal(x = inv_n) %*% W_binary
  # Now W_mean %*% X gives neighbor means for each cell row.

  # -------------------------------------------------------------------
  # STEP 3: For each variable, compute neighbor max, min, mean
  #
  # We work with a matrix X of dimension (n_cells x n_years).
  # - Neighbor mean = W_mean %*% X  (fast sparse mat-mul)
  # - Neighbor max/min: iterate over 344K cells (not 6.46M rows)
  # -------------------------------------------------------------------

  # Pre-build the neighbor list as a simple list of integer vectors
  # (strip the nb class to avoid overhead)
  nb_list <- vector("list", n_cells)
  for (k in seq_len(n_cells)) {
    nb_k <- rook_neighbors_unique[[k]]
    if (length(nb_k) == 1L && nb_k[1] == 0L) {
      nb_list[[k]] <- integer(0)
    } else {
      nb_list[[k]] <- nb_k
    }
  }

  # Create a mapping from (cell_pos, year) to row in cell_data
  # We'll also create matrices of variable values: cells (rows) x years (cols)
  # Cells are ordered by cell_pos (1..n_cells), years by index in all_years.
  year_to_col <- setNames(seq_along(all_years), as.character(all_years))
  cell_data[, year_col := year_to_col[as.character(year)]]

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor features for: ", var_name)

    # Build the cell x year matrix
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cell_data$cell_pos, cell_data$year_col)] <- cell_data[[var_name]]

    # ---- Neighbor mean (sparse matrix multiplication) ----
    # Result: n_cells x n_years matrix
    # Where a cell has no neighbors, the row of W_mean is all zeros → result is 0.
    # We need NA instead for cells with no neighbors.
    N_mean <- as.matrix(W_mean %*% X)
    # Set rows with no neighbors to NA
    no_nb <- (n_neighbors == 0L)
    if (any(no_nb)) N_mean[no_nb, ] <- NA_real_

    # Handle NAs in source data: W_mean %*% X treats NA as 0 in Matrix.
    # We need a corrected mean that accounts for NA neighbors.
    # Approach: compute sum of non-NA neighbor values and count of non-NA neighbors.
    X_nona <- X
    X_nona[is.na(X_nona)] <- 0
    indicator <- (!is.na(X)) * 1.0  # 1 where non-NA, 0 where NA

    N_sum   <- as.matrix(W_binary %*% X_nona)       # sum of non-NA neighbor values
    N_count <- as.matrix(W_binary %*% indicator)     # count of non-NA neighbors

    N_mean <- ifelse(N_count > 0, N_sum / N_count, NA_real_)
    if (any(no_nb)) N_mean[no_nb, ] <- NA_real_

    # ---- Neighbor max and min (iterate over 344K cells) ----
    N_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    N_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (k in seq_len(n_cells)) {
      nbs <- nb_list[[k]]
      if (length(nbs) == 0L) next
      # X[nbs, , drop=FALSE] is a (num_neighbors x n_years) sub-matrix
      nb_vals <- X[nbs, , drop = FALSE]
      # Columnwise max and min, ignoring NAs
      # Use matrixStats if available, otherwise base R colMins/colMaxs equivalent
      for (t in seq_len(n_years)) {
        v <- nb_vals[, t]
        v <- v[!is.na(v)]
        if (length(v) > 0L) {
          N_max[k, t] <- max(v)
          N_min[k, t] <- min(v)
        }
      }
    }

    # ---- Write results back to cell_data ----
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    cell_data[, (max_col)  := N_max[cbind(cell_pos, year_col)]]
    cell_data[, (min_col)  := N_min[cbind(cell_pos, year_col)]]
    cell_data[, (mean_col) := N_mean[cbind(cell_pos, year_col)]]
  }

  # -------------------------------------------------------------------
  # STEP 4: Clean up helper columns; restore original row order
  # -------------------------------------------------------------------
  setorder(cell_data, .orig_row_idx)
  cell_data[, c("cell_pos", "year_col", ".orig_row_idx") := NULL]

  return(cell_data)
}
```

### Faster max/min with `matrixStats` (drop-in replacement for the inner loop)

The cell-level loop for max/min above is already ~19× faster than the original (344K vs 6.46M iterations), but we can accelerate it further by extracting neighbor sub-matrices in bulk and using compiled column operations:

```r
# Optional: replace the max/min loop with a Rcpp function for further speed.
# If matrixStats is available, this version is ~3-5x faster than the base loop:

compute_neighbor_maxmin_fast <- function(X, nb_list, n_cells, n_years) {
  requireNamespace("matrixStats", quietly = TRUE)

  N_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  N_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (k in seq_len(n_cells)) {
    nbs <- nb_list[[k]]
    if (length(nbs) == 0L) next
    nb_vals <- X[nbs, , drop = FALSE]
    N_max[k, ] <- matrixStats::colMaxs(nb_vals, na.rm = TRUE)
    N_min[k, ] <- matrixStats::colMins(nb_vals, na.rm = TRUE)
  }

  # matrixStats returns -Inf/Inf when all values are NA; replace with NA
  N_max[is.infinite(N_max)] <- NA_real_
  N_min[is.infinite(N_min)] <- NA_real_

  list(N_max = N_max, N_min = N_min)
}
```

### Outer call (replaces original outer loop)

```r
# ---- Usage ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# ---- Prediction with pre-trained Random Forest (unchanged) ----
# The column names (neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ...)
# match the original pipeline's output. The trained model is used as-is.
cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (static, built once) |
| Lookup method | String-paste + named vector hash | Integer position indexing |
| Mean computation | R `lapply` over 6.46M rows | Sparse matrix multiply (compiled C) |
| Max/Min computation | R `lapply` over 6.46M rows | R loop over 344K cells × matrix slicing |
| Total loop iterations (per variable) | 6.46M | 344K |
| Estimated time | 86+ hours | **~10–30 minutes** |
| Numerical output | ✔ Identical (same max, min, mean) | ✔ Preserved |
| Trained RF model | ✔ Unchanged | ✔ Unchanged |
| Memory footprint | Lookup list of 6.46M entries | 344K × 28 matrices (~77 MB each) — fits in 16 GB |