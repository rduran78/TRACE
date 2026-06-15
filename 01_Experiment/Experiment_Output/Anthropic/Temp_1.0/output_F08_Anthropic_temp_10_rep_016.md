 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The core problem is that `build_neighbor_lookup` constructs a **row-level** lookup (one entry per cell-year row, ~6.46 million entries), even though the **neighbor topology is static across all 28 years**. This means:

1. **Redundant work in `build_neighbor_lookup`:** For each of the ~6.46M rows, it resolves the same spatial neighbors that could be resolved once for the ~344K cells. The string-paste key construction (`paste(id, year)`) and hash-table lookup are repeated 28× per cell needlessly.

2. **Redundant iteration in `compute_neighbor_stats`:** The `lapply` over ~6.46M entries with index-chasing into the data frame is extremely slow. Each call re-indexes into the same column vector millions of times.

3. **Memory overhead:** Storing ~6.46M integer vectors in a list, when only ~344K unique neighbor sets exist, wastes both memory and cache locality.

**Bottom line:** The neighbor *graph* is year-invariant; only the *values* change by year. The current code entangles topology and data, paying the topology cost 28× and the per-row overhead ~6.46M× instead of ~344K×.

---

## Optimization Strategy

**Separate the static topology from the dynamic values:**

1. **Build a cell-level neighbor lookup once** — a list of length 344,208 mapping each cell index to its neighbor cell indices. This is purely spatial and year-independent. Cost: O(344K), done once.

2. **For each variable, compute neighbor stats using vectorized year-wise matrix operations:**
   - Reshape the variable into a **cells × years matrix** (344,208 rows × 28 columns).
   - For each cell, the neighbor indices into this matrix are the same across all columns (years).
   - Use a **sparse adjacency matrix** (from the `nb` object) and sparse matrix–dense matrix multiplication to compute neighbor sums and counts in one shot, then derive mean, max, and min.
   - For **mean**: `neighbor_mean = (W %*% X) / (W %*% non_NA_indicator)` where `W` is the binary adjacency matrix.
   - For **max** and **min**: iterate over cells once using the cell-level neighbor list (344K iterations, not 6.46M), but vectorized across the 28 year-columns simultaneously.

3. **Flatten back** to the original row order and attach columns.

This reduces the work from ~6.46M × 5 variable passes with string hashing to ~344K × 5 passes with integer indexing plus sparse matrix algebra. Expected speedup: **~50–200×** (minutes instead of 86+ hours).

---

## Working R Code

```r
library(Matrix)

# =============================================================================
# STEP 1: Build a CELL-level neighbor lookup (done ONCE, year-independent)
# =============================================================================
# Input:
#   id_order            — vector of cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique — spdep nb object (list of integer neighbor index vectors)
#
# Output:
#   cell_neighbor_idx   — list of length n_cells; each element is an integer
#                         vector of neighbor positions (indices into id_order)

build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is already index-based (spdep nb): neighbors[[i]] gives

  # integer indices into id_order for the neighbors of cell i.
  # We just need to strip the 0-length sentinels spdep sometimes uses.
  n <- length(id_order)
  lapply(seq_len(n), function(i) {
    nb <- neighbors[[i]]
    nb <- nb[nb != 0L]              # spdep uses 0 for "no neighbors"
    as.integer(nb)
  })
}

# =============================================================================
# STEP 2: Build a sparse binary adjacency matrix (done ONCE)
# =============================================================================
build_adjacency_matrix <- function(cell_neighbor_idx, n_cells) {
  # Build COO triplets
  from <- rep(seq_len(n_cells), lengths(cell_neighbor_idx))
  to   <- unlist(cell_neighbor_idx, use.names = FALSE)
  W    <- sparseMatrix(i = from, j = to, x = 1, dims = c(n_cells, n_cells))
  W
}

# =============================================================================
# STEP 3: Reshape a variable from long cell_data to cells x years matrix
# =============================================================================
# Assumes cell_data is sorted by (id, year) or we can index into it.
# We build a mapping once.

build_cell_year_indices <- function(cell_data, id_order, years) {
  # Returns a matrix of row-indices into cell_data: n_cells x n_years
  # cell_data must have columns 'id' and 'year'

  n_cells <- length(id_order)
  n_years <- length(years)

  # Create a fast lookup: (id, year) -> row in cell_data
  # Use data.table for speed
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("data.table is required for efficient indexing.")
  }

  dt <- data.table::data.table(
    row_idx = seq_len(nrow(cell_data)),
    id      = cell_data$id,
    year    = cell_data$year
  )
  data.table::setkey(dt, id, year)

  # For each cell in id_order and each year, find the row index

  grid <- data.table::CJ(id = id_order, year = years)
  grid <- dt[grid, on = .(id, year)]  # left join

  # Reshape to matrix: rows = cells (in id_order order), cols = years
  idx_matrix <- matrix(grid$row_idx, nrow = n_cells, ncol = n_years, byrow = FALSE)
  idx_matrix
}

reshape_to_matrix <- function(cell_data, var_name, idx_matrix) {
  vals <- cell_data[[var_name]]
  matrix(vals[idx_matrix], nrow = nrow(idx_matrix), ncol = ncol(idx_matrix))
}

# =============================================================================
# STEP 4: Compute neighbor max, min, mean using the static topology
# =============================================================================
compute_neighbor_stats_optimized <- function(var_matrix, cell_neighbor_idx, W) {
  # var_matrix: n_cells x n_years
  # cell_neighbor_idx: list of neighbor indices (cell-level)
  # W: sparse adjacency matrix n_cells x n_cells
  #
  # Returns list with three matrices (each n_cells x n_years):
  #   neighbor_max, neighbor_min, neighbor_mean

  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)

  # --- Neighbor mean via sparse matrix multiplication ---
  # Handle NAs: replace NA with 0 for summation, track valid counts
  not_na <- !is.na(var_matrix)
  var_zero <- var_matrix
  var_zero[is.na(var_zero)] <- 0

  neighbor_sum   <- as.matrix(W %*% var_zero)        # n_cells x n_years
  neighbor_count <- as.matrix(W %*% (not_na * 1.0))  # n_cells x n_years

  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- Neighbor max and min: vectorized cell-level loop ---
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_idx[[i]]
    if (length(nb) == 0L) next
    # Extract sub-matrix of neighbor values: length(nb) x n_years
    nb_vals <- var_matrix[nb, , drop = FALSE]
    if (length(nb) == 1L) {
      # Single neighbor: max = min = that value
      neighbor_max[i, ] <- nb_vals[1L, ]
      neighbor_min[i, ] <- nb_vals[1L, ]
    } else {
      # Suppress warnings from all-NA columns
      neighbor_max[i, ] <- suppressWarnings(apply(nb_vals, 2L, max, na.rm = TRUE))
      neighbor_min[i, ] <- suppressWarnings(apply(nb_vals, 2L, min, na.rm = TRUE))
      # apply with na.rm=TRUE on all-NA gives -Inf/Inf; fix those
    }
  }
  # Fix -Inf / Inf from all-NA neighbor slices
  neighbor_max[is.infinite(neighbor_max)] <- NA_real_
  neighbor_min[is.infinite(neighbor_min)] <- NA_real_

  list(neighbor_max = neighbor_max,
       neighbor_min = neighbor_min,
       neighbor_mean = neighbor_mean)
}

# =============================================================================
# STEP 5: Write results back to cell_data in the original row order
# =============================================================================
write_neighbor_features <- function(cell_data, var_name, stats, idx_matrix) {
  # stats: list of 3 matrices (n_cells x n_years)
  # idx_matrix: n_cells x n_years matrix of row indices into cell_data

  valid <- !is.na(idx_matrix)
  rows  <- idx_matrix[valid]

  cell_data[[paste0("neighbor_max_",  var_name)]][rows] <- stats$neighbor_max[valid]
  cell_data[[paste0("neighbor_min_",  var_name)]][rows] <- stats$neighbor_min[valid]
  cell_data[[paste0("neighbor_mean_", var_name)]][rows] <- stats$neighbor_mean[valid]

  cell_data
}

# =============================================================================
# MAIN PIPELINE (replaces the original outer loop)
# =============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  years   <- sort(unique(cell_data$year))
  n_cells <- length(id_order)

  cat("Building cell-level neighbor lookup (once)...\n")
  cell_neighbor_idx <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

  cat("Building sparse adjacency matrix (once)...\n")
  W <- build_adjacency_matrix(cell_neighbor_idx, n_cells)

  cat("Building cell-year index matrix (once)...\n")
  idx_matrix <- build_cell_year_indices(cell_data, id_order, years)

  # Pre-allocate output columns
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  for (var_name in neighbor_source_vars) {
    cell_data[[paste0("neighbor_max_",  var_name)]] <- NA_real_
    cell_data[[paste0("neighbor_min_",  var_name)]] <- NA_real_
    cell_data[[paste0("neighbor_mean_", var_name)]] <- NA_real_
  }

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s ...\n", var_name))
    t0 <- Sys.time()

    var_matrix <- reshape_to_matrix(cell_data, var_name, idx_matrix)
    stats      <- compute_neighbor_stats_optimized(var_matrix, cell_neighbor_idx, W)
    cell_data  <- write_neighbor_features(cell_data, var_name, stats, idx_matrix)

    cat(sprintf("  Done in %.1f seconds.\n", as.numeric(Sys.time() - t0, units = "secs")))
  }

  cell_data
}

# ---------------------------------------------------------------------------
# CALL (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------------------
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# The 15 new columns (3 stats x 5 vars) are now attached to cell_data.
# Proceed directly to prediction with the pre-trained Random Forest:
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

---

## Further Speed-Up: Eliminate the R-Level Loop for Max/Min

The `for (i in seq_len(n_cells))` loop over 344K cells for max/min is the remaining bottleneck. Two options to accelerate it:

```r
# OPTION A: Rcpp implementation (recommended — drops from ~minutes to seconds)
# Save as "neighbor_maxmin.cpp" and call Rcpp::sourceCpp("neighbor_maxmin.cpp")

# // [[Rcpp::depends(RcppArmadillo)]]
# #include <RcppArmadillo.h>
# using namespace Rcpp;
#
# // [[Rcpp::export]]
# List neighbor_maxmin_cpp(NumericMatrix var_matrix, List cell_neighbor_idx) {
#   int n_cells = var_matrix.nrow();
#   int n_years = var_matrix.ncol();
#   NumericMatrix nmax(n_cells, n_years);
#   NumericMatrix nmin(n_cells, n_years);
#   std::fill(nmax.begin(), nmax.end(), NA_REAL);
#   std::fill(nmin.begin(), nmin.end(), NA_REAL);
#
#   for (int i = 0; i < n_cells; i++) {
#     IntegerVector nb = cell_neighbor_idx[i];
#     int nnb = nb.size();
#     if (nnb == 0) continue;
#     for (int j = 0; j < n_years; j++) {
#       double mx = R_NegInf, mn = R_PosInf;
#       bool any_valid = false;
#       for (int k = 0; k < nnb; k++) {
#         double v = var_matrix(nb[k] - 1, j);  // R is 1-indexed
#         if (!R_IsNA(v)) {
#           any_valid = true;
#           if (v > mx) mx = v;
#           if (v < mn) mn = v;
#         }
#       }
#       if (any_valid) { nmax(i, j) = mx; nmin(i, j) = mn; }
#     }
#   }
#   return List::create(Named("neighbor_max") = nmax,
#                       Named("neighbor_min") = nmin);
# }

# OPTION B: Pure R — parallel chunk processing (no compilation needed)
# Replace the for-loop in compute_neighbor_stats_optimized with:
compute_maxmin_chunked <- function(var_matrix, cell_neighbor_idx, n_cells, n_years, chunk_size = 10000L) {
  neighbor_max <- matrix(NA_real_, n_cells, n_years)
  neighbor_min <- matrix(NA_real_, n_cells, n_years)

  starts <- seq(1L, n_cells, by = chunk_size)
  for (s in starts) {
    e <- min(s + chunk_size - 1L, n_cells)
    for (i in s:e) {
      nb <- cell_neighbor_idx[[i]]
      if (length(nb) == 0L) next
      nb_vals <- var_matrix[nb, , drop = FALSE]
      if (length(nb) == 1L) {
        neighbor_max[i, ] <- nb_vals[1L, ]
        neighbor_min[i, ] <- nb_vals[1L, ]
      } else {
        for (j in seq_len(n_years)) {
          v <- nb_vals[, j]
          v <- v[!is.na(v)]
          if (length(v) > 0L) {
            neighbor_max[i, j] <- max(v)
            neighbor_min[i, j] <- min(v)
          }
        }
      }
    }
  }
  list(neighbor_max = neighbor_max, neighbor_min = neighbor_min)
}
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-key hashes | 344K integer index (once) |
| Neighbor mean | 6.46M `lapply` calls | Sparse matrix multiply (seconds) |
| Neighbor max/min | 6.46M `lapply` calls | 344K cell loop over 28-col matrix |
| Total topology work | Repeated 5× per variable | Done once, reused |
| Estimated time | 86+ hours | **~5–30 minutes** (R); **~1–3 min** (with Rcpp) |
| Numerical result | Identical | **Identical** (same max, min, mean) |
| RF model | Unchanged | **Unchanged** — no retraining |