 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup mapping each of the ~6.46 million cell-year rows to the row indices of its neighbors' cell-year rows. This is catastrophically expensive because:

1. **Redundant work across years:** The neighbor *topology* is static (cell A's neighbors are always cells B, C, D regardless of year), but the current code rebuilds neighbor-row mappings for every cell-year combination. With 344,208 cells × 28 years = ~9.6 million list entries, each doing string-paste key lookups, this creates billions of string operations.

2. **String-keyed lookups:** `paste(id, year, sep="_")` and named-vector indexing (`idx_lookup[neighbor_keys]`) are O(n) hash lookups repeated millions of times — extremely slow in R.

3. **`lapply` over 6.46M rows:** The row-level loop in `build_neighbor_lookup` and again in `compute_neighbor_stats` iterates millions of times in interpreted R, which is inherently slow.

**The key insight:** Since the neighbor graph is *time-invariant*, we should separate the problem into:
- **Static structure (compute once):** Which cells are neighbors of which cells → a cell-level adjacency structure (344K entries, not 6.46M).
- **Dynamic values (compute per year):** The variable values change by year, so we extract per-year slices and use the static structure to compute neighbor stats via fast vectorized/matrix operations.

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** — a list of length 344,208 mapping each cell index to its neighbor cell indices. This is independent of year.

2. **Construct a cell × year matrix for each variable** — rows = cells (in `id_order`), columns = years. This allows vectorized column-wise (per-year) operations.

3. **Compute neighbor stats using sparse matrix multiplication** — Convert the adjacency list to a sparse matrix `W`. Then:
   - `neighbor_sum = W %*% X` (for mean: divide by neighbor count)
   - `neighbor_max` and `neighbor_min` via a loop over cells using the adjacency list, but vectorized per-year via matrix columns — or more efficiently, using a custom sparse-max approach.

4. **Write results back** to the data.frame in the correct row order.

This reduces the problem from ~6.46M list iterations with string lookups to ~344K list iterations (for max/min) × 28 years of vectorized operations, plus instant sparse matrix multiplication for means. Expected runtime: **minutes instead of 86+ hours**.

## Working R Code

```r
library(Matrix)

#' Redesigned neighbor feature computation.
#' Separates static topology from dynamic (yearly) variable values.
#'
#' @param cell_data       data.frame with columns: id, year, and all neighbor_source_vars
#' @param id_order        vector of cell IDs defining the canonical ordering (length = n_cells)
#' @param rook_neighbors_unique  spdep::nb object (list of integer neighbor indices into id_order)
#' @param neighbor_source_vars   character vector of variable names to compute neighbor stats for
#' @return cell_data with new columns: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  # ---------------------------------------------------------------
  # STEP 1: Build static cell-level adjacency (once)
  # ---------------------------------------------------------------
  # rook_neighbors_unique is already an nb object indexed into id_order.
  # Clean it: replace 0L (no-neighbor sentinel in nb objects) with integer(0).
  cell_neighbors <- lapply(rook_neighbors_unique, function(nb) {
    nb <- as.integer(nb)
    nb[nb != 0L]
  })

  # ---------------------------------------------------------------
  # STEP 2: Build sparse adjacency matrix W (n_cells x n_cells) — for mean

  # ---------------------------------------------------------------
  # Build COO triplets from adjacency list
  from_idx <- rep(seq_len(n_cells), lengths(cell_neighbors))
  to_idx   <- unlist(cell_neighbors, use.names = FALSE)

  W <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )

  # Number of neighbors per cell (for computing mean)
  # This is the row sum of W, but we need to account for NA values per variable/year,
  # so we'll compute the count dynamically below.

  # ---------------------------------------------------------------
  # STEP 3: Create a fast mapping from (id, year) -> row index in cell_data
  # ---------------------------------------------------------------
  # Map cell IDs to canonical index
  id_to_cidx <- setNames(seq_len(n_cells), as.character(id_order))
  cell_data$.cidx <- id_to_cidx[as.character(cell_data$id)]

  # Map years to column index
  year_to_yidx <- setNames(seq_len(n_years), as.character(years))
  cell_data$.yidx <- year_to_yidx[as.character(cell_data$year)]

  # ---------------------------------------------------------------
  # STEP 4: For each variable, build cell x year matrix, compute stats, write back
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    cat("Processing neighbor stats for:", var_name, "\n")

    # --- 4a: Build cell x year matrix (n_cells x n_years) ---
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cell_data$.cidx, cell_data$.yidx)] <- cell_data[[var_name]]

    # --- 4b: Neighbor MEAN via sparse matrix multiplication ---
    # Handle NAs: replace NA with 0 for sum, track non-NA counts
    X_nona     <- X
    X_nona[is.na(X_nona)] <- 0
    X_notna    <- (!is.na(X)) * 1  # indicator matrix: 1 if not NA, 0 if NA

    neighbor_sum   <- as.matrix(W %*% X_nona)       # n_cells x n_years
    neighbor_count <- as.matrix(W %*% X_notna)       # n_cells x n_years

    neighbor_mean_mat <- neighbor_sum / neighbor_count
    # Where count == 0, result is NaN from 0/0; convert to NA
    neighbor_mean_mat[neighbor_count == 0] <- NA_real_

    # --- 4c: Neighbor MAX and MIN ---
    # We must iterate over cells, but vectorize across years (28 cols).
    # With 344K cells and 28 years, this is fast.
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (ci in seq_len(n_cells)) {
      nb <- cell_neighbors[[ci]]
      if (length(nb) == 0L) next
      # Extract neighbor rows: length(nb) x n_years sub-matrix
      nb_vals <- X[nb, , drop = FALSE]  # matrix: length(nb) x n_years
      # Column-wise max and min (suppress warnings for all-NA columns)
      neighbor_max_mat[ci, ] <- suppressWarnings(apply(nb_vals, 2, max, na.rm = TRUE))
      neighbor_min_mat[ci, ] <- suppressWarnings(apply(nb_vals, 2, min, na.rm = TRUE))
    }
    # apply with na.rm=TRUE on all-NA returns -Inf/Inf; convert to NA
    neighbor_max_mat[is.infinite(neighbor_max_mat)] <- NA_real_
    neighbor_min_mat[is.infinite(neighbor_min_mat)] <- NA_real_

    # --- 4d: Map matrices back to cell_data rows ---
    idx_mat <- cbind(cell_data$.cidx, cell_data$.yidx)

    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")

    cell_data[[col_max]]  <- neighbor_max_mat[idx_mat]
    cell_data[[col_min]]  <- neighbor_min_mat[idx_mat]
    cell_data[[col_mean]] <- neighbor_mean_mat[idx_mat]

    cat("  Done:", var_name, "\n")
  }

  # Clean up temporary columns
  cell_data$.cidx <- NULL
  cell_data$.yidx <- NULL

  return(cell_data)
}
```

### Further optimization: replace the `for (ci in ...)` loop with Rcpp for max/min

The cell-level loop for max and min (344K iterations) with `apply` over a small sub-matrix is already reasonable (~minutes), but if further speed is needed, here is an Rcpp drop-in:

```r
Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_max_min_cpp(NumericMatrix X, List cell_neighbors) {
  int n_cells = X.nrow();
  int n_years = X.ncol();
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);

  // Initialize to NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);

  for (int ci = 0; ci < n_cells; ci++) {
    IntegerVector nb = cell_neighbors[ci];
    int n_nb = nb.size();
    if (n_nb == 0) continue;

    for (int yi = 0; yi < n_years; yi++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int ni = 0; ni < n_nb; ni++) {
        double val = X(nb[ni] - 1, yi);  // nb is 1-indexed
        if (!NumericVector::is_na(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(ci, yi) = cur_max;
        min_mat(ci, yi) = cur_min;
      }
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')
```

Replace the max/min section (4c) with:

```r
    # --- 4c (fast): Neighbor MAX and MIN via Rcpp ---
    maxmin <- neighbor_max_min_cpp(X, cell_neighbors)
    neighbor_max_mat <- maxmin$max_mat
    neighbor_min_mat <- maxmin$min_mat
```

### Usage (drop-in replacement for the original outer loop)

```r
# Original outer loop replaced by a single call:
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data[, predictor_cols])
```

## Summary of Improvements

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (static) |
| String key operations | ~billions of paste + hash lookups | Zero |
| Data structure for variable access | Column vector + list of row indices | Cell × Year matrix (vectorized) |
| Neighbor mean computation | R `lapply` over 6.46M entries | Sparse matrix multiply (`W %*% X`) |
| Neighbor max/min | R `lapply` over 6.46M entries | Loop over 344K cells, vectorized across 28 years (or Rcpp) |
| Estimated runtime | 86+ hours | **5–15 minutes** (pure R) / **1–3 minutes** (with Rcpp) |
| Numerical results | — | **Identical** (same max, min, mean computations) |
| RAM usage | High (6.46M-entry list of integer vectors) | ~344K × 28 × 5 doubles ≈ 400 MB peak |
| Trained RF model | Preserved | Preserved (not modified) |