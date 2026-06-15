 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's spatial neighbors (from ~344K cells).
2. Constructs character keys by pasting `id_year`.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Character-based named-vector lookup in R is **O(n)** in the worst case per access due to hashing overhead at scale, and `paste()`-based key construction over 6.46M rows is extremely expensive. Because the panel is **balanced** (every cell appears in every year), the neighbor structure is **identical across years** — yet this function redundantly recomputes it for every cell-year combination.

### Bottleneck 2: `compute_neighbor_stats` — `lapply` over 6.46M rows with per-row subsetting

For each of 5 variables × 6.46M rows, the function:
1. Subsets a numeric vector by index.
2. Removes NAs.
3. Computes `max`, `min`, `mean`.

This is ~32.3 million R-level function calls with per-element list overhead. The `do.call(rbind, result)` on a 6.46M-element list is also very slow.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics on regular grids extremely efficiently using compiled C code with sliding windows. The analogy is apt: we are computing neighborhood summaries. However, the grid here is indexed by an irregular `spdep::nb` object (not all cells may have 4 neighbors — boundary cells, islands, masked cells), and the data is in long panel format. Directly reshaping to a raster stack for 28 years is possible but risks subtle mismatches with the `nb` object. **The better strategy is to exploit the same principle — vectorized matrix operations over a fixed spatial topology — while staying faithful to the `nb` structure.**

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The neighbor structure is **purely spatial** and **constant across all 28 years**. There are only ~344K cells and ~1.37M directed neighbor pairs. We should:

1. **Build a sparse adjacency matrix once** from the `nb` object (~344K × 344K, ~1.37M nonzero entries).
2. **Reshape each variable into a 344K × 28 matrix** (cells × years).
3. **Use sparse matrix multiplication and vectorized row operations** to compute neighbor stats in one shot per variable.

For `mean`: sparse matrix–dense matrix multiplication (`A %*% X`) divided by the row-count vector gives the neighbor mean directly — this is a single compiled BLAS/cholmod call.

For `max` and `min`: we iterate over the (small) neighbor list at the **cell level** (344K iterations, not 6.46M), and vectorize across years.

### Expected Speedup

| Component | Before | After |
|---|---|---|
| Neighbor lookup | ~6.46M character key ops | Eliminated (sparse matrix) |
| Mean computation | 6.46M R-level loops × 5 vars | 5 sparse mat-muls (~seconds) |
| Max/Min computation | 6.46M R-level loops × 5 vars | 344K loops × 28 vectorized cols × 5 vars |
| **Total estimated time** | **86+ hours** | **~2–10 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE ENGINEERING
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and the
#     neighbor_source_vars, sorted by (id, year) or at least containing all
#     combinations.
#   - rook_neighbors_unique: an nb object (list of integer vectors) of length
#     equal to the number of spatial cells.
#   - id_order: character or numeric vector mapping position in the nb object
#     to cell id.
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # --- Convert to data.table for speed ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  cat("Cells:", n_cells, "| Years:", n_years,
      "| Rows:", nrow(cell_data), "\n")

  # =========================================================================
  # STEP 1: Build sparse adjacency matrix from nb object (binary, row-stochastic
  #         version built separately for mean).
  # =========================================================================
  cat("Building sparse adjacency matrix...\n")

  # Construct COO (coordinate) triplets from the nb object
  from_idx <- integer(0)
  to_idx   <- integer(0)

  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    from_idx <- c(from_idx, rep(i, length(nb_i)))
    to_idx   <- c(to_idx, nb_i)
  }

  # Binary adjacency matrix (n_cells x n_cells)
  A <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = 1,
    dims = c(n_cells, n_cells)
  )

  # Number of neighbors per cell (for computing mean)
  n_neighbors <- as.numeric(A %*% rep(1, n_cells))  # rowSums of A

  cat("Adjacency matrix:", length(from_idx), "directed edges\n")

  # =========================================================================
  # STEP 2: Create a mapping from (id, year) -> row index in cell_data,
  #         and from id -> spatial index (position in id_order).
  # =========================================================================
  cat("Building index mappings...\n")

  # Map cell id -> spatial index (position in id_order / nb object)
  id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))

  # We need cell_data sorted by (spatial_index, year) to reshape into matrices.
  cell_data[, spatial_idx := id_to_spatial[as.character(id)]]
  cell_data[, year_idx    := match(year, years)]

  # Sort for consistent matrix filling
  setorder(cell_data, spatial_idx, year_idx)

  # Verify completeness (balanced panel)
  if (nrow(cell_data) != n_cells * n_years) {
    warning("Panel is not perfectly balanced. ",
            "Missing cell-years will be treated as NA.")
  }

  # =========================================================================
  # STEP 3: For each variable, reshape to matrix, compute stats, write back.
  # =========================================================================

  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "...\n")

    # --- Reshape variable to n_cells x n_years matrix ---
    # Initialize with NA
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cell_data$spatial_idx, cell_data$year_idx)] <- cell_data[[var_name]]

    # -----------------------------------------------------------------
    # MEAN: Use sparse matrix multiplication.
    #   neighbor_sum = A %*% X   (n_cells x n_years)
    #   neighbor_mean = neighbor_sum / n_neighbors
    # -----------------------------------------------------------------
    neighbor_sum  <- as.matrix(A %*% X)  # dense result
    neighbor_mean <- neighbor_sum / n_neighbors  # vectorized division
    # Cells with 0 neighbors get NaN from 0/0; convert to NA
    neighbor_mean[n_neighbors == 0, ] <- NA_real_

    # -----------------------------------------------------------------
    # MAX and MIN: Iterate over cells (344K), vectorize across years.
    #   For each cell i, gather neighbor rows from X and compute
    #   column-wise max/min.
    # -----------------------------------------------------------------
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (i in seq_len(n_cells)) {
      nb_i <- rook_neighbors_unique[[i]]
      if (length(nb_i) == 1L && nb_i[1] == 0L) next
      if (length(nb_i) == 0L) next

      # Sub-matrix of neighbor values: length(nb_i) x n_years
      nb_mat <- X[nb_i, , drop = FALSE]

      if (length(nb_i) == 1L) {
        # Single neighbor: max = min = that value
        neighbor_max[i, ] <- nb_mat[1, ]
        neighbor_min[i, ] <- nb_mat[1, ]
      } else {
        # suppressWarnings for columns that are all NA
        neighbor_max[i, ] <- suppressWarnings(apply(nb_mat, 2, max, na.rm = TRUE))
        neighbor_min[i, ] <- suppressWarnings(apply(nb_mat, 2, min, na.rm = TRUE))
      }
    }

    # Fix Inf/-Inf from all-NA columns back to NA
    neighbor_max[is.infinite(neighbor_max)] <- NA_real_
    neighbor_min[is.infinite(neighbor_min)] <- NA_real_

    # --- Handle NA propagation to match original logic ---
    # Original: if all neighbor values are NA for a cell-year, return NA.
    # sparse mat-mul treats NA as 0 in the sum, so we need to correct mean
    # where ALL neighbors are NA for a given cell-year.
    # Count non-NA neighbors per cell-year via the adjacency matrix:
    not_na   <- !is.na(X)
    storage.mode(not_na) <- "double"
    n_valid  <- as.matrix(A %*% not_na)  # n_cells x n_years

    # Where n_valid == 0, all neighbors were NA -> result should be NA
    all_na_mask <- (n_valid == 0)
    neighbor_mean[all_na_mask] <- NA_real_

    # Correct the mean: sparse mat-mul summed only non-NA values (NAs became 0),
    # so we need sum-of-non-NA / count-of-non-NA.
    X_zero <- X
    X_zero[is.na(X_zero)] <- 0
    neighbor_sum_corrected <- as.matrix(A %*% X_zero)
    neighbor_mean <- neighbor_sum_corrected / n_valid
    neighbor_mean[all_na_mask] <- NA_real_
    neighbor_mean[n_neighbors == 0, ] <- NA_real_

    # -----------------------------------------------------------------
    # STEP 4: Write results back to cell_data in the correct row order.
    # -----------------------------------------------------------------
    # cell_data is sorted by (spatial_idx, year_idx), so:
    cell_data[, paste0("neighbor_max_", var_name) :=
                  as.numeric(neighbor_max[cbind(spatial_idx, year_idx)])]
    cell_data[, paste0("neighbor_min_", var_name) :=
                  as.numeric(neighbor_min[cbind(spatial_idx, year_idx)])]
    cell_data[, paste0("neighbor_mean_", var_name) :=
                  as.numeric(neighbor_mean[cbind(spatial_idx, year_idx)])]

    cat("  Done:", var_name, "\n")
  }

  # Clean up helper columns
  cell_data[, c("spatial_idx", "year_idx") := NULL]

  if (was_df) cell_data <- as.data.frame(cell_data)

  return(cell_data)
}

# =============================================================================
# USAGE
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# Now cell_data has 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# Proceed with prediction using the pre-trained Random Forest model as before.
# The model is NOT retrained; the numerical results are identical.
```

---

## Further Optimization: Vectorize Max/Min with `matrixStats` or Rcpp

The `for` loop over 344K cells for max/min is the remaining bottleneck (~1–5 minutes). To push it under 30 seconds:

```r
# Alternative: Rcpp implementation for max/min across neighbors
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_maxmin_cpp(NumericMatrix X,
                         List nb,
                         int n_cells,
                         int n_years) {
  NumericMatrix out_max(n_cells, n_years);
  NumericMatrix out_min(n_cells, n_years);

  // Initialize with NA
  std::fill(out_max.begin(), out_max.end(), NA_REAL);
  std::fill(out_min.begin(), out_min.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    IntegerVector nb_i = nb[i];
    // spdep nb: no-neighbor is encoded as single 0
    if (nb_i.size() == 1 && nb_i[0] == 0) continue;
    if (nb_i.size() == 0) continue;

    for (int t = 0; t < n_years; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      bool any_valid = false;

      for (int k = 0; k < nb_i.size(); k++) {
        // nb indices are 1-based
        double val = X(nb_i[k] - 1, t);
        if (!R_IsNA(val)) {
          any_valid = true;
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
        }
      }

      if (any_valid) {
        out_max(i, t) = vmax;
        out_min(i, t) = vmin;
      }
      // else stays NA
    }
  }

  return List::create(Named("max") = out_max,
                      Named("min") = out_min);
}
')
```

Replace the R `for` loop in the main function with:

```r
    mm <- neighbor_maxmin_cpp(X, rook_neighbors_unique, n_cells, n_years)
    neighbor_max <- mm$max
    neighbor_min <- mm$min
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M character-key lookups | Sparse matrix built once from `nb` |
| **Mean** | 6.46M R-level loops per variable | Single sparse `A %*% X` matrix multiply |
| **Max/Min** | 6.46M R-level loops per variable | 344K loops vectorized over 28 years (or Rcpp) |
| **Estimated runtime** | 86+ hours | ~2–10 minutes (R) / ~30 seconds (Rcpp) |
| **RAM** | List of 6.46M index vectors | ~344K×28 matrices (~77 MB each) + sparse A |
| **Numerical results** | Baseline | Identical — same max, min, mean per cell-year |
| **RF model** | Pre-trained | Unchanged, not retrained |