 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance bottleneck is in `build_neighbor_lookup`. This function constructs a lookup mapping **per row** of the panel dataset (~6.46 million rows). For each row, it:

1. Finds the cell's spatial neighbors from the `nb` object (which is **static** — it never changes across years).
2. Then searches for those neighbors **within the same year** by constructing string keys and looking them up in a named vector.

This means the function is redundantly recomputing the same spatial neighbor relationships 28 times (once per year) for each of the 344,208 cells. It produces ~6.46 million list entries, each built via expensive string concatenation (`paste`) and named-vector lookup. The `compute_neighbor_stats` function then iterates over this massive list for each of 5 variables.

**Key insight:** The neighbor *topology* is year-invariant (cell A is always a rook neighbor of cell B regardless of year). Only the *variable values* change by year. The current code conflates these two dimensions, paying an enormous cost to re-derive static structure within a year-varying context.

**Secondary issues:**
- `paste(..., sep="_")` and named-vector lookups over 6.46M keys are slow.
- `lapply` over 6.46M elements with per-element anonymous functions is slow.
- The result is assembled via `do.call(rbind, ...)` on a 6.46M-element list of 3-vectors.

## Optimization Strategy

1. **Separate static topology from dynamic values.** Build the neighbor index once at the *cell level* (344K entries), not the *cell-year level* (6.46M entries). This is a list where element `i` contains the integer positions of cell `i`'s neighbors within the cell-ID ordering.

2. **Operate year-by-year using matrix indexing.** Reshape each variable into a matrix of dimension `(n_cells × n_years)`. For each cell, the neighbor indices point to rows in this matrix. Compute neighbor max/min/mean column-wise (i.e., per year) using vectorized operations.

3. **Use vectorized C-level operations.** Replace `lapply` + per-element R functions with a single pass using pre-built sparse or grouped index structures, or a compiled Rcpp function for the inner loop.

4. **Avoid string operations entirely.** Work with integer indices throughout.

This reduces the problem from ~6.46M list iterations to ~344K list iterations (or a single vectorized pass), and the string-key overhead is eliminated completely.

**Expected speedup:** From ~86+ hours to minutes.

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE: Separate static topology from dynamic variable values
# =============================================================================

library(data.table)

# -------------------------------------------------------------------------
# Step 1: Build a CELL-LEVEL neighbor lookup (done ONCE, static)
#
#   Input:
#     id_order          — vector of cell IDs in the order matching rook_neighbors_unique
#     rook_neighbors_unique — spdep nb object (list of integer neighbor index vectors)
#
#   Output:
#     cell_neighbor_idx — list of length n_cells; element i = integer vector of
#                         neighbor positions in id_order
# -------------------------------------------------------------------------

build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is an nb object: neighbors[[i]] gives integer indices into id_order
  # for the neighbors of cell id_order[i].
  # We just need to strip the nb class and ensure clean integer vectors.
  n <- length(id_order)
  cell_neighbor_idx <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    cell_neighbor_idx[[i]] <- as.integer(nb_i)
  }
  cell_neighbor_idx
}

# -------------------------------------------------------------------------
# Step 2: Reshape panel data into cell × year matrices for each variable
#
#   Assumptions:
#     - cell_data is a data.frame / data.table with columns: id, year, and
#       the neighbor_source_vars.
#     - Every cell appears in every year (balanced panel).
#     - id_order defines the canonical cell ordering.
# -------------------------------------------------------------------------

reshape_to_matrix <- function(cell_data, id_order, years, var_name) {
  # Convert to data.table for speed if not already
  dt <- as.data.table(cell_data)[, .(id, year, val = get(var_name))]

  # Create mapping from cell id -> row index in matrix
  id_map <- setNames(seq_along(id_order), as.character(id_order))
  year_map <- setNames(seq_along(years), as.character(years))

  dt[, row_idx := id_map[as.character(id)]]
  dt[, col_idx := year_map[as.character(year)]]

  mat <- matrix(NA_real_, nrow = length(id_order), ncol = length(years))
  mat[cbind(dt$row_idx, dt$col_idx)] <- dt$val
  mat
}

# -------------------------------------------------------------------------
# Step 3: Compute neighbor stats (max, min, mean) using cell-level topology
#         applied to the cell × year matrix.
#
#   This is the workhorse. For each cell i, we gather the rows of the matrix
#   corresponding to its neighbors, then compute column-wise (per-year)
#   max, min, mean.
#
#   Output: three matrices, each n_cells × n_years
# -------------------------------------------------------------------------

compute_neighbor_stats_matrix <- function(var_matrix, cell_neighbor_idx) {
  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)

  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb_idx <- cell_neighbor_idx[[i]]
    if (length(nb_idx) == 0L) next
    # Extract sub-matrix: rows = neighbors, cols = years
    # If only one neighbor, ensure it's still a matrix
    if (length(nb_idx) == 1L) {
      nb_vals <- matrix(var_matrix[nb_idx, ], nrow = 1L)
    } else {
      nb_vals <- var_matrix[nb_idx, , drop = FALSE]
    }
    # colwise max, min, mean — handling NAs
    max_mat[i, ]  <- apply(nb_vals, 2, function(x) {
      x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x)
    })
    min_mat[i, ]  <- apply(nb_vals, 2, function(x) {
      x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x)
    })
    mean_mat[i, ] <- apply(nb_vals, 2, function(x) {
      x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x)
    })
  }

  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# -------------------------------------------------------------------------
# Step 3-ALT (FAST): Rcpp version for the inner loop
#   If Rcpp is available, this replaces the R loop above and runs in seconds.
# -------------------------------------------------------------------------

if (requireNamespace("Rcpp", quietly = TRUE)) {

  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List compute_neighbor_stats_cpp(NumericMatrix var_matrix,
                                List cell_neighbor_idx) {
  int n_cells = var_matrix.nrow();
  int n_years = var_matrix.ncol();

  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  NumericMatrix mean_mat(n_cells, n_years);

  // Initialize to NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  std::fill(mean_mat.begin(), mean_mat.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    IntegerVector nb_idx = cell_neighbor_idx[i];
    int n_nb = nb_idx.size();
    if (n_nb == 0) continue;

    for (int j = 0; j < n_years; j++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int count = 0;

      for (int k = 0; k < n_nb; k++) {
        // nb_idx is 1-based R index
        double val = var_matrix(nb_idx[k] - 1, j);
        if (!R_IsNA(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
          count++;
        }
      }
      if (count > 0) {
        max_mat(i, j)  = vmax;
        min_mat(i, j)  = vmin;
        mean_mat(i, j) = vsum / count;
      }
    }
  }

  return List::create(
    Named("max")  = max_mat,
    Named("min")  = min_mat,
    Named("mean") = mean_mat
  );
}
')

  use_rcpp <- TRUE
} else {
  use_rcpp <- FALSE
}

# -------------------------------------------------------------------------
# Step 4: Write stats back to the panel data.table
# -------------------------------------------------------------------------

write_stats_to_panel <- function(cell_data_dt, id_order, years,
                                 stats, var_name) {
  id_map   <- setNames(seq_along(id_order), as.character(id_order))
  year_map <- setNames(seq_along(years), as.character(years))

  row_idx <- id_map[as.character(cell_data_dt$id)]
  col_idx <- year_map[as.character(cell_data_dt$year)]
  lin_idx <- cbind(row_idx, col_idx)

  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  cell_data_dt[, (max_col)  := stats$max[lin_idx]]
  cell_data_dt[, (min_col)  := stats$min[lin_idx]]
  cell_data_dt[, (mean_col) := stats$mean[lin_idx]]

  cell_data_dt
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Convert to data.table for efficient column operations
cell_data <- as.data.table(cell_data)

# Unique sorted years
years <- sort(unique(cell_data$year))

# --- STATIC (once) ---
message("Building cell-level neighbor lookup (static topology)...")
cell_neighbor_idx <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# --- DYNAMIC (per variable) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Processing neighbor stats for: %s", var_name))

  # Reshape variable to cell × year matrix
  var_matrix <- reshape_to_matrix(cell_data, id_order, years, var_name)

  # Compute neighbor stats
  if (use_rcpp) {
    stats <- compute_neighbor_stats_cpp(var_matrix, cell_neighbor_idx)
  } else {
    stats <- compute_neighbor_stats_matrix(var_matrix, cell_neighbor_idx)
  }

  # Write back to panel
  cell_data <- write_stats_to_panel(cell_data, id_order, years, stats, var_name)

  # Free memory
  rm(var_matrix, stats)
  gc()
}

message("Neighbor feature computation complete.")

# cell_data now has the same 15 neighbor columns (5 vars × {max, min, mean})
# as the original pipeline, with identical numerical values.
# Proceed to Random Forest prediction with the pre-trained model as before.
# e.g.:  cell_data$prediction <- predict(trained_rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup granularity** | Cell-year (~6.46M entries) | Cell (~344K entries, built once) |
| **String operations** | `paste` + named-vector lookup over 6.46M keys | None — pure integer indexing |
| **Inner loop iterations** | ~6.46M × 5 vars = ~32.3M | ~344K × 5 vars = ~1.72M |
| **Data structure** | Long-format column access per row | Cell×Year matrix — cache-friendly |
| **Rcpp fast path** | N/A | Single C++ loop: ~344K cells × 28 years × ~4 neighbors × 5 vars |
| **Estimated runtime** | 86+ hours | **~2–10 minutes** (Rcpp) / ~30–60 min (pure R) |
| **Numerical result** | Reference | Identical (same max/min/mean over same neighbor sets) |
| **RF model** | Unchanged | Unchanged — no retraining |