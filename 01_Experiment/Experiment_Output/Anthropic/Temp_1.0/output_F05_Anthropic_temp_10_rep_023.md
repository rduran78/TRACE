 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### Local Inefficiency
The `paste(..., sep="_")` call inside `build_neighbor_lookup` constructs ~6.46M string keys for `idx_lookup` once, but then *for each of the 6.46M rows*, it re-pastes neighbor IDs with the current year to probe that lookup. With an average of ~8 rook neighbors per cell (1,373,394 directed pairs / 344,208 cells ≈ 4 per cell, doubled for bidirectional ≈ 8), that's ~51.7 million `paste` + hash-lookup operations inside the `lapply`. String hashing on millions of keys is inherently slow in R.

### Broader Algorithmic Problem
The real issue is that **the neighbor topology is year-invariant** — a cell's rook neighbors are the same in every year. The current code re-discovers this for every cell-year row by encoding `(id, year)` pairs as strings. This means the 344,208-cell neighbor structure is redundantly traversed 28 times (once per year), doing string work each time.

Furthermore, `compute_neighbor_stats` is called sequentially for each of 5 variables, each time iterating over all 6.46M rows. This is another layer of repeated work.

### Root Cause Summary

| Layer | Waste | Scale |
|---|---|---|
| String key construction | `paste` + named-vector lookup per row | 6.46M × ~8 neighbors = ~51.7M string ops |
| Year-invariant topology re-traversal | Same neighbor set recomputed 28× | 28 × 344,208 = 9,637,824 redundant lookups |
| Per-variable iteration | Full 6.46M-row pass per variable | 5 × 6.46M = 32.3M row visits |

## Optimization Strategy

1. **Separate topology from time**: Build the neighbor index list once at the cell level (344K entries), not the cell-year level (6.46M entries).

2. **Eliminate all string operations**: Use integer indexing throughout. Map cell IDs to integer positions, and for each year-slice, compute a simple integer offset to jump from cell-level neighbor indices to cell-year-level row indices.

3. **Vectorize across years and variables**: Instead of `lapply` over 6.46M rows, operate on a matrix representation where rows = cells, columns = years. For each cell, the neighbor set (row indices into the matrix) is the same across all years. Use matrix slicing to extract neighbor values, then compute `max/min/mean` in a vectorized or compiled fashion.

4. **Use `data.table` for fast joins** and avoid repeated data frame copies.

5. **Process all 5 variables in one pass** over the neighbor structure.

## Working R Code

```r
library(data.table)

# ===========================================================================
# optimized_neighbor_features.R
#
# Replaces: build_neighbor_lookup, compute_neighbor_stats, and the outer loop.
# Preserves: the trained Random Forest model and the original numerical
#            estimand (max, min, mean of each neighbor source variable).
# ===========================================================================

build_and_apply_neighbor_features <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -----------------------------------------------------------------------
  # 0. Convert to data.table for speed; keep original row order
  # -----------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]
  
  # -----------------------------------------------------------------------
  # 1. Build year-invariant, integer-indexed neighbor list  (344K entries)
  #
  #    id_order is the vector of cell IDs in the same order as

  #    rook_neighbors_unique (an spdep nb object): element k of the nb
  #    list gives indices into id_order for the neighbors of id_order[k].
  #
  #    We map every cell ID that appears in dt to its position in id_order.
  # -----------------------------------------------------------------------
  id_order_chr <- as.character(id_order)
  n_cells      <- length(id_order)
  id_to_pos    <- setNames(seq_len(n_cells), id_order_chr)
  
  # spdep nb objects: neighbors[[k]] is an integer vector of positions
  # into id_order (0L means no neighbors — spdep convention).
  # Convert to a clean list: for each cell position, the integer positions
  # of its neighbors in id_order.
  cell_neighbors <- lapply(seq_len(n_cells), function(k) {
    nb <- rook_neighbors_unique[[k]]
    nb <- nb[nb != 0L]           # drop the spdep "no-neighbor" sentinel
    as.integer(nb)
  })
  
  # -----------------------------------------------------------------------
  # 2. Reshape each source variable into a matrix:  cells × years
  #
  #    Row i  = cell at position i in id_order
  #    Col j  = j-th year in sorted unique years
  #
  #    This lets us look up neighbor values with pure integer indexing.
  # -----------------------------------------------------------------------
  years_sorted <- sort(unique(dt$year))
  n_years      <- length(years_sorted)
  year_to_col  <- setNames(seq_len(n_years), as.character(years_sorted))
  
  # Map each row in dt to (cell_position, year_column)
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]
  
  # Build a linear index into an n_cells × n_years matrix for fast fill
  # R matrices are column-major: element [i, j] is at position i + (j-1)*n_cells
  lin_idx <- dt$cell_pos + (dt$year_col - 1L) * n_cells
  
  # Pre-allocate matrices for each variable
  var_matrices <- lapply(neighbor_source_vars, function(vn) {
    mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat[lin_idx] <- dt[[vn]]
    mat
  })
  names(var_matrices) <- neighbor_source_vars
  
  # -----------------------------------------------------------------------
  # 3. Compute neighbor stats:  for every (cell, year), for every variable,
  #    gather the neighbor values, compute max / min / mean.
  #
  #    Strategy: iterate over cells (344K), vectorise across years (28)
  #    and variables (5) for each cell's neighbor set.
  #
  #    For a cell with nb_k neighbors, extracting a nb_k × n_years sub-
  #    matrix and calling colMins / colMaxs / colMeans is fast.
  # -----------------------------------------------------------------------
  
  n_vars <- length(neighbor_source_vars)
  
  # Result matrices: n_cells × n_years for each of 3 stats × n_vars
  # We store them as lists of matrices, then map back to dt.
  res_max  <- lapply(seq_len(n_vars), function(v)
    matrix(NA_real_, nrow = n_cells, ncol = n_years))
  res_min  <- lapply(seq_len(n_vars), function(v)
    matrix(NA_real_, nrow = n_cells, ncol = n_years))
  res_mean <- lapply(seq_len(n_vars), function(v)
    matrix(NA_real_, nrow = n_cells, ncol = n_years))
  
  # Main loop: 344,208 iterations (fast; inner work is vectorised)
  for (k in seq_len(n_cells)) {
    nb <- cell_neighbors[[k]]
    if (length(nb) == 0L) next          # all stats stay NA
    
    for (v in seq_len(n_vars)) {
      # Sub-matrix: neighbors × years  (typically 3-8 rows × 28 cols)
      sub <- var_matrices[[v]][nb, , drop = FALSE]
      
      # colMeans / colMins / colMaxs handle NA via na.rm
      # For small sub-matrices, a simple apply is fine and avoids matrixStats
      # dependency, but matrixStats would be faster if available.
      #
      # We write a small vectorized version that handles all-NA columns.
      for (j in seq_len(n_years)) {
        vals <- sub[, j]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) next    # stays NA
        res_max[[v]][k, j]  <- max(vals)
        res_min[[v]][k, j]  <- min(vals)
        res_mean[[v]][k, j] <- mean(vals)
      }
    }
  }
  
  # -----------------------------------------------------------------------
  # 3b. (Optional, faster) Replace the inner j-loop with matrixStats if
  #     available.  This version is provided as an alternative.
  # -----------------------------------------------------------------------
  use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)
  
  if (use_matrixStats) {
    # Re-run with matrixStats for speed; overwrite results
    for (k in seq_len(n_cells)) {
      nb <- cell_neighbors[[k]]
      if (length(nb) == 0L) next
      
      for (v in seq_len(n_vars)) {
        sub <- var_matrices[[v]][nb, , drop = FALSE]
        res_max[[v]][k, ]  <- matrixStats::colMaxs(sub,  na.rm = TRUE)
        res_min[[v]][k, ]  <- matrixStats::colMins(sub,  na.rm = TRUE)
        res_mean[[v]][k, ] <- matrixStats::colMeans2(sub, na.rm = TRUE)
      }
    }
    
    # matrixStats returns -Inf/Inf for all-NA columns; fix to NA
    for (v in seq_len(n_vars)) {
      res_max[[v]][is.infinite(res_max[[v]])]   <- NA_real_
      res_min[[v]][is.infinite(res_min[[v]])]   <- NA_real_
      res_mean[[v]][is.nan(res_mean[[v]])]      <- NA_real_
    }
  }
  
  # -----------------------------------------------------------------------
  # 4. Map results back to the data.table rows
  #
  #    Each result matrix is n_cells × n_years.  dt already has cell_pos
  #    and year_col, so we use the same lin_idx to extract values.
  # -----------------------------------------------------------------------
  for (v in seq_len(n_vars)) {
    vn <- neighbor_source_vars[v]
    col_max  <- paste0(vn, "_neighbor_max")
    col_min  <- paste0(vn, "_neighbor_min")
    col_mean <- paste0(vn, "_neighbor_mean")
    
    dt[, (col_max)  := res_max[[v]][lin_idx]]
    dt[, (col_min)  := res_min[[v]][lin_idx]]
    dt[, (col_mean) := res_mean[[v]][lin_idx]]
  }
  
  # -----------------------------------------------------------------------
  # 5. Restore original order, drop helper columns, return
  # -----------------------------------------------------------------------
  setorder(dt, .row_order)
  dt[, c(".row_order", "cell_pos", "year_col") := NULL]
  
  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# ===========================================================================
# USAGE — drop-in replacement for the original outer loop
# ===========================================================================
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_and_apply_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # The trained Random Forest model is untouched. Feed cell_data into
# # predict(rf_model, cell_data) exactly as before.
```

## Further Optimization: Rcpp Inner Loop (Optional, Massive Speedup)

The 344K × 28 × 5 inner loop in pure R is already ~100× faster than the original, but if it's still a bottleneck (estimated ~10–20 minutes), the innermost loop can be pushed to C++ trivially:

```r
# Save as neighbor_stats.cpp, then: Rcpp::sourceCpp("neighbor_stats.cpp")

Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List compute_neighbor_stats_cpp(NumericMatrix varmat, List cell_neighbors) {
  int n_cells = varmat.nrow();
  int n_years = varmat.ncol();
  
  NumericMatrix rmax(n_cells, n_years);
  NumericMatrix rmin(n_cells, n_years);
  NumericMatrix rmean(n_cells, n_years);
  
  // Fill with NA
  std::fill(rmax.begin(),  rmax.end(),  NA_REAL);
  std::fill(rmin.begin(),  rmin.end(),  NA_REAL);
  std::fill(rmean.begin(), rmean.end(), NA_REAL);
  
  for (int k = 0; k < n_cells; k++) {
    IntegerVector nb = cell_neighbors[k];
    int n_nb = nb.size();
    if (n_nb == 0) continue;
    
    for (int j = 0; j < n_years; j++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;
      
      for (int m = 0; m < n_nb; m++) {
        double val = varmat(nb[m] - 1, j);   // R is 1-indexed
        if (ISNA(val)) continue;
        if (val > vmax) vmax = val;
        if (val < vmin) vmin = val;
        vsum += val;
        cnt++;
      }
      
      if (cnt > 0) {
        rmax(k, j)  = vmax;
        rmin(k, j)  = vmin;
        rmean(k, j) = vsum / cnt;
      }
    }
  }
  
  return List::create(
    Named("max")  = rmax,
    Named("min")  = rmin,
    Named("mean") = rmean
  );
}
')
```

Replace the inner loop in the R function with:

```r
for (v in seq_len(n_vars)) {
  stats <- compute_neighbor_stats_cpp(var_matrices[[v]], cell_neighbors)
  res_max[[v]]  <- stats$max
  res_min[[v]]  <- stats$min
  res_mean[[v]] <- stats$mean
}
```

## Performance Comparison

| Aspect | Original | Optimized (R) | Optimized (Rcpp) |
|---|---|---|---|
| String operations | ~51.7M `paste` + hash | **0** | **0** |
| Neighbor lookups | 6.46M (cell-year) | 344K (cell) | 344K (cell) |
| Per-variable passes over rows | 5 × 6.46M | 5 × 344K (matrix) | 5 × 344K (compiled) |
| RAM overhead | Large named character vector | ~5 matrices of 344K × 28 doubles (~38 MB each) | Same |
| Estimated time | **86+ hours** | **~15–30 min** | **~1–3 min** |

The key insight: **factor the topology (spatial) from the panel (temporal) dimension**, eliminate all string encoding, and operate on dense integer-indexed matrices. The trained Random Forest model and all numerical outputs are preserved identically.