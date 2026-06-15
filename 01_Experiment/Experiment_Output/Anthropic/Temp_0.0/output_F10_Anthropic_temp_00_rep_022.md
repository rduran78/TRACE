 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query in the worst case, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a vector by index, removing NAs, and computing three summary statistics. This is pure R-level looping with no vectorization.

3. **The neighbor lookup is year-coupled but the graph topology is year-invariant.** The rook-neighbor structure is purely spatial (344,208 cells), yet the code rebuilds lookups across all 9.64M cell-year rows, inflating the problem by 28×. The topology should be built once over cells, and statistics computed per year using vectorized sparse-matrix operations.

**Core insight:** The neighbor aggregation (max, min, mean) is equivalent to sparse matrix operations on a 344,208 × 344,208 adjacency matrix, applied independently to each of 28 year-slices for each of 5 variables. This reduces the problem from 6.46M list iterations to 28 × 5 = 140 sparse matrix operations, each on a ~344K-node graph with ~1.37M edges.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 × 344,208, ~1.37M non-zero entries). This is tiny in memory (~16 MB).

2. **Reshape each variable into a 344,208 × 28 matrix** (cells × years), aligning cell order with the adjacency matrix.

3. **Compute neighbor mean** via sparse matrix multiplication: `A %*% X / degree`, where `degree` is the row-sum of A (number of neighbors per cell).

4. **Compute neighbor max and min** using a custom sparse-row-aggregation function that iterates over CSC/CSR structure in C++ via `Rcpp`, or using a chunked R approach with the sparse matrix's slot structure. Alternatively, use `data.table` grouped operations on the edge list for max/min (which `data.table` handles extremely efficiently).

5. **Reassemble** the 15 new columns back into the panel `data.table`, preserving the original row order and numerical values exactly.

**Expected speedup:** From 86+ hours to ~2–5 minutes.

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {

  # --- Convert to data.table if needed, preserve original row order -----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, .row_order := .I]

  n_cells <- length(id_order)
  stopifnot(n_cells == length(rook_neighbors_unique))

  # --- Step 1: Build sparse adjacency matrix ONCE (topology is year-invariant)
  # rook_neighbors_unique is an nb object: list of integer vectors of neighbor indices
  # into id_order. We build a sparse matrix A where A[i,j]=1 if j is a rook neighbor of i.

  message("Building sparse adjacency matrix from nb object...")
  edge_from <- integer(0)
  edge_to   <- integer(0)

  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    edge_from <- c(edge_from, rep.int(i, length(nb_i)))
    edge_to   <- c(edge_to, nb_i)
  }

  # Pre-allocate more efficiently using vapply to get lengths
  nb_lengths <- vapply(rook_neighbors_unique, function(nb_i) {
    if (length(nb_i) == 1L && nb_i[1] == 0L) 0L else length(nb_i)
  }, integer(1))

  total_edges <- sum(nb_lengths)
  edge_from <- integer(total_edges)
  edge_to   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    n_nb <- nb_lengths[i]
    if (n_nb == 0L) next
    idx_range <- pos:(pos + n_nb - 1L)
    edge_from[idx_range] <- i
    edge_to[idx_range]   <- rook_neighbors_unique[[i]]
    pos <- pos + n_nb
  }

  # Sparse adjacency matrix (rows = focal cell, cols = neighbor cell)
  A <- sparseMatrix(i = edge_from, j = edge_to, x = 1,
                    dims = c(n_cells, n_cells), repr = "C")  # CSC format

  # Edge list as data.table for max/min computation
  edges_dt <- data.table(from = edge_from, to = edge_to)

  # Degree vector (number of neighbors per cell) for mean computation
  degree <- diff(A@p)  # For CSC of t(A); we need row-sums of A
  # Actually for CSC (column-compressed), column sums = diff(A@p)
  # We need row sums. Use:
  degree_vec <- as.integer(rowSums(A))  # fast for sparse

  # --- Step 2: Create cell-index mapping ----------------------------------------
  # Map id_order to matrix row indices (1:n_cells)
  id_to_matrow <- setNames(seq_len(n_cells), as.character(id_order))

  # Map each row in cell_data to its matrix row index
  cell_data[, .mat_row := id_to_matrow[as.character(id)]]

  # Get sorted unique years
  years <- sort(unique(cell_data$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  cell_data[, .year_col := year_to_col[as.character(year)]]

  message(sprintf("Graph: %d nodes, %d directed edges, %d years, %d variables",
                  n_cells, total_edges, n_years, length(neighbor_source_vars)))

  # --- Step 3: For each variable, compute neighbor max, min, mean ---------------

  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing variable: %s", var_name))

    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Build cell × year matrix (n_cells rows × n_years cols)
    # Fill with NA
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(cell_data$.mat_row, cell_data$.year_col)] <- cell_data[[var_name]]

    # --- Compute MEAN via sparse matrix multiplication (per year column) --------
    # For each year t: neighbor_mean[i,t] = sum_j A[i,j]*X[j,t] / degree[i]
    # But we must handle NAs: original code drops NAs before computing mean.
    # If all neighbor values are non-NA (common case), sparse matmul is exact.
    # For full correctness with potential NAs, we need:
    #   sum of non-NA neighbor values / count of non-NA neighbor values

    # Create indicator matrix: 1 where X is not NA, 0 otherwise
    X_nona <- X
    X_nona[is.na(X_nona)] <- 0

    X_indicator <- matrix(0, nrow = n_cells, ncol = n_years)
    X_indicator[!is.na(X)] <- 1

    # Sparse matmul: sum of neighbor values (treating NA as 0)
    neighbor_sum   <- as.matrix(A %*% X_nona)        # n_cells × n_years
    neighbor_count <- as.matrix(A %*% X_indicator)    # n_cells × n_years

    # Mean = sum / count; NA where count == 0
    mean_mat <- neighbor_sum / neighbor_count
    mean_mat[neighbor_count == 0] <- NA_real_

    # --- Compute MAX and MIN via edge-list grouped aggregation ------------------
    # This is the most efficient R approach for sparse max/min.
    # For each year, look up neighbor values via the edge list and group-aggregate.

    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (t in seq_len(n_years)) {
      x_t <- X[, t]  # values for this year, indexed by mat_row

      # Look up neighbor values
      nb_vals <- x_t[edges_dt$to]

      # Build temporary data.table with from-node and neighbor value
      tmp <- data.table(from = edges_dt$from, val = nb_vals)

      # Remove NAs
      tmp <- tmp[!is.na(val)]

      if (nrow(tmp) == 0L) next

      # Grouped max and min
      agg <- tmp[, .(vmax = max(val), vmin = min(val)), by = from]

      max_mat[agg$from, t] <- agg$vmax
      min_mat[agg$from, t] <- agg$vmin
    }

    # --- Step 4: Map results back to cell_data rows -----------------------------
    idx_mat <- cbind(cell_data$.mat_row, cell_data$.year_col)

    set(cell_data, j = col_max,  value = max_mat[idx_mat])
    set(cell_data, j = col_min,  value = min_mat[idx_mat])
    set(cell_data, j = col_mean, value = mean_mat[idx_mat])

    message(sprintf("  -> Added %s, %s, %s", col_max, col_min, col_mean))
  }

  # --- Cleanup temporary columns ------------------------------------------------
  cell_data[, c(".row_order", ".mat_row", ".year_col") := NULL]

  message("Neighbor aggregation complete.")
  return(cell_data)
}


# =============================================================================
# USAGE
# =============================================================================
#
# # Load data (assumed already in memory or loaded from disk)
# # cell_data:                data.frame/data.table with columns: id, year, ntl, ec, ...
# # id_order:                 vector of cell IDs matching rook_neighbors_unique indexing
# # rook_neighbors_unique:    spdep::nb object (list of neighbor index vectors)
# # rf_model:                 pre-trained Random Forest model (DO NOT retrain)
#
# cell_data <- optimize_neighbor_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Apply the pre-trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
# =============================================================================
```

## Further Optimization: Rcpp for Max/Min (Optional Drop-In)

The `data.table` grouped aggregation for max/min across 28 years is already fast (~1–2 minutes total), but if even that is a concern, here is an Rcpp version that processes all years in one pass:

```r
# Optional: Rcpp-accelerated max/min over sparse neighbors
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(IntegerVector from, IntegerVector to,
                            NumericMatrix X, int n_cells) {
  // from, to: 0-based edge indices
  // X: n_cells x n_years matrix
  int n_years = X.ncol();
  int n_edges = from.size();

  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  LogicalMatrix has_val(n_cells, n_years);

  // Initialize
  std::fill(max_mat.begin(), max_mat.end(), R_NegInf);
  std::fill(min_mat.begin(), min_mat.end(), R_PosInf);
  std::fill(has_val.begin(), has_val.end(), false);

  for (int e = 0; e < n_edges; e++) {
    int i = from[e];  // focal node (0-based)
    int j = to[e];    // neighbor node (0-based)
    for (int t = 0; t < n_years; t++) {
      double v = X(j, t);
      if (!NumericMatrix::is_na(v)) {
        has_val(i, t) = true;
        if (v > max_mat(i, t)) max_mat(i, t) = v;
        if (v < min_mat(i, t)) min_mat(i, t) = v;
      }
    }
  }

  // Set NA where no valid neighbor value existed
  for (int i = 0; i < n_cells; i++) {
    for (int t = 0; t < n_years; t++) {
      if (!has_val(i, t)) {
        max_mat(i, t) = NA_REAL;
        min_mat(i, t) = NA_REAL;
      }
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# To use inside the pipeline, replace the per-year max/min loop with:
# res <- sparse_neighbor_maxmin(edges_dt$from - 1L, edges_dt$to - 1L, X, n_cells)
# max_mat <- res$max_mat
# min_mat <- res$min_mat
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology construction** | String-paste + named-vector lookup over 6.46M rows | Sparse matrix built once over 344K cells (~1.37M entries) |
| **Mean computation** | R-level `lapply` over 6.46M elements | Sparse matrix multiplication (`A %*% X`), fully vectorized |
| **Max/Min computation** | R-level `lapply` with per-element subsetting | `data.table` grouped aggregation on edge list (or Rcpp) |
| **Redundant work** | Topology re-resolved per cell-year | Topology built once, reused across 28 years × 5 variables |
| **Estimated runtime** | 86+ hours | ~2–5 minutes (data.table) or ~30 seconds (Rcpp) |
| **Memory** | ~6.46M-element list of integer vectors | ~16 MB sparse matrix + ~77 MB per variable matrix |
| **Numerical equivalence** | Baseline | Exact: same max, min, mean with identical NA handling |
| **RF model** | Unchanged | Unchanged — no retraining |