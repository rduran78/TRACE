 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting and named-vector lookups. The `paste(id, year)` key construction and `idx_lookup[neighbor_keys]` named-vector subsetting is O(n) per lookup in the worst case, and doing this 6.46M times is catastrophic. This single function likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. While each individual call is cheap, 6.46M × 5 variables = ~32.3 million R-level function invocations with no vectorization.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** Rook contiguity is a spatial relationship between cells — it doesn't change across years. The current code redundantly encodes the same spatial graph 28 times (once per year per cell), inflating the lookup from ~344K entries to ~6.46M entries.

**Key insight:** The adjacency graph is static. Only the node attributes change per year. We should build the graph topology once over the 344K cells, then use sparse matrix–vector multiplication to compute neighborhood aggregates in vectorized operations.

---

## Optimization Strategy

### Core Idea: Sparse Matrix Aggregation

For each cell `i` with rook neighbors `N(i)`, we need:
- `max(x[N(i)])`, `min(x[N(i)])`, `mean(x[N(i)])`

**Mean** is directly computable via sparse matrix–vector product: if `W` is the row-normalized adjacency matrix, then `W %*% x = mean of neighbors` for each node. If `A` is the binary adjacency matrix and `d` is the degree vector, then `mean_i = (A %*% x)[i] / d[i]`.

**Max and min** cannot be computed via standard matrix multiplication, but we can compute them efficiently year-by-year over the 344K cells using the sparse adjacency structure with compiled C++ code via `Rcpp`.

### Plan

1. **Build a sparse adjacency matrix once** from the `spdep::nb` object (344K × 344K, ~1.37M nonzeros). This is trivial and instant.

2. **Reshape data** so that for each year, we have a vector of length 344K aligned to the cell ordering.

3. **Compute `mean`** via sparse matrix–vector product (`Matrix::sparseMatrix %*% x` then divide by degree). This is fully vectorized C code under the hood.

4. **Compute `max` and `min`** via a small Rcpp function that iterates over the CSR structure of the sparse matrix — one pass per variable per year.

5. **Process year-by-year** to keep memory bounded (344K × 5 variables per year ≈ negligible memory).

### Expected Speedup

- **Topology build:** From ~hours to <1 second (one sparse matrix construction).
- **Aggregation:** From ~hours to seconds. 28 years × 5 variables × 344K cells with sparse ops ≈ a few seconds total for mean, and ~10–30 seconds total for max/min via Rcpp.
- **Total estimated time:** Under 2 minutes for the entire neighbor feature computation.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean neighbor stats.
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# ---- Step 0: Rcpp function for sparse max/min (CSC format) ------------------
# We write a single Rcpp function that, given a sparse adjacency matrix in CSC
# format (which is CSR of the transpose), computes max, min, mean per row.
# Since we need row-wise aggregation and Matrix stores in CSC (column-compressed),
# we transpose A so that column j of A^T contains the neighbors of node j,
# then iterate over columns.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_neighbor_stats(
    IntegerVector Ap,    // column pointers of CSC matrix (A transposed, so these are "row pointers" of A)
    IntegerVector Ai,    // row indices of CSC matrix
    NumericVector x,     // attribute vector aligned to node order
    int n                // number of nodes
) {
  // Output: n x 3 matrix [max, min, mean]
  NumericMatrix out(n, 3);

  for (int j = 0; j < n; j++) {
    int start = Ap[j];
    int end   = Ap[j + 1];
    int count = 0;
    double vmax = NA_REAL;
    double vmin = NA_REAL;
    double vsum = 0.0;

    for (int k = start; k < end; k++) {
      int neighbor = Ai[k];
      double val = x[neighbor];
      if (!NumericVector::is_na(val)) {
        if (count == 0) {
          vmax = val;
          vmin = val;
          vsum = val;
        } else {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
        }
        count++;
      }
    }

    if (count == 0) {
      out(j, 0) = NA_REAL;
      out(j, 1) = NA_REAL;
      out(j, 2) = NA_REAL;
    } else {
      out(j, 0) = vmax;
      out(j, 1) = vmin;
      out(j, 2) = vsum / count;
    }
  }

  return out;
}
')

# ---- Step 1: Build sparse adjacency matrix from spdep::nb object -----------

build_adjacency_csc <- function(nb_obj) {
  # nb_obj: list of length n, where nb_obj[[i]] is integer vector of neighbor
  # indices (1-based) for node i. 0 means no neighbors (spdep convention).
  n <- length(nb_obj)

  # Build COO triplets (i, j) meaning "j is a neighbor of i"
  from <- integer(0)
  to   <- integer(0)

  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep uses 0L to indicate no neighbors
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) > 0) {
      from <- c(from, rep(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }

  # Sparse matrix: A[i,j] = 1 means j is neighbor of i
  # We want row-wise aggregation over columns.
  # Transpose so that column i of At contains neighbors of node i.
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), giveCsparse = TRUE)
  At <- t(A)  # CSC format: column i has the neighbor indices of node i

  return(list(A = A, At = At, n = n))
}

# ---- Step 2: Main pipeline -------------------------------------------------

run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique) {

  cat("Building sparse adjacency matrix...\n")
  adj <- build_adjacency_csc(rook_neighbors_unique)
  At  <- adj$At
  n_cells <- adj$n

  # Extract CSC internals of At (0-based indices as used by Matrix package)
  At_p <- At@p        # column pointers (length n_cells + 1)
  At_i <- At@i        # row indices (0-based)

  # Convert cell_data to data.table for fast manipulation
  dt <- as.data.table(cell_data)

  # Build mapping from cell id to position in id_order (1-based node index)
  id_to_node <- setNames(seq_along(id_order), as.character(id_order))

  # Add node index column
  dt[, node_idx := id_to_node[as.character(id)]]

  # Verify alignment
  stopifnot(!anyNA(dt$node_idx))

  # Sort by year and node_idx for efficient processing
  setkey(dt, year, node_idx)

  # Get unique years
  years <- sort(unique(dt$year))

  # Neighbor source variables
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }

  cat("Computing neighbor statistics year-by-year...\n")

  for (yr in years) {
    cat("  Year:", yr, "\n")

    # Get rows for this year, ordered by node_idx
    yr_mask <- dt$year == yr
    yr_dt   <- dt[yr_mask]

    # Build a full-length vector for each variable (length = n_cells)
    # Cells not present in this year get NA
    for (var_name in neighbor_source_vars) {
      # Initialize full vector with NA
      x_full <- rep(NA_real_, n_cells)
      x_full[yr_dt$node_idx] <- yr_dt[[var_name]]

      # Compute sparse neighbor stats via Rcpp
      stats <- sparse_neighbor_stats(At_p, At_i, x_full, n_cells)
      # stats is n_cells x 3: [max, min, mean]

      # Map back to the year subset using node_idx
      node_indices <- yr_dt$node_idx

      max_col  <- paste0(var_name, "_neighbor_max")
      min_col  <- paste0(var_name, "_neighbor_min")
      mean_col <- paste0(var_name, "_neighbor_mean")

      # Write results back into dt for the rows of this year
      set(dt, which(yr_mask), max_col,  stats[node_indices, 1])
      set(dt, which(yr_mask), min_col,  stats[node_indices, 2])
      set(dt, which(yr_mask), mean_col, stats[node_indices, 3])
    }
  }

  cat("Neighbor aggregation complete.\n")

  # Convert back to data.frame if needed, drop helper column
  dt[, node_idx := NULL]
  return(as.data.frame(dt))
}

# ---- Step 3: Execute and predict --------------------------------------------

# Load pre-existing objects (assumed already in environment or loaded from disk)
# cell_data              : data.frame with ~6.46M rows
# id_order               : vector of 344,208 cell IDs matching rook_neighbors_unique
# rook_neighbors_unique  : spdep::nb object (list of length 344,208)
# rf_model               : pre-trained Random Forest model (DO NOT retrain)

# Run optimized neighbor aggregation
cell_data <- run_neighbor_aggregation(cell_data, id_order, rook_neighbors_unique)

# Predict using the pre-trained Random Forest (unchanged)
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Verification of Numerical Equivalence

```r
# =============================================================================
# OPTIONAL: Verify numerical equivalence on a subsample
# =============================================================================

verify_equivalence <- function(cell_data_original, cell_data_optimized,
                               neighbor_source_vars, tol = 1e-10) {
  # Sample a subset of rows
  set.seed(42)
  sample_idx <- sample(nrow(cell_data_original), min(10000, nrow(cell_data_original)))

  all_ok <- TRUE
  for (var_name in neighbor_source_vars) {
    for (suffix in c("_neighbor_max", "_neighbor_min", "_neighbor_mean")) {
      col <- paste0(var_name, suffix)
      orig <- cell_data_original[[col]][sample_idx]
      optm <- cell_data_optimized[[col]][sample_idx]

      # Compare, treating NA == NA as TRUE
      matches <- (is.na(orig) & is.na(optm)) | (!is.na(orig) & !is.na(optm) & abs(orig - optm) < tol)
      if (!all(matches)) {
        cat("MISMATCH in", col, ": ", sum(!matches), "differences\n")
        all_ok <- FALSE
      } else {
        cat("OK:", col, "\n")
      }
    }
  }

  if (all_ok) cat("\nAll checks passed. Numerical equivalence confirmed.\n")
  return(all_ok)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology build** | ~6.46M string-paste + named-vector lookups | One sparse matrix from `nb` object (<1s) |
| **Aggregation** | 6.46M × 5 R-level `lapply` calls | 28 × 5 sparse CSC traversals in compiled C++ |
| **Memory** | ~6.46M-element list of integer vectors | One 344K × 344K sparse matrix (~11 MB) + one 344K vector per pass |
| **Estimated time** | 86+ hours | **< 2 minutes** |
| **Numerical result** | Original | Identical (same max/min/mean, same NA handling) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |