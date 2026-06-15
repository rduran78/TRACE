 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three major bottlenecks:

1. **O(n²)-scale list construction in `build_neighbor_lookup`**: For each of the ~6.46 million rows, it performs string-pasting, named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but still carry significant per-call overhead at this scale. The `paste(..., sep="_")` call alone creates millions of temporary string objects.

2. **Row-wise `lapply` in `compute_neighbor_stats`**: Called 5 times (once per variable) × 6.46M rows = ~32.3 million R-level function invocations, each performing subsetting, NA removal, and three summary statistics. R's interpreted loop overhead dominates here.

3. **Redundant topology work per row**: The neighbor lookup is built at the *cell-year* level (~6.46M entries), but the topology is invariant across years. There are only ~344K cells and ~1.37M directed edges. The same adjacency structure is needlessly replicated 28 times.

**Memory pressure**: The 6.46M-element list-of-integer-vectors in `neighbor_lookup`, plus intermediate string vectors, likely causes repeated GC pauses and can approach the 16 GB ceiling.

---

## Optimization Strategy

### Core insight: Separate topology (cell-level) from attributes (cell-year level)

1. **Build a sparse adjacency matrix once** at the cell level (344K × 344K, ~1.37M non-zeros). Use `Matrix::sparseMatrix` in CSC format.

2. **Reshape each variable into a cell × year matrix** (344K × 28). This is compact (~77 MB per variable in double precision).

3. **Compute neighbor statistics via sparse matrix–dense matrix multiplication and analogous sparse operations**:
   - **Mean**: `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix (each row sums to 1 over its neighbors). This gives exact neighbor means in one sparse matrix multiply.
   - **Max / Min**: Use a CSR traversal. R's `Matrix` package stores CSC; we transpose to get CSR-equivalent access. Then iterate over cells (not cell-years) — only 344K iterations, each touching ~4 neighbors on average. This is done column-by-column (28 years) in compiled C++ via `Rcpp`.

4. **Reshape results back** to the long panel format and column-bind to `cell_data`.

5. **Predict** using the pre-trained Random Forest model unchanged.

**Expected speedup**: From ~86+ hours to **minutes**. The sparse matrix multiply for means is essentially free (~seconds). The Rcpp loop for max/min over 344K × 28 × 5 ≈ 48M cell-year-variable computations with ~4 neighbors each is ~200M comparisons — trivial for compiled code.

---

## Working R Code

```r
# =============================================================================
# Optimized spatial neighbor feature computation
# =============================================================================
# Prerequisites:
#   cell_data            — data.frame/data.table with columns: id, year, ntl, ec,
#                          pop_density, def, usd_est_n2, ... (~6.46M rows)
#   id_order             — integer vector of 344,208 cell IDs in the order used
#                          by rook_neighbors_unique
#   rook_neighbors_unique — spdep nb object (list of length 344,208)
#   rf_model             — pre-trained Random Forest model
# =============================================================================

library(data.table)
library(Matrix)
library(Rcpp)

# ---------- Step 0: Convert cell_data to data.table for speed ----------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---------- Step 1: Build sparse adjacency matrix (cell-level) ---------------
# One-time cost. rook_neighbors_unique is an nb object: list of integer index
# vectors (indices into id_order).

message("Building sparse adjacency matrix...")
n_cells <- length(id_order)

# Construct COO triplets from nb object
edge_from <- integer(0)
edge_to   <- integer(0)

# Pre-allocate by estimating total edges
total_edges <- sum(vapply(rook_neighbors_unique, length, integer(1)))
edge_from   <- integer(total_edges)
edge_to     <- integer(total_edges)

pos <- 1L
for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  # spdep nb objects use 0L for no-neighbor indicator; filter those
  nb_i <- nb_i[nb_i > 0L]
  n_nb <- length(nb_i)
  if (n_nb > 0L) {
    edge_from[pos:(pos + n_nb - 1L)] <- i
    edge_to[pos:(pos + n_nb - 1L)]   <- nb_i
    pos <- pos + n_nb
  }
}
# Trim if over-allocated
edge_from <- edge_from[seq_len(pos - 1L)]
edge_to   <- edge_to[seq_len(pos - 1L)]

# Adjacency matrix A: A[i,j] = 1 means j is a neighbor of i
# (row i aggregates over its neighbors in columns)
A <- sparseMatrix(
  i = edge_from,
  j = edge_to,
  x = rep(1, length(edge_from)),
  dims = c(n_cells, n_cells)
)

# Row-normalized version for computing means
row_counts <- diff(A@p)  # number of non-zeros per column in CSC — but we need per row
# For CSR-like row sums:
row_sums <- rowSums(A)
row_sums[row_sums == 0] <- NA_real_  # cells with no neighbors -> NA mean

# Normalized adjacency for mean computation
A_norm <- A
# Divide each row by its count: Diagonal^{-1} %*% A
inv_row_sums <- ifelse(is.na(row_sums), 0, 1 / row_sums)
D_inv <- Diagonal(x = inv_row_sums)
A_norm <- D_inv %*% A

message(sprintf("  Adjacency: %d cells, %d directed edges", n_cells, length(edge_from)))

# ---------- Step 2: Create cell index mapping --------------------------------
# Map cell id -> position (1..n_cells) matching id_order
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# Map years to column indices
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_len(n_years), as.character(years))

# Add matrix indices to cell_data
cell_data[, `:=`(
  cell_pos = id_to_pos[as.character(id)],
  year_col = year_to_col[as.character(year)]
)]

# ---------- Step 3: Rcpp function for sparse max/min ------------------------
cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// A is in dgCMatrix (CSC) format. We need row-wise access, so we
// work with the transpose (which gives us CSC of A^T = CSR of A).
// At_p, At_i, At_x come from the CSC representation of t(A).
// X is n_cells x n_years matrix (column-major).
// Returns a list of two matrices: max_mat and min_mat (n_cells x n_years).

// [[Rcpp::export]]
List sparse_row_max_min(IntegerVector At_p, IntegerVector At_i,
                        NumericMatrix X) {
  int n = X.nrow();
  int m = X.ncol();
  NumericMatrix max_mat(n, m);
  NumericMatrix min_mat(n, m);

  // Initialize to NA
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    int start = At_p[i];
    int end   = At_p[i + 1];
    if (start == end) continue;  // no neighbors, stays NA

    for (int yr = 0; yr < m; yr++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int k = start; k < end; k++) {
        int j = At_i[k];  // neighbor index (0-based)
        double val = X(j, yr);
        if (!R_IsNA(val) && !ISNAN(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(i, yr) = cur_max;
        min_mat(i, yr) = cur_min;
      }
      // else stays NA
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
', depends = "Rcpp")

# Precompute the transpose of A for CSR-like row access in Rcpp
At <- t(A)  # At is CSC of A^T = CSR of A
# Extract slots (0-based indices as used by Matrix package)
At_p <- At@p
At_i <- At@i

# ---------- Step 4: Process each variable ------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor statistics for ", length(neighbor_source_vars), " variables...")

for (var_name in neighbor_source_vars) {
  message(sprintf("  Processing: %s", var_name))

  # --- 4a: Reshape long -> cell x year matrix --------------------------------
  # Use data.table fast indexing
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- cell_data[[var_name]]
  cpos <- cell_data$cell_pos
  ycol <- cell_data$year_col

  # Vectorized assignment
  idx <- cbind(cpos, ycol)
  X[idx] <- vals

  # --- 4b: Neighbor MEAN via sparse matrix multiply --------------------------
  # A_norm %*% X gives (n_cells x n_years) matrix of neighbor means.
  # Cells with no neighbors get 0 from the multiply (since inv_row_sums=0);
  # we need to convert those to NA.
  mean_mat <- as.matrix(A_norm %*% X)
  # Mask out cells with no neighbors
  no_neighbor <- is.na(row_sums)
  if (any(no_neighbor)) {
    mean_mat[no_neighbor, ] <- NA_real_
  }
  # Also, if all neighbor values were NA for a cell-year, A_norm %*% X would
  # produce a value based on 0s (NA * 0 = 0 in sparse multiply). We need to
  # handle this correctly.
  # The sparse multiply treats NA as 0. We must correct for this.

  # More robust mean: compute sum of non-NA neighbors and count of non-NA neighbors.
  # Replace NAs with 0 for sum, and use indicator for count.
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  X_ind  <- matrix(1, nrow = n_cells, ncol = n_years)
  X_ind[is.na(X)] <- 0

  sum_mat   <- as.matrix(A %*% X_zero)      # sum of non-NA neighbor values
  count_mat <- as.matrix(A %*% X_ind)        # count of non-NA neighbors

  mean_mat <- ifelse(count_mat > 0, sum_mat / count_mat, NA_real_)

  # --- 4c: Neighbor MAX and MIN via Rcpp sparse traversal --------------------
  maxmin <- sparse_row_max_min(At_p, At_i, X)
  max_mat <- maxmin$max_mat
  min_mat <- maxmin$min_mat

  # --- 4d: Map results back to long format -----------------------------------
  # Use the same (cell_pos, year_col) index to read back
  new_max  <- max_mat[idx]
  new_min  <- min_mat[idx]
  new_mean <- mean_mat[idx]

  # Add columns to cell_data (preserving original naming convention)
  max_col  <- paste0(var_name, "_max")
  min_col  <- paste0(var_name, "_min")
  mean_col <- paste0(var_name, "_mean")

  set(cell_data, j = max_col,  value = new_max)
  set(cell_data, j = min_col,  value = new_min)
  set(cell_data, j = mean_col, value = new_mean)

  message(sprintf("    Added: %s, %s, %s", max_col, min_col, mean_col))
}

# ---------- Step 5: Clean up temporary columns --------------------------------
cell_data[, `:=`(cell_pos = NULL, year_col = NULL)]

message("Neighbor feature computation complete.")

# ---------- Step 6: Predict using pre-trained Random Forest -------------------
# The RF model is unchanged — same features, same numerical values.
# Convert back to data.frame if the model expects one.
predictions <- predict(rf_model, newdata = as.data.frame(cell_data))

message("Prediction complete.")
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Topology representation** | 6.46M-element list of integer vectors (replicated across years) | Single 344K × 344K sparse matrix (~1.37M non-zeros, ~33 MB) |
| **Mean computation** | R `lapply` → `mean()` per row (6.46M calls) | Sparse matrix multiply `A %*% X` (milliseconds) |
| **Max/Min computation** | R `lapply` → `max()`/`min()` per row | Rcpp CSR traversal: 344K rows × 28 years × ~4 neighbors (seconds) |
| **Per-variable overhead** | ~17 hours | ~10–30 seconds |
| **Total estimated time** | 86+ hours | **2–5 minutes** |
| **Peak RAM** | Exceeded 16 GB (string keys, lists) | ~2–3 GB (sparse matrix + five 344K×28 dense matrices) |
| **Numerical equivalence** | — | ✅ Identical `max`, `min`, `mean` over non-NA rook neighbors per cell-year |
| **RF model** | — | ✅ Untouched, used only for `predict()` |

### Why numerical equivalence is preserved

- **Mean**: We compute `sum(non-NA neighbor values) / count(non-NA neighbors)` using two sparse multiplies on the NA-zeroed matrix and an indicator matrix, then divide — this is algebraically identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min**: The Rcpp function iterates over exactly the same neighbor set (same sparse adjacency) and applies the same NA-skip logic, yielding identical results to `max(neighbor_vals[!is.na(neighbor_vals)])` and `min(...)`.
- **Edge cases**: Zero-neighbor cells and all-NA-neighbor cell-years both produce `NA`, matching the original `if (length(idx) == 0) return(c(NA, NA, NA))` and `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` logic.