 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting, named-vector lookup, and NA filtering. This is O(n) with enormous constant factors because R's named vector lookup (`idx_lookup[neighbor_keys]`) is hash-based per element but repeated millions of times with string allocation overhead.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements** in an `lapply` loop, extracting and aggregating neighbor values one node-year at a time. This is pure R-level looping with no vectorization.

3. **The topology is year-invariant** (rook neighbors don't change across years), but the lookup is rebuilt monolithically across all cell-years, entangling spatial structure with temporal indexing. This means 28× redundant work in the neighbor resolution step.

**Root cause:** The algorithm treats the problem as a flat list-of-lists over 6.46M rows. It should instead exploit the fact that the spatial adjacency graph has only 344,208 nodes and is **constant across all 28 years**. Neighbor aggregation is a sparse matrix–vector product, which can be done in milliseconds with the right representation.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 × 344,208, ~1.37M nonzeros). This is the graph topology.

2. **Reshape each variable into a 344,208 × 28 matrix** (cells × years). This separates spatial topology from temporal indexing.

3. **Compute neighbor aggregations as sparse matrix operations:**
   - **Mean:** `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix.
   - **Max and Min:** Use a single C++ loop via `Rcpp` over the sparse matrix CSR structure, or use grouped operations. Since sparse matrix algebra doesn't natively support max/min, we use an efficient Rcpp routine operating on the CSC/CSR structure.

4. **Flatten results back** to the original long-format data frame, join, and predict with the existing Random Forest.

**Expected speedup:** From 86+ hours to **minutes**. The sparse matrix has ~1.37M entries; multiplied by 28 years × 5 variables = 140 sparse mat-vec products for mean. Max/min via Rcpp over the same structure is comparably fast.

## Working R Code

```r
# ===========================================================================
# Optimized spatial neighbor aggregation via sparse graph operations
# Preserves numerical equivalence with the original pipeline
# ===========================================================================

library(Matrix)
library(Rcpp)
library(data.table)

# ---- Step 0: Rcpp helper for sparse row-wise max, min over neighbor values --
# This operates on the CSC (dgCMatrix) structure of the adjacency matrix
# and computes, for each row i, the max and min of X[j, ] for all j in N(i).

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// adj is a dgCMatrix (CSC format) of dimension n x n
// xmat is an n x T numeric matrix
// Returns a list with two n x T matrices: max_mat and min_mat
// [[Rcpp::export]]
List sparse_neighbor_maxmin(S4 adj, NumericMatrix xmat) {
  // CSC components
  IntegerVector p = adj.slot("p");       // column pointers, length n+1
  IntegerVector i_idx = adj.slot("i");   // row indices
  // x slot is all 1s for unweighted adjacency; we ignore it

  int n = xmat.nrow();
  int T = xmat.ncol();

  NumericMatrix max_mat(n, T);
  NumericMatrix min_mat(n, T);
  IntegerVector degree(n);

  // Initialize
  std::fill(max_mat.begin(), max_mat.end(), NA_REAL);
  std::fill(min_mat.begin(), min_mat.end(), NA_REAL);
  std::fill(degree.begin(), degree.end(), 0);

  // We need row-oriented iteration, but dgCMatrix is CSC.
  // Strategy: build CSR (row pointers) from the CSC structure.
  // Since adj may not be symmetric, we transpose: in CSC, column j
  // has entries in rows i_idx[p[j]..p[j+1]-1].
  // For row-wise neighbors: row i has neighbor j iff adj[i,j] != 0.
  // In CSC, adj[i,j] != 0 means i appears in column j entries.
  // So we need to iterate by rows -> build CSR.

  // Count entries per row
  std::vector<int> row_count(n, 0);
  int nnz = i_idx.size();
  for (int k = 0; k < nnz; k++) {
    row_count[i_idx[k]]++;
  }

  // Build row pointers
  std::vector<int> row_ptr(n + 1, 0);
  for (int r = 0; r < n; r++) {
    row_ptr[r + 1] = row_ptr[r] + row_count[r];
  }

  // Fill column indices per row
  std::vector<int> col_idx(nnz);
  std::vector<int> row_pos(n, 0); // current fill position per row
  for (int j = 0; j < n; j++) {
    for (int k = p[j]; k < p[j + 1]; k++) {
      int row = i_idx[k];
      int dest = row_ptr[row] + row_pos[row];
      col_idx[dest] = j;
      row_pos[row]++;
    }
  }

  // Now iterate rows and compute max/min
  for (int row = 0; row < n; row++) {
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    int deg = end - start;
    if (deg == 0) continue;

    for (int t = 0; t < T; t++) {
      double mx = R_NegInf;
      double mn = R_PosInf;
      int valid = 0;
      for (int k = start; k < end; k++) {
        double val = xmat(col_idx[k], t);
        if (!R_IsNA(val)) {
          if (val > mx) mx = val;
          if (val < mn) mn = val;
          valid++;
        }
      }
      if (valid > 0) {
        max_mat(row, t) = mx;
        min_mat(row, t) = mn;
      }
      // else remains NA
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# ---- Step 1: Build sparse adjacency matrix once ----------------------------
# rook_neighbors_unique: spdep nb object (list of integer vectors)
# id_order: vector of cell IDs in the order matching the nb object

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj[[i]] contains integer indices of neighbors of node i
  # 0L in spdep means no neighbors
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    from <- c(from, rep(i, length(nbrs)))
    to   <- c(to, nbrs)
  }
  # Sparse matrix: A[i,j] = 1 means j is a neighbor of i
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)  # 344,208
cat("Building sparse adjacency matrix...\n")
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Row-normalized version for mean computation
row_sums <- rowSums(A)
row_sums[row_sums == 0] <- NA  # will produce NA for isolated nodes
A_norm <- Diagonal(x = 1 / ifelse(is.na(row_sums), 1, row_sums)) %*% A

# ---- Step 2: Convert data to data.table for fast reshaping -----------------
cat("Preparing data structures...\n")
dt <- as.data.table(cell_data)

# Create a mapping from cell id to matrix row index (matching nb object order)
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Add the row index
dt[, cell_row := id_to_row[as.character(id)]]

# Determine year columns
years <- sort(unique(dt$year))  # 1992:2019, 28 years
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))
dt[, year_col := year_to_col[as.character(year)]]

# ---- Step 3: For each variable, reshape to matrix, compute stats, reshape back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Function to reshape a variable to n_cells x n_years matrix
var_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_row, dt$year_col)] <- dt[[var_name]]
  mat
}

cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")

for (var_name in neighbor_source_vars) {
  cat("  Processing:", var_name, "... ")
  t0 <- proc.time()

  # Reshape to matrix
  X <- var_to_matrix(dt, var_name, n_cells, n_years)

  # --- Mean: sparse matrix multiplication ---
  # A_norm %*% X gives row i = mean of neighbor values (if all non-NA)
  # For exact equivalence with the original (which filters NAs), we need
  # NA-aware aggregation. Sparse mat-mul treats NAs as values, so we handle:
  # mean_i = sum(A[i,j] * X[j,t], j in N(i) & !is.na) / count(!is.na in N(i))

  # Replace NA with 0 for summation, track counts
  X_nona <- X
  X_nona[is.na(X_nona)] <- 0
  indicator <- (!is.na(X)) * 1.0  # 1 where valid, 0 where NA

  sum_mat   <- as.matrix(A %*% X_nona)         # sum of non-NA neighbor values
  count_mat <- as.matrix(A %*% indicator)       # count of non-NA neighbors

  mean_mat <- sum_mat / count_mat               # NA where count == 0 (0/0 = NaN)
  mean_mat[count_mat == 0] <- NA_real_

  # --- Max and Min: via Rcpp ---
  maxmin <- sparse_neighbor_maxmin(A, X)
  max_mat <- maxmin$max_mat
  min_mat <- maxmin$min_mat

  # --- Write results back to data.table ---
  idx <- cbind(dt$cell_row, dt$year_col)

  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(dt, j = col_max,  value = max_mat[idx])
  set(dt, j = col_min,  value = min_mat[idx])
  set(dt, j = col_mean, value = mean_mat[idx])

  elapsed <- (proc.time() - t0)[3]
  cat(round(elapsed, 1), "seconds\n")
}

# ---- Step 4: Convert back and predict with existing Random Forest ----------
cat("Preparing final dataset...\n")

# Remove helper columns
dt[, c("cell_row", "year_col") := NULL]

# Convert back to data.frame (or keep as data.table for predict)
cell_data <- as.data.frame(dt)

# Predict using the pre-trained Random Forest (do NOT retrain)
cat("Generating predictions...\n")
predictions <- predict(trained_rf_model, newdata = cell_data)

cat("Done.\n")
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | One 344K×344K sparse matrix (built once) |
| **Mean aggregation** | R-level lapply over 6.46M entries | Sparse matrix multiplication (`A %*% X`), vectorized |
| **Max/Min aggregation** | R-level lapply with per-element extraction | Single-pass Rcpp over CSR structure |
| **Topology reuse** | Entangled with year indexing | Spatial graph built once, applied to all 28 year-columns simultaneously |
| **Memory** | ~6.46M-element list of integer vectors | ~1.37M-nonzero sparse matrix + 344K×28 dense matrices (~77 MB each) |
| **Estimated time** | 86+ hours | **~2–5 minutes** |
| **Numerical equivalence** | Baseline | Preserved: identical NA handling for max, min, mean |
| **Trained RF model** | — | Unchanged; only `predict()` is called |

The key insight is that **rook-neighbor aggregation is a sparse linear operator over a time-invariant graph**. Mean is a literal sparse matrix–vector product; max/min require one Rcpp pass over the same sparse structure. Reshaping from long panel to (cells × years) matrices eliminates all per-row string operations and allows the 28 years to be processed as matrix columns in a single sparse operation.