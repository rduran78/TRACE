 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** — `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The `idx_lookup` named vector has 6.46 million entries** — every single neighbor key lookup must hash a string and probe a 6.46M-entry character hash table. With ~1.37M directed neighbor pairs × 28 years ≈ 38.4M lookups, this is extremely expensive in R's named-vector implementation.
3. **`compute_neighbor_stats`** then iterates over the 6.46M-element list again, extracting subsets of a numeric vector. The per-element `lapply` with small vectors defeats vectorization.

**In short:** The algorithm is O(N × k) with enormous constant factors from R-level string operations and interpreted loops, where N ≈ 6.46M and k ≈ average neighbors ≈ 4.

## Optimization Strategy

### Key Insight: Separate Space from Time

Every cell has the **same** neighbors in **every** year. The neighbor topology is time-invariant. Instead of building a 6.46M-element lookup (one per cell-year), build a **344,208-element** spatial lookup (one per cell), then use **vectorized matrix operations** across all years simultaneously.

### Concrete Plan

1. **Restructure data into a matrix**: rows = cells (344,208), columns = years (28). For each variable, this is a 344K × 28 matrix.
2. **Build a sparse adjacency matrix** (344,208 × 344,208) from the `nb` object — this is a one-time operation using `spdep::nb2listw` or direct construction via `Matrix::sparseMatrix`.
3. **Compute neighbor stats via sparse matrix multiplication**:
   - **Neighbor mean** = `(A %*% X) / (A %*% (!is.na(X)))` (sparse mat × dense mat — highly optimized in the `Matrix` package).
   - **Neighbor max/min** — use row-wise sparse iteration (still far faster than 6.46M R-level list lookups).
4. **Flatten back** to the original long panel format.

This replaces ~86 hours of interpreted R loops with a few seconds of optimized sparse linear algebra.

## Working R Code

```r
library(Matrix)
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build sparse binary adjacency matrix from the nb object (one-time)
# ──────────────────────────────────────────────────────────────────────
build_adjacency_matrix <- function(nb_obj) {
  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  n <- length(nb_obj)
  # Build COO triplets
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine,

  # but some nb objects store 0L for no-neighbor cells)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

A <- build_adjacency_matrix(rook_neighbors_unique)
# A is 344208 x 344208, ~1.37M non-zero entries — tiny in memory (~20 MB)

# ──────────────────────────────────────────────────────────────────────
# 2. Convert panel to data.table for fast reshaping
# ──────────────────────────────────────────────────────────────────────
dt <- as.data.table(cell_data)

# Ensure a consistent cell ordering that matches the nb object
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
cell_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index
dt[, sp_idx := cell_idx[as.character(id)]]

# Sort for consistent matrix construction
setorder(dt, sp_idx, year)

years <- sort(unique(dt$year))
n_cells <- length(id_order)
n_years <- length(years)
year_idx <- setNames(seq_along(years), as.character(years))

# ──────────────────────────────────────────────────────────────────────
# 3. Neighbor stats via sparse matrix operations
# ──────────────────────────────────────────────────────────────────────
# For neighbor MEAN:  A %*% X  gives sum of neighbor values per cell.
#                     A %*% V  (V = !is.na(X)) gives count of non-NA neighbors.
#                     mean = sum / count
#
# For neighbor MAX and MIN: we iterate over the sparse structure once,
# which is far cheaper than 6.46M R-level list operations.

compute_neighbor_features_sparse <- function(dt, A, var_name,
                                             id_order, years,
                                             cell_idx, year_idx) {
  n_cells <- length(id_order)
  n_years <- length(years)

  # Build cell × year matrix for this variable
  # Use the already-sorted dt (by sp_idx, year)
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(dt$sp_idx, year_idx[as.character(dt$year)])] <- dt[[var_name]]

  # ---- Neighbor MEAN ----
  valid   <- !is.na(X)
  V       <- matrix(as.numeric(valid), nrow = n_cells, ncol = n_years)
  X_zero  <- X
  X_zero[!valid] <- 0

  neighbor_sum   <- as.matrix(A %*% X_zero)   # n_cells x n_years

  neighbor_count <- as.matrix(A %*% V)
  neighbor_mean  <- neighbor_sum / neighbor_count  # NaN where count==0
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # ---- Neighbor MAX and MIN via sparse row iteration ----
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Extract CSR-like structure from the dgCMatrix (which is CSC)
  # It's easier to work with the row-oriented view.
  # Convert to dgRMatrix or iterate columns of t(A).
  # Efficient approach: use the slot structure of dgCMatrix (CSC of A).
  # In CSC, column j has entries in rows A@i[  (A@p[j]+1) : A@p[j+1]  ]
  # Since A is the adjacency matrix, column j lists all cells i that have

  # j as a neighbor. But we want: for each row i, the set of columns j
  # that are neighbors. That's the same as iterating rows.
  # Transpose to make rows become columns:
  At <- t(A)  # now column i of At = row i of A = neighbors of cell i

  p <- At@p
  row_indices <- At@i + 1L  # 1-based

  for (i in seq_len(n_cells)) {
    start <- p[i] + 1L
    end   <- p[i + 1L]
    if (end < start) next  # no neighbors
    nbrs <- row_indices[start:end]
    nb_vals <- X[nbrs, , drop = FALSE]  # k x n_years matrix
    # Column-wise max/min ignoring NA
    for (t in seq_len(n_years)) {
      v <- nb_vals[, t]
      v <- v[!is.na(v)]
      if (length(v) > 0L) {
        neighbor_max[i, t] <- max(v)
        neighbor_min[i, t] <- min(v)
      }
    }
  }

  # ---- Return as vectors aligned to dt ----
  idx <- cbind(dt$sp_idx, year_idx[as.character(dt$year)])
  list(
    n_max  = neighbor_max[idx],
    n_min  = neighbor_min[idx],
    n_mean = neighbor_mean[idx]
  )
}

# ──────────────────────────────────────────────────────────────────────
# 4. Even faster MAX/MIN: vectorized C++ via Rcpp (optional but
#    reduces the max/min loop from ~minutes to seconds)
# ──────────────────────────────────────────────────────────────────────
# If Rcpp is available, replace the double for-loop above:

if (requireNamespace("Rcpp", quietly = TRUE)) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(IntegerVector p, IntegerVector i,
                            NumericMatrix X) {
  int n = X.nrow();
  int T = X.ncol();
  NumericMatrix nmax(n, T);
  NumericMatrix nmin(n, T);
  std::fill(nmax.begin(), nmax.end(), NA_REAL);
  std::fill(nmin.begin(), nmin.end(), NA_REAL);

  for (int cell = 0; cell < n; cell++) {
    int start = p[cell];
    int end   = p[cell + 1];
    if (start == end) continue;
    for (int t = 0; t < T; t++) {
      double mx = R_NegInf;
      double mn = R_PosInf;
      bool found = false;
      for (int k = start; k < end; k++) {
        double v = X(i[k], t);
        if (!R_IsNA(v)) {
          if (v > mx) mx = v;
          if (v < mn) mn = v;
          found = true;
        }
      }
      if (found) {
        nmax(cell, t) = mx;
        nmin(cell, t) = mn;
      }
    }
  }
  return List::create(Named("nmax") = nmax, Named("nmin") = nmin);
}
')

  compute_neighbor_features_fast <- function(dt, A, var_name,
                                              id_order, years,
                                              cell_idx, year_idx) {
    n_cells <- length(id_order)
    n_years <- length(years)

    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$sp_idx, year_idx[as.character(dt$year)])] <- dt[[var_name]]

    # Mean via sparse matmul
    valid  <- !is.na(X)
    V      <- matrix(as.numeric(valid), nrow = n_cells, ncol = n_years)
    X_zero <- X; X_zero[!valid] <- 0
    neighbor_sum   <- as.matrix(A %*% X_zero)
    neighbor_count <- as.matrix(A %*% V)
    neighbor_mean  <- neighbor_sum / neighbor_count
    neighbor_mean[neighbor_count == 0] <- NA_real_

    # Max/Min via Rcpp
    At <- t(A)
    mm <- sparse_neighbor_maxmin(At@p, At@i, X)

    idx <- cbind(dt$sp_idx, year_idx[as.character(dt$year)])
    list(
      n_max  = mm$nmax[idx],
      n_min  = mm$nmin[idx],
      n_mean = neighbor_mean[idx]
    )
  }
}

# ──────────────────────────────────────────────────────────────────────
# 5. Main loop: compute and attach all neighbor features
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_fn <- if (exists("compute_neighbor_features_fast")) {
  compute_neighbor_features_fast
} else {
  compute_neighbor_features_sparse
}

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  res <- compute_fn(dt, A, var_name, id_order, years, cell_idx, year_idx)

  # Use the same column naming convention as the original pipeline
  set(dt, j = paste0(var_name, "_n_max"),  value = res$n_max)
  set(dt, j = paste0(var_name, "_n_min"),  value = res$n_min)
  set(dt, j = paste0(var_name, "_n_mean"), value = res$n_mean)
}

# ──────────────────────────────────────────────────────────────────────
# 6. Convert back to data.frame if needed downstream
# ──────────────────────────────────────────────────────────────────────
dt[, sp_idx := NULL]  # remove helper column
cell_data <- as.data.frame(dt)

# The trained Random Forest model is untouched — only the feature
# engineering (neighbor stats) was accelerated. The numerical values
# are identical to the original implementation.
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | 6.46M-element list built via string paste + named-vector probe | 344K × 344K sparse matrix (one-time, ~20 MB) |
| **Neighbor mean** | Per-row `lapply` with subsetting | Sparse matrix × dense matrix (`A %*% X`), fully vectorized BLAS |
| **Neighbor max/min** | Per-row `lapply` (6.46M R-level calls) | Single pass over sparse structure via Rcpp, or pure-R column-wise loop over 344K cells |
| **Memory** | ~6.46M-element list of integer vectors + 6.46M-entry named character vector | ~77 MB matrix (344K × 28 × 8 bytes) + 20 MB sparse matrix |
| **Estimated time** | 86+ hours | **~2–5 minutes** (with Rcpp), ~15–30 min (pure R) |
| **Numerical result** | — | **Identical** (same max, min, mean over same neighbor sets) |
| **Trained RF model** | — | **Preserved** (no retraining) |