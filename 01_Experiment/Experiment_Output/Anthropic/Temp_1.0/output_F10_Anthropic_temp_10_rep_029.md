 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical performance bottlenecks:

### 1. **O(N²)-scale neighbor lookup construction (~6.46M list elements)**
`build_neighbor_lookup` creates a list of length equal to the number of *rows* (cell-years ≈ 6.46M). For each row, it performs string-pasting, named-vector lookups via character keys, and `NA` filtering. Named vector lookups in R are hash-table scans that degrade with size. With ~6.46M keys, each lookup is expensive, and doing it ~6.46M times is catastrophic.

### 2. **Redundant topology recomputation across years**
The rook-neighbor graph is purely spatial — it does not change across years. Yet the lookup embeds year into keys, effectively rebuilding the same adjacency structure 28 times (once per year). The ~1.37M directed edges × 28 years = ~38.5M edge lookups, all done via slow string hashing.

### 3. **Row-wise `lapply` in `compute_neighbor_stats`**
For each of ~6.46M rows, R calls an anonymous function, subsets a vector by index, removes NAs, and computes three summary statistics. The per-call overhead of R function dispatch, memory allocation, and GC pressure over 6.46M iterations (×5 variables = ~32.3M calls) dominates runtime.

**Summary:** The 86+ hour runtime is caused by (a) string-key hashing at scale, (b) redundant year-wise topology reconstruction, and (c) millions of interpreted R function calls where vectorized or compiled operations should be used.

---

## Optimization Strategy

1. **Build the spatial adjacency graph once as a sparse matrix.** Convert the `spdep::nb` object to a sparse `dgCMatrix` (from the `Matrix` package). This is a one-time O(E) operation over ~1.37M edges. The sparse matrix natively supports fast row-wise neighbor extraction.

2. **Process year-by-year using matrix slicing.** For each of the 28 years, extract the submatrix of cells for that year. Since all ~344K cells appear in each year (balanced panel), we can directly map cell indices to matrix rows. The sparse adjacency matrix rows give neighbor indices in O(degree) time.

3. **Vectorized aggregation via sparse matrix–vector multiplication.** 
   - **Mean:** `(A %*% x) / (A %*% ones)` — sparse matrix–vector multiply gives the sum of neighbor values; dividing by neighbor count gives the mean.
   - **Max and Min:** Use compiled row-wise operations. The `{slam}` package or a small Rcpp function can compute row-wise max/min over sparse structures. Alternatively, iterate over cells using the CSC/CSR structure directly, which is far faster than `lapply` with R-level indexing.

4. **Avoid all string operations.** Use integer indexing exclusively. Map cell IDs to integer positions once.

5. **Memory efficiency.** A sparse matrix with ~1.37M entries uses ~16 MB. Processing one year at a time means holding ~344K × (number of variables) in memory — trivially fits in 16 GB.

6. **Preserve numerical equivalence.** The sparse matrix multiply for mean is algebraically identical to `mean(neighbor_vals[!is.na(vals)])` when NAs are handled correctly. Max and min are computed identically. The trained Random Forest model is loaded and used as-is.

---

## Optimized R Code

```r
# ==============================================================================
# Optimized Spatial Neighbor Feature Engineering Pipeline
# ==============================================================================
# Preserves numerical equivalence with the original compute_neighbor_stats.
# Preserves the pre-trained Random Forest model (no retraining).
# ==============================================================================

library(Matrix)   # sparse matrices
library(data.table)  # fast data manipulation

# --------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix from spdep::nb object (ONE TIME)
# --------------------------------------------------------------------------
# rook_neighbors_unique : spdep nb object (list of integer vectors)
# id_order              : vector of cell IDs in the order matching the nb object
#
# The adjacency matrix A is N_cells x N_cells where A[i,j] = 1 means
# cell j is a rook neighbor of cell i.

build_sparse_adjacency <- function(nb_obj) {
  n <- length(nb_obj)
  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    from <- c(from, rep.int(i, length(nbrs)))
    to   <- c(to, nbrs)
  }
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

# Pre-allocate more efficiently for large nb objects:
build_sparse_adjacency_fast <- function(nb_obj) {
  n <- length(nb_obj)
  lens <- vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1))
  total_edges <- sum(lens)
  
  from <- integer(total_edges)
  to   <- integer(total_edges)
  pos  <- 1L
  for (i in seq_len(n)) {
    k <- lens[i]
    if (k == 0L) next
    idx <- pos:(pos + k - 1L)
    from[idx] <- i
    to[idx]   <- nb_obj[[i]]
    pos <- pos + k
  }
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency_fast(rook_neighbors_unique)
n_cells <- nrow(A)
cat(sprintf("Adjacency matrix: %d x %d, %d edges\n", n_cells, n_cells, nnzero(A)))

# Precompute neighbor counts per cell (used for mean calculation)
# A %*% ones = number of neighbors per cell
neighbor_counts <- as.numeric(A %*% rep(1, n_cells))

# --------------------------------------------------------------------------
# STEP 2: Convert cell_data to data.table, create integer cell index
# --------------------------------------------------------------------------
dt <- as.data.table(cell_data)

# Map cell IDs to the integer positions in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
dt[, cell_idx := id_to_pos[as.character(id)]]

# Ensure sorted by year then cell_idx for fast slicing
setkey(dt, year, cell_idx)

# Get sorted unique years
years <- sort(unique(dt$year))
cat(sprintf("Panel: %d cells x %d years = %d rows\n", n_cells, length(years), nrow(dt)))

# --------------------------------------------------------------------------
# STEP 3: Vectorized neighbor stat computation using sparse matrix ops
# --------------------------------------------------------------------------
# For each variable and year:
#   - Extract the value vector x (length n_cells, ordered by cell_idx)
#   - Handle NAs: create a valid-indicator vector v (1 if not NA, 0 if NA)
#   - neighbor_sum   = A %*% x_clean        (where x_clean has NA replaced by 0)
#   - neighbor_count = A %*% v              (count of non-NA neighbors)
#   - neighbor_mean  = neighbor_sum / neighbor_count  (NA where count == 0)
#
# For max and min, we use the CSC structure of A directly.

# --- Helper: row-wise max and min over sparse A applied to a value vector ---
# This iterates over columns of A (CSC format) and updates row-wise max/min.
# Much faster than R-level lapply because it uses vectorized updates.

compute_sparse_row_max_min <- function(A, x) {
  # A is dgCMatrix (CSC). We need row-wise operations.
  # Convert to dgRMatrix (CSR) for efficient row access, or use column iteration.
  # Strategy: iterate over CSC columns, which gives us (row, col) pairs.
  
  n <- nrow(A)
  row_max <- rep(-Inf, n)
  row_min <- rep(Inf, n)
  row_has_valid <- logical(n)  # tracks if any valid neighbor value seen
  
  # CSC storage: A@p (column pointers), A@i (row indices, 0-based), A@x (values)
  p <- A@p
  ri <- A@i  # 0-based row indices
  
  for (j in seq_len(ncol(A))) {
    idx_range <- (p[j] + 1L):p[j + 1L]
    if (length(idx_range) == 0L || p[j] == p[j + 1L]) next
    
    val_j <- x[j]
    if (is.na(val_j)) next
    
    rows <- ri[idx_range] + 1L  # convert to 1-based
    row_has_valid[rows] <- TRUE
    row_max[rows] <- pmax(row_max[rows], val_j)
    row_min[rows] <- pmin(row_min[rows], val_j)
  }
  
  row_max[!row_has_valid] <- NA_real_
  row_min[!row_has_valid] <- NA_real_
  
  list(max = row_max, min = row_min)
}

# --------------------------------------------------------------------------
# STEP 4: Main loop — process each variable, vectorized by year
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]
}

cat("Computing neighbor features...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Variable: %s\n", var_name))
  
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  for (yr in years) {
    # Extract rows for this year (dt is keyed by year, cell_idx)
    yr_rows <- dt[.(yr), which = TRUE]
    
    # Build the full-length value vector aligned to cell_idx
    # dt is keyed by (year, cell_idx), so within a year slice,
    # cell_idx is sorted. We need a vector of length n_cells.
    yr_cell_idx <- dt$cell_idx[yr_rows]
    yr_vals_raw <- dt[[var_name]][yr_rows]
    
    # Create a full vector for all cells (some cells might be missing for a year)
    x <- rep(NA_real_, n_cells)
    x[yr_cell_idx] <- yr_vals_raw
    
    # --- Mean via sparse matrix-vector multiply ---
    x_clean <- x
    x_clean[is.na(x_clean)] <- 0
    v <- as.numeric(!is.na(x))
    
    neighbor_sum   <- as.numeric(A %*% x_clean)
    neighbor_nvalid <- as.numeric(A %*% v)
    
    n_mean <- ifelse(neighbor_nvalid > 0, neighbor_sum / neighbor_nvalid, NA_real_)
    
    # --- Max and Min via CSC iteration ---
    maxmin <- compute_sparse_row_max_min(A, x)
    
    # Write results back — only for cells present this year
    set(dt, i = yr_rows, j = col_max,  value = maxmin$max[yr_cell_idx])
    set(dt, i = yr_rows, j = col_min,  value = maxmin$min[yr_cell_idx])
    set(dt, i = yr_rows, j = col_mean, value = n_mean[yr_cell_idx])
  }
}

elapsed <- proc.time() - t0
cat(sprintf("Neighbor features computed in %.1f seconds\n", elapsed[3]))

# --------------------------------------------------------------------------
# STEP 5: Convert back and predict with the pre-trained Random Forest
# --------------------------------------------------------------------------
cell_data <- as.data.frame(dt)

# The Random Forest model (rf_model) is already loaded; do NOT retrain.
# Predict using the updated cell_data with new neighbor features.
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Rcpp for Max/Min (Optional, Recommended)

The `compute_sparse_row_max_min` R function above iterates column-by-column with vectorized `pmax`/`pmin`, which is good but still involves R-level looping over `ncol(A)` = 344K columns. If this remains a bottleneck, the following Rcpp drop-in replacement eliminates all R overhead:

```r
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_row_max_min_cpp(IntegerVector Ap, IntegerVector Ai,
                            NumericVector x, int nrow) {
  NumericVector rmax(nrow, R_NegInf);
  NumericVector rmin(nrow, R_PosInf);
  LogicalVector has_valid(nrow, false);
  int ncol = Ap.size() - 1;

  for (int j = 0; j < ncol; j++) {
    double val = x[j];
    if (NumericVector::is_na(val)) continue;
    for (int idx = Ap[j]; idx < Ap[j+1]; idx++) {
      int row = Ai[idx];  // 0-based
      has_valid[row] = true;
      if (val > rmax[row]) rmax[row] = val;
      if (val < rmin[row]) rmin[row] = val;
    }
  }
  for (int i = 0; i < nrow; i++) {
    if (!has_valid[i]) {
      rmax[i] = NA_REAL;
      rmin[i] = NA_REAL;
    }
  }
  return List::create(Named("max") = rmax, Named("min") = rmin);
}
')

# Drop-in replacement in the inner loop:
# maxmin <- sparse_row_max_min_cpp(A@p, A@i, x, n_cells)
```

---

## Expected Performance

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (string hashing 6.46M keys) | ~2 seconds (sparse matrix from nb) |
| Stats per variable-year | ~10 min (lapply over 344K rows) | ~0.3 sec (SpMV + CSC scan) |
| Total stats (5 vars × 28 yrs) | ~86+ hours | **~1–3 minutes** (pure R) / **~30 sec** (with Rcpp) |
| Memory | ~10+ GB (list of 6.46M vectors + string keys) | ~200 MB (sparse matrix + data.table) |

**Speedup: ~2,000–5,000×**, well within a 16 GB laptop's capabilities.

### Why Numerical Equivalence Is Preserved

- **Mean**: `(A %*% x_clean) / (A %*% v)` computes exactly `sum(x[neighbors & !is.na]) / count(!is.na(x[neighbors]))`, identical to the original `mean(neighbor_vals)` after NA removal.
- **Max/Min**: The CSC iteration visits every (row, col) entry in A where `A[row,col]=1`, testing `x[col]` against the running row-wise max/min — algebraically identical to `max(vals[idx])` and `min(vals[idx])` in the original code.
- The Random Forest model object is never modified or retrained.