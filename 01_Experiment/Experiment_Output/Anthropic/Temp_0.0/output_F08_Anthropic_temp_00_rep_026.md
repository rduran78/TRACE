 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this identically for each of the 28 years a cell appears. This means:

1. **Redundant topology work ×28:** The neighbor graph has ~344K cells and ~1.37M directed edges. The lookup rebuilds this 28 times (once per year-copy of each cell), yielding ~38.4M string-paste + hash-lookup operations instead of ~1.37M.
2. **String-key indirection is slow:** Using `paste(id, year)` as a hash key and `match`/`[` on named vectors is far slower than integer indexing.
3. **`compute_neighbor_stats` iterates 6.46M list entries:** Each call to `lapply` over 6.46M elements with R-level anonymous functions is extremely slow, and this is repeated 5 times (once per variable).
4. **No vectorization:** The stats (max, min, mean) are computed one row at a time in pure R rather than via vectorized/matrix operations.

**Net effect:** ~86+ hours on a 16 GB laptop.

## Optimization Strategy

**Key insight:** Separate the *static topology* (which cells are neighbors of which) from the *dynamic values* (which change by year).

1. **Build the neighbor graph once at the cell level (344K cells, not 6.46M rows).** Store it as a sparse adjacency structure using integer cell indices (not string keys).

2. **For each variable, reshape values into a matrix of cells × years.** This allows extracting all neighbor values via integer row-indexing into a matrix column (one column per year), fully vectorized.

3. **Compute neighbor stats using sparse matrix multiplication** (via the `Matrix` package) or, equivalently, using a CSR-style adjacency to do vectorized grouped operations. Specifically:
   - Construct a sparse binary adjacency matrix **W** of dimension 344,208 × 344,208 (only ~1.37M non-zeros — tiny in memory).
   - For each year and each variable, the column of values `v` allows: `neighbor_sum = W %*% v`, `neighbor_count = W %*% (!is.na(v))`, `neighbor_mean = neighbor_sum / neighbor_count`.
   - For max and min, use a sparse-matrix trick: iterate over the adjacency list but in compiled C++ via `Matrix` internals, or use a small Rcpp snippet / vectorized R approach with the CSR representation.

4. **Flatten back** to the original cell-year data frame and attach the 15 new columns (5 vars × 3 stats).

5. **Feed into the pre-trained Random Forest** exactly as before — column names and numerical values are preserved.

**Expected speedup:** From ~86 hours to **minutes**. The sparse matrix–vector product for mean is O(nnz) ≈ 1.37M per year per variable = 1.37M × 28 × 5 ≈ 192M operations, trivially fast. Max/min via the CSR loop is the same order.

## Working R Code

```r
library(Matrix)
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 0.  Inputs assumed to exist:
#       cell_data            : data.frame/data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, …
#       id_order             : integer/character vector of unique cell IDs (length 344,208)
#       rook_neighbors_unique: spdep nb object (list of length 344,208, integer neighbor indices)
#       rf_model             : pre-trained Random Forest model object
#       neighbor_source_vars : c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table for speed (non-destructive if already data.table)
cell_dt <- as.data.table(cell_data)

n_cells <- length(id_order)
years   <- sort(unique(cell_dt$year))
n_years <- length(years)

# ──────────────────────────────────────────────────────────────────────
# 1.  Build the STATIC sparse binary adjacency matrix  (done ONCE)
# ──────────────────────────────────────────────────────────────────────
message("Building sparse adjacency matrix …")

# CSR-style vectors
adj_i <- integer(0)
adj_j <- integer(0)

for (k in seq_len(n_cells)) {
  nb <- rook_neighbors_unique[[k]]
  if (length(nb) > 0L && !(length(nb) == 1L && nb[1] == 0L)) {
    adj_i <- c(adj_i, rep.int(k, length(nb)))
    adj_j <- c(adj_j, nb)
  }
}

W <- sparseMatrix(i = adj_i, j = adj_j, x = 1,
                  dims = c(n_cells, n_cells), giveCsparse = TRUE)

# Also store adjacency as a list of integer vectors for max/min
# (reuse rook_neighbors_unique directly — it already is this)
adj_list <- rook_neighbors_unique

rm(adj_i, adj_j)

# ──────────────────────────────────────────────────────────────────────
# 2.  Create a cell-index column in cell_dt for fast matrix mapping
# ──────────────────────────────────────────────────────────────────────
# Map each cell id to its position in id_order (1-based integer index)
id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))
cell_dt[, cell_idx := id_to_cellidx[as.character(id)]]

# Ensure rows are sorted by (year, cell_idx) for predictable matrix fill
setkey(cell_dt, year, cell_idx)

# ──────────────────────────────────────────────────────────────────────
# 3.  Helper: build a cells × years matrix from a variable column
# ──────────────────────────────────────────────────────────────────────
build_cell_year_matrix <- function(dt, var_name, n_cells, years) {
  # Pre-allocate matrix (NA by default handles any missing cell-years)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  colnames(mat) <- as.character(years)
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    sub <- dt[year == yr, .(cell_idx, val = get(var_name))]
    mat[sub$cell_idx, yi] <- sub$val
  }
  mat
}

# ──────────────────────────────────────────────────────────────────────
# 4.  Compute neighbor stats for one variable (vectorized over cells)
# ──────────────────────────────────────────────────────────────────────
compute_neighbor_stats_fast <- function(W, adj_list, val_mat, n_cells, n_years) {
  # Output matrices: cells × years
  nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (yi in seq_len(n_years)) {
    v <- val_mat[, yi]                       # length n_cells
    not_na <- !is.na(v)
    v_zero <- v
    v_zero[is.na(v_zero)] <- 0               # for sparse multiply (NAs → 0)
    
    # --- Neighbor mean via sparse matrix multiply ---
    nb_sum   <- as.numeric(W %*% v_zero)     # sum of neighbor values (NA→0)
    nb_count <- as.numeric(W %*% as.numeric(not_na))  # count of non-NA neighbors
    
    mean_vec <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
    nb_mean[, yi] <- mean_vec
    
    # --- Neighbor max and min via adjacency list ---
    # Vectorized as much as possible; this loop is over 344K cells (fast)
    max_vec <- rep(NA_real_, n_cells)
    min_vec <- rep(NA_real_, n_cells)
    
    for (k in seq_len(n_cells)) {
      nb <- adj_list[[k]]
      if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) next
      nb_vals <- v[nb]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      max_vec[k] <- max(nb_vals)
      min_vec[k] <- min(nb_vals)
    }
    
    nb_max[, yi] <- max_vec
    nb_min[, yi] <- min_vec
    
    if (yi %% 5 == 0) message("    year ", yi, "/", n_years, " done")
  }
  
  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# ──────────────────────────────────────────────────────────────────────
# 5.  Main loop: iterate over the 5 variables (not 6.46M rows)
# ──────────────────────────────────────────────────────────────────────
for (var_name in neighbor_source_vars) {
  message("Processing neighbor stats for: ", var_name)
  
  # 5a. Reshape variable to cells × years matrix
  val_mat <- build_cell_year_matrix(cell_dt, var_name, n_cells, years)
  
  # 5b. Compute stats (vectorized / sparse)
  stats <- compute_neighbor_stats_fast(W, adj_list, val_mat, n_cells, n_years)
  
  # 5c. Map results back to cell_dt rows
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  
  # Flatten matrices back to the row order of cell_dt (keyed by year, cell_idx)
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    rows <- cell_dt[year == yr, which = TRUE]
    cidx <- cell_dt$cell_idx[rows]
    
    set(cell_dt, i = rows, j = col_max,  value = stats$nb_max [cidx, yi])
    set(cell_dt, i = rows, j = col_min,  value = stats$nb_min [cidx, yi])
    set(cell_dt, i = rows, j = col_mean, value = stats$nb_mean[cidx, yi])
  }
  
  rm(val_mat, stats)
  gc()
  message("  ✓ ", var_name, " complete.")
}

# Clean up helper column
cell_dt[, cell_idx := NULL]

# ──────────────────────────────────────────────────────────────────────
# 6.  Predict with the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
cell_dt[, prediction := predict(rf_model, newdata = cell_dt)]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)
```

---

### Optional: Replace the inner max/min R loop with Rcpp for further speedup

The per-year R loop over 344K cells for max/min is the remaining bottleneck (~5–10 min total). If desired, this Rcpp function eliminates it:

```cpp
// file: neighbor_maxmin.cpp
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_maxmin(List adj_list, NumericVector v) {
  int n = adj_list.size();
  NumericMatrix out(n, 2);  // col 0 = max, col 1 = min
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    IntegerVector nb = adj_list[i];
    if (nb.size() == 0 || (nb.size() == 1 && nb[0] == 0)) continue;
    double mx = R_NegInf, mn = R_PosInf;
    bool found = false;
    for (int j = 0; j < nb.size(); j++) {
      double val = v[nb[j] - 1];  // R is 1-indexed
      if (NumericVector::is_na(val)) continue;
      found = true;
      if (val > mx) mx = val;
      if (val < mn) mn = val;
    }
    if (found) {
      out(i, 0) = mx;
      out(i, 1) = mn;
    }
  }
  return out;
}
```

```r
Rcpp::sourceCpp("neighbor_maxmin.cpp")

# Replace the inner for-k loop in step 4 with:
mm <- neighbor_maxmin(adj_list, v)   # returns n_cells × 2 matrix
nb_max[, yi] <- mm[, 1]
nb_min[, yi] <- mm[, 2]
```

---

### Summary of improvements

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (built once) |
| Key mechanism | String paste + named-vector hash | Integer index + sparse matrix |
| Mean computation | R `lapply` over 6.46M entries | Sparse matrix multiply (`W %*% v`) |
| Max/Min computation | R `lapply` over 6.46M entries | R loop over 344K cells (or Rcpp) |
| Times repeated | 5 vars × 6.46M = 32.3M R calls | 5 vars × 28 years × 344K cells |
| Estimated runtime | 86+ hours | **~5–15 minutes** (pure R) / **~2–5 min** (with Rcpp) |
| Numerical results | — | **Identical** (same max, min, mean) |
| RF model | Unchanged | Unchanged |
| Peak RAM | Lookup list ~6.46M entries | Sparse matrix ~22 MB + one 344K×28 matrix ~77 MB |