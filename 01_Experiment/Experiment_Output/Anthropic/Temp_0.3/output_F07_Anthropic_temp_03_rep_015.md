 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by pasting strings, looking up named-vector keys, and filtering NAs — all inside an `lapply` over every row. This is **O(N × k)** with enormous per-element overhead from:

1. **Character key construction and named-vector lookup** (`paste` + indexing into `idx_lookup`) for every single row — 6.46M calls.
2. **`lapply` over 6.46M rows** in pure R (no vectorization).
3. **`compute_neighbor_stats`** then does another `lapply` over 6.46M elements, extracting subsets of a numeric vector — tolerable but still slow.
4. The combination, repeated for 5 variables, yields the estimated 86+ hour runtime.

The fundamental issue: the code treats an **inherently vectorizable sparse-matrix operation** as a row-by-row scripting problem.

---

## Optimization Strategy

**Replace the entire lookup + stats pipeline with a sparse adjacency matrix multiply.**

Key insight: neighbor max, min, and mean over a spatial weights structure are expressible as operations on a **sparse row-standardized (or binary) adjacency matrix `W`** of dimension `N_cells × N_cells`. For each year `t`, the neighbor mean of variable `x` is simply `W %*% x_t`. Neighbor max and min require a slightly different approach (iterating over sparse entries), but the `Matrix` package makes this efficient.

### Concrete steps:

1. **Convert `rook_neighbors_unique` (spdep nb) → sparse matrix `W`** once. This is a 344,208 × 344,208 sparse matrix with ~1.37M non-zero entries — trivially small in memory (~20 MB).
2. **For each year, extract the column vector, compute `W %*% x`** for the mean (then divide by neighbor count), and use grouped sparse-row operations for max/min.
3. This replaces 6.46M R-level iterations with ~28 sparse matrix operations per variable — **seconds instead of days**.

We avoid retraining the Random Forest; we only reproduce the exact same 15 derived columns (`{var}_{max,min,mean}` for 5 variables) with identical numerical values.

---

## Working R Code

```r
library(Matrix)
library(spdep)
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build sparse binary adjacency matrix from spdep nb object (once)
# ──────────────────────────────────────────────────────────────────────
build_sparse_W <- function(nb_obj, n) {
  # nb_obj: spdep nb list (length n), each element is integer vector of neighbor indices
  # Returns: n x n sparse binary matrix (dgCMatrix)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove zero-neighbor placeholders (spdep uses 0L for no-neighbor entries)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(rook_neighbors_unique)  # 344,208
W <- build_sparse_W(rook_neighbors_unique, n_cells)

# Precompute number of neighbors per cell (used for mean)
# This is the row sum of W (constant across years)
n_neighbors <- as.numeric(rowSums(W))  # length = n_cells

# ──────────────────────────────────────────────────────────────────────
# 2. Convert cell_data to data.table for fast grouped operations
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)

# id_order maps position index (1..n_cells) <-> cell id
# We need a map from cell id -> matrix row index
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Add matrix row index to data
cell_dt[, mat_row := id_to_row[as.character(id)]]

# Sort for efficient year-wise extraction
setkey(cell_dt, year, mat_row)

# ──────────────────────────────────────────────────────────────────────
# 3. Efficient neighbor max / min via sparse matrix structure
# ──────────────────────────────────────────────────────────────────────
# For max and min we cannot use matrix multiply directly.
# Strategy: use the CSC (compressed sparse column) representation of W.
# W is stored as dgCMatrix. We iterate over rows using the transpose.
#
# But even better: use a Rcpp-free pure-R approach that is still fast.
# For each year we build a full vector x (length n_cells, NA-safe),
# then compute neighbor stats using W's sparse structure.

# Pre-extract W structure for row-wise traversal
# Convert to dgRMatrix (compressed sparse row) for efficient row access
Wr <- as(W, "RsparseMatrix")  # dgRMatrix

compute_neighbor_stats_sparse <- function(x_vec, Wr_p, Wr_j, n) {
  # x_vec: numeric vector length n (values for one year, ordered by mat_row)
  # Wr_p:  row pointers (0-based, length n+1)
  # Wr_j:  column indices (0-based)
  # Returns: matrix n x 3 (max, min, mean)
  
  n_max  <- numeric(n)
  n_min  <- numeric(n)
  n_mean <- numeric(n)
  
  for (i in seq_len(n)) {
    start <- Wr_p[i] + 1L      # convert 0-based to 1-based
    end   <- Wr_p[i + 1L]
    if (end < start) {
      n_max[i]  <- NA_real_
      n_min[i]  <- NA_real_
      n_mean[i] <- NA_real_
      next
    }
    cols <- Wr_j[start:end] + 1L  # 0-based to 1-based
    vals <- x_vec[cols]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      n_max[i]  <- NA_real_
      n_min[i]  <- NA_real_
      n_mean[i] <- NA_real_
    } else {
      n_max[i]  <- max(vals)
      n_min[i]  <- min(vals)
      n_mean[i] <- mean(vals)
    }
  }
  list(n_max = n_max, n_min = n_min, n_mean = n_mean)
}

# ──────────────────────────────────────────────────────────────────────
# 3b. MUCH faster: vectorized approach using Matrix ops
#     - mean:  (W %*% x) / n_neighbors  (exact, handles NA via replacement)
#     - max/min: use row-wise ops on a modified sparse matrix
# ──────────────────────────────────────────────────────────────────────
# For max/min we use a trick: create a sparse matrix where non-zero
# entries store the neighbor values, then compute row max/min.
# This avoids the R-level for loop entirely.

compute_year_neighbor_stats <- function(x_vec, W, Wr, n_neighbors, n) {
  # Handle NAs: replace with NA-safe sentinel for sparse ops
  x_safe <- x_vec
  has_na <- is.na(x_safe)
  
  # --- MEAN (exact, matching original) ---
  # Replace NA with 0 for multiplication, then adjust count
  x_for_sum <- x_safe
  x_for_sum[has_na] <- 0
  
  w_sum <- as.numeric(W %*% x_for_sum)  # sum of non-NA neighbor values
  
  # Count of non-NA neighbors per cell
  not_na_indicator <- as.numeric(!has_na)
  w_count <- as.numeric(W %*% not_na_indicator)
  
  n_mean <- ifelse(w_count > 0, w_sum / w_count, NA_real_)
  
  # --- MAX and MIN via sparse value matrix ---
  # Build a sparse matrix V where V[i,j] = x[j] for each neighbor j of i
  # Then row-max of V = neighbor max, row-min = neighbor min
  
  # Extract triplet form from Wr (row-sparse)
  Wt <- as(Wr, "TsparseMatrix")  # dgTMatrix: i, j, x (0-based)
  
  # Neighbor values
  neighbor_vals <- x_vec[Wt@j + 1L]
  
  # Filter out NA neighbor values
  valid <- !is.na(neighbor_vals)
  vi <- Wt@i[valid]
  vj <- Wt@j[valid]
  vx <- neighbor_vals[valid]
  
  # For max: we need row-wise max of sparse entries
  # Use data.table for speed
  dt <- data.table(row = vi + 1L, val = vx)
  
  max_dt <- dt[, .(nmax = max(val)), by = row]
  min_dt <- dt[, .(nmin = min(val)), by = row]
  
  n_max <- rep(NA_real_, n)
  n_min <- rep(NA_real_, n)
  n_max[max_dt$row] <- max_dt$nmax
  n_min[min_dt$row] <- min_dt$nmin
  
  list(n_max = n_max, n_min = n_min, n_mean = n_mean)
}

# ──────────────────────────────────────────────────────────────────────
# 4. Main loop: per variable, per year
# ──────────────────────────────────────────────────────────────────────
years <- sort(unique(cell_dt$year))

# Pre-convert Wr to TsparseMatrix once (reused every call)
Wr_T <- as(Wr, "TsparseMatrix")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  max_col  <- paste0(var_name, "_max")
  min_col  <- paste0(var_name, "_min")
  mean_col <- paste0(var_name, "_mean")
  
  # Pre-allocate result columns
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]
  
  for (yr in years) {
    # Extract rows for this year (already keyed by year, mat_row)
    yr_idx <- cell_dt[.(yr), which = TRUE]
    yr_sub <- cell_dt[yr_idx]
    
    # Build full-length vector (some cells may be missing in a year)
    x_full <- rep(NA_real_, n_cells)
    x_full[yr_sub$mat_row] <- yr_sub[[var_name]]
    
    # Compute stats
    stats <- compute_year_neighbor_stats(x_full, W, Wr_T, n_neighbors, n_cells)
    
    # Write back only for cells present this year
    rows_in_mat <- yr_sub$mat_row
    set(cell_dt, i = yr_idx, j = max_col,  value = stats$n_max[rows_in_mat])
    set(cell_dt, i = yr_idx, j = min_col,  value = stats$n_min[rows_in_mat])
    set(cell_dt, i = yr_idx, j = mean_col, value = stats$n_mean[rows_in_mat])
  }
  
  message("Done: ", var_name)
}

# ──────────────────────────────────────────────────────────────────────
# 5. Convert back to data.frame if needed downstream
# ──────────────────────────────────────────────────────────────────────
cell_data <- as.data.frame(cell_dt[, mat_row := NULL])
```

---

## Performance Analysis

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M `paste` + named lookups → hours | One-time sparse matrix build → **< 1 sec** |
| Stats per variable | `lapply` over 6.46M rows × 5 vars | 28 sparse matrix multiplies + `data.table` group-by per var | 
| Total estimated time | **86+ hours** | **~5–15 minutes** |
| Peak RAM | ~6.46M-element list of integer vectors | ~344K × 344K sparse matrix (~20 MB) + data.table (~2 GB) |

## Why Numerical Equivalence Is Preserved

- **Mean**: `(W %*% x) / count_non_na_neighbors` is algebraically identical to `mean(neighbor_vals[!is.na()])` — the sum and count are computed exactly via sparse matrix–vector products.
- **Max/Min**: We extract the exact same `(i, j)` pairs from the adjacency matrix and group-aggregate with `max`/`min` — identical to indexing `vals[idx]` and calling `max`/`min`.
- **No retraining needed**: The 15 derived columns are numerically identical; the trained Random Forest model is untouched.