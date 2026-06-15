 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` call that performs character coercion, string pasting, and named-vector lookups per row.** This is O(n) string operations over ~6.46M rows, each touching variable-length neighbor sets. The subsequent `compute_neighbor_stats` then loops over that 6.46M-element list again per variable, indexing into a numeric vector — which is actually fast, but is called 5 times redundantly.

**Root causes of the ~86-hour runtime:**

1. **String-key lookups in `build_neighbor_lookup`:** `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are extremely slow at scale. For each of the 6.46M rows, you paste neighbor IDs with the year, then look them up in a 6.46M-length named character vector. Named vector lookup in R is O(n) per query in the worst case (hash collisions aside, the overhead of character matching is enormous).

2. **`lapply` over 6.46M elements:** Each closure call has R interpreter overhead. 6.46M iterations × 5 variables = ~32M R-level function calls.

3. **Redundant structure:** The neighbor topology is *time-invariant* — cell A's rook neighbors are the same in every year. Yet the lookup is built at the cell-year level, duplicating the same adjacency information 28 times per cell.

## Optimization Strategy

### Key Insight: Separate spatial topology from temporal indexing

Since rook neighbors don't change over time, we can:

1. **Build a sparse adjacency matrix once** (344,208 × 344,208) from the `nb` object — this is a standard `spdep::nb2listw` / `Matrix::sparseMatrix` operation.
2. **Reshape each variable into a matrix** of dimension (344,208 cells × 28 years).
3. **Compute neighbor max/min/mean via sparse matrix operations** — a single sparse matrix–dense matrix multiply gives neighbor sums; neighbor means follow by dividing by neighbor counts. For max and min, we iterate over the sparse structure but in a vectorized/compiled way.

This replaces 6.46M string lookups and 32M R function calls with a handful of sparse matrix operations that run in compiled C/Fortran code in seconds.

### Estimated speedup

| Step | Before | After |
|---|---|---|
| Build lookup | ~hours (string ops) | ~seconds (sparse matrix) |
| Neighbor mean (per var) | ~hours (lapply) | ~seconds (sparse %*% dense) |
| Neighbor max/min (per var) | ~hours (lapply) | ~1-2 min (vectorized C++ via data.table or Matrix) |
| **Total** | **~86 hours** | **~2-5 minutes** |

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves the original numerical estimand exactly.
# Preserves the trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)
library(data.table)

# ---- Step 0: Ensure cell_data is a data.table for speed --------------------
cell_dt <- as.data.table(cell_data)

# ---- Step 1: Build sparse adjacency matrix from the nb object (once) -------
# id_order is the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique is an nb object (list of integer index vectors)

build_sparse_adjacency <- function(nb_obj, n) {
  # nb_obj[[i]] contains integer indices of neighbors of cell i
  # 0L in nb means no neighbors (spdep convention)
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # Remove the spdep "no neighbor" sentinel (0)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
W <- build_sparse_adjacency(rook_neighbors_unique, n_cells)

# Neighbor count per cell (used for means) — time-invariant
neighbor_counts <- as.numeric(rowSums(W))  # length = n_cells

cat("Adjacency matrix:", nrow(W), "x", ncol(W),
    "with", nnzero(W), "non-zero entries\n")

# ---- Step 2: Map cell IDs to matrix row indices ----------------------------
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# ---- Step 3: Build the cell-year indexing structure -------------------------
# We need a mapping: (cell_row_index, year) -> row in cell_dt
# and the reverse so we can write results back.

cell_dt[, cell_row := id_to_row[as.character(id)]]

years <- sort(unique(cell_dt$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

cell_dt[, year_col := year_to_col[as.character(year)]]

# ---- Step 4: Function to pivot a variable into a (n_cells x n_years) matrix -
pivot_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Initialize with NA
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$cell_row, dt$year_col)] <- dt[[var_name]]
  mat
}

# ---- Step 5: Compute neighbor stats via sparse matrix operations -----------

# For MEAN: W %*% val_matrix gives neighbor sums; divide by neighbor_counts.
# For MAX and MIN: we must iterate over the sparse structure, but we do it
# in a vectorized way using the sparse triplet representation.

compute_neighbor_stats_sparse <- function(W, val_matrix, neighbor_counts) {
  n_cells <- nrow(val_matrix)
  n_years <- ncol(val_matrix)
  
  # --- Neighbor MEAN (and SUM) via sparse matrix multiply ---
  # Replace NA with 0 for summation, but track counts of non-NA neighbors
  val_nona <- val_matrix
  is_valid <- !is.na(val_matrix)  # logical matrix
  val_nona[!is_valid] <- 0
  
  # Neighbor sum of values (ignoring NAs correctly)
  neighbor_sum <- as.matrix(W %*% val_nona)        # n_cells x n_years
  # Neighbor count of non-NA values
  neighbor_nvalid <- as.matrix(W %*% (is_valid * 1))  # n_cells x n_years
  
  neighbor_mean <- neighbor_sum / neighbor_nvalid
  neighbor_mean[neighbor_nvalid == 0] <- NA_real_
  
  # --- Neighbor MAX and MIN via sparse structure iteration ---
  # Extract the sparse structure once
  Wt <- as(W, "TsparseMatrix")  # triplet form: Wt@i (0-based row), Wt@j (0-based col)
  from_idx <- Wt@i + 1L  # 1-based: the cell whose neighbor stats we're computing
  to_idx   <- Wt@j + 1L  # 1-based: the neighbor cell
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Process year by year to keep memory bounded
  for (t in seq_len(n_years)) {
    vals_t <- val_matrix[, t]          # values for this year
    nvals  <- vals_t[to_idx]           # neighbor values along edges
    
    # Use data.table for fast grouped max/min
    edge_dt <- data.table(
      from = from_idx,
      nval = nvals
    )
    # Remove edges where neighbor value is NA
    edge_dt <- edge_dt[!is.na(nval)]
    
    if (nrow(edge_dt) > 0) {
      stats <- edge_dt[, .(nmax = max(nval), nmin = min(nval)), by = from]
      neighbor_max[stats$from, t] <- stats$nmax
      neighbor_min[stats$from, t] <- stats$nmin
    }
  }
  
  list(neighbor_max = neighbor_max,
       neighbor_min = neighbor_min,
       neighbor_mean = neighbor_mean)
}

# ---- Step 6: Run for each source variable and write back to cell_dt --------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-compute the row-back-mapping: for each (cell_row, year_col), 
# which row in cell_dt does it correspond to?
# We'll use this to scatter results back.
back_idx <- cell_dt[, .(dt_row = .I, cell_row, year_col)]

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  
  # Pivot to matrix
  val_mat <- pivot_to_matrix(cell_dt, var_name, n_cells, n_years)
  
  # Compute stats
  stats <- compute_neighbor_stats_sparse(W, val_mat, neighbor_counts)
  
  # Extract results back to cell_dt rows
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  
  # Vectorized scatter using the index mapping
  set(cell_dt, j = max_col,
      value = stats$neighbor_max[cbind(back_idx$cell_row, back_idx$year_col)])
  set(cell_dt, j = min_col,
      value = stats$neighbor_min[cbind(back_idx$cell_row, back_idx$year_col)])
  set(cell_dt, j = mean_col,
      value = stats$neighbor_mean[cbind(back_idx$cell_row, back_idx$year_col)])
  
  cat("  Done:", max_col, min_col, mean_col, "\n")
}

# ---- Step 7: Convert back to data.frame if needed by downstream code ------
cell_dt[, c("cell_row", "year_col") := NULL]
cell_data <- as.data.frame(cell_dt)

cat("All neighbor features computed. Ready for prediction with existing RF model.\n")
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | `W` is built directly from `rook_neighbors_unique`, the identical `nb` object. Every `1` in the sparse matrix corresponds to exactly one directed rook-neighbor edge. |
| **Same max/min/mean** | `max`, `min`, and `mean` are computed over exactly the same sets of neighbor values, with the same NA-handling logic (`na.rm = TRUE` equivalent: NAs are excluded, and if all neighbors are NA, the result is NA). |
| **Same column names** | Output columns follow the same `{var}_neighbor_max/min/mean` naming convention. |
| **RF model untouched** | No model object is modified. Only the feature columns in `cell_data` are (re)computed identically. |

## Memory Profile

| Object | Size |
|---|---|
| Sparse matrix W (344K × 344K, ~1.37M entries) | ~22 MB |
| One dense matrix (344K × 28) | ~77 MB |
| `edge_dt` per year (up to ~1.37M rows × 2 cols) | ~22 MB |
| `cell_dt` (6.46M × ~115 cols) | ~5.5 GB |
| **Peak total** | **~8-9 GB** (fits in 16 GB) |

**Expected runtime: 2–5 minutes** on a standard laptop, down from 86+ hours — roughly a **1,000×–2,500× speedup**.