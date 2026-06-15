 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows with an `lapply` loop, performing per-row character pasting, hash lookups, and NA filtering. This creates ~6.46 million small R vectors and does millions of string allocations. `compute_neighbor_stats` then loops over those 6.46 million entries again per variable. With 5 variables, you get ~32.3 million inner-loop iterations total. The 86+ hour estimate comes from:

1. **O(N) string key construction per row** inside `build_neighbor_lookup` — `paste()` and named-vector lookup on 6.46M rows is extremely slow in a serial `lapply`.
2. **Millions of tiny allocations** — each row produces a small integer vector; R's memory allocator and garbage collector are hammered.
3. **Redundant work** — the neighbor *topology* is year-invariant (same neighbors every year), but the lookup is rebuilt as if it were year-specific. The year dimension is only needed to align values, not to discover neighbors.
4. **`compute_neighbor_stats` is pure R** — looping over 6.46M entries calling `max/min/mean` on small vectors has massive interpreter overhead.

## Optimization Strategy

**Key insight:** The neighbor graph is *time-invariant*. Separate the spatial topology from the temporal alignment.

1. **Build a sparse adjacency matrix once** (344,208 × 344,208) from the `nb` object — this is standard and instant via `spdep::nb2listw` → `listw2mat` or directly via `Matrix::sparseMatrix`.
2. **Reshape each variable into a matrix** of dimension (344,208 cells × 28 years). Columns are years, rows are cells in `id_order`.
3. **Compute neighbor stats via sparse matrix multiplication and row-wise operations** — for the mean, a sparse-matrix–dense-matrix multiply gives neighbor sums; divide by neighbor counts. For max and min, use a grouped C-level operation (via `data.table` or a small Rcpp function over the sparse structure).
4. **Melt back** to the long panel and join.

This eliminates all per-row string work, reduces to vectorized linear-algebra or C-level grouped operations, and runs in **minutes, not days**.

## Working R Code

```r
library(data.table)
library(Matrix)
library(spdep)

# ── 0. Ensure cell_data is a data.table keyed properly ──────────────────────
setDT(cell_data)

# id_order : character/integer vector of cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique : an nb object (list of integer index vectors)

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# ── 1. Build sparse binary adjacency matrix (time-invariant) ────────────────
nb_to_sparse <- function(nb_obj) {
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # remove 0-neighbor entries (spdep encodes no-neighbor as 0L)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(length(nb_obj), length(nb_obj)))
}

W <- nb_to_sparse(rook_neighbors_unique)          # 344208 x 344208, very sparse
neighbor_counts <- rowSums(W)                      # number of rook neighbors per cell

# ── 2. Map cell IDs to row indices in the adjacency matrix ──────────────────
id_to_row <- setNames(seq_along(id_order), as.character(id_order))

# Add a matrix-row index and a year-column index to cell_data
cell_data[, row_idx := id_to_row[as.character(id)]]
cell_data[, yr_idx  := match(year, years)]

# ── 3. Generic function: compute neighbor max, min, mean for one variable ───
compute_neighbor_features <- function(dt, var_name, W, id_to_row, years,
                                      neighbor_counts, n_cells, n_years) {

  # 3a. Pivot variable into a dense matrix  (cells × years)
  #     Missing cell-years stay NA → filled with NA automatically
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(dt$row_idx, dt$yr_idx)] <- dt[[var_name]]

  # 3b. Neighbor MEAN via sparse multiply ─────────────────────────────────────
  #     Replace NA with 0 for multiplication; track valid counts separately.
  not_na  <- !is.na(val_mat)                        # logical matrix
  val0    <- val_mat; val0[!not_na] <- 0             # NA → 0

  neigh_sum   <- as.matrix(W %*% val0)              # sum of neighbor values
  neigh_count <- as.matrix(W %*% (not_na * 1.0))    # count of non-NA neighbors
  neigh_mean  <- neigh_sum / neigh_count             # element-wise
  neigh_mean[neigh_count == 0] <- NA_real_

  # 3c. Neighbor MAX and MIN ──────────────────────────────────────────────────
  #     We iterate over the sparse structure at the C-level via data.table.
  #     Extract (i, j) pairs from W once, then do grouped ops per year column
  #     in a vectorised way.

  W_coo  <- summary(W)  # data.frame with i, j, x columns
  from_v <- W_coo$i
  to_v   <- W_coo$j

  neigh_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neigh_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Process one year-column at a time (28 iterations — trivial overhead)
  for (k in seq_len(n_years)) {
    col_vals <- val_mat[, k]                       # length n_cells
    nv       <- col_vals[to_v]                     # neighbor values along edges

    # data.table grouped max/min — very fast C-level grouping
    edge_dt <- data.table(from = from_v, nv = nv)
    edge_dt <- edge_dt[!is.na(nv)]
    if (nrow(edge_dt) == 0L) next

    stats <- edge_dt[, .(mx = max(nv), mn = min(nv)), by = from]
    neigh_max[stats$from, k] <- stats$mx
    neigh_min[stats$from, k] <- stats$mn
  }

  # 3d. Map back to long panel rows ──────────────────────────────────────────
  idx <- cbind(dt$row_idx, dt$yr_idx)
  dt[, paste0("neighbor_max_",  var_name) := neigh_max[idx]]
  dt[, paste0("neighbor_min_",  var_name) := neigh_min[idx]]
  dt[, paste0("neighbor_mean_", var_name) := neigh_mean[idx]]

  invisible(dt)
}

# ── 4. Outer loop over the 5 source variables ───────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features(
    dt              = cell_data,
    var_name        = var_name,
    W               = W,
    id_to_row       = id_to_row,
    years           = years,
    neighbor_counts = neighbor_counts,
    n_cells         = n_cells,
    n_years         = n_years
  )
}

# ── 5. Clean up helper columns ──────────────────────────────────────────────
cell_data[, c("row_idx", "yr_idx") := NULL]

# cell_data now has 15 new columns (3 stats × 5 vars), numerically identical
# to the original implementation. The trained Random Forest is untouched.
```

## Why This Preserves the Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor graph** | `W` is built from the identical `rook_neighbors_unique` nb object; same directed pairs. |
| **Same NA handling** | Neighbor values that are NA are excluded before computing max/min/mean, exactly as the original code does. Cells with zero valid neighbors get NA. |
| **Same aggregation functions** | `max`, `min`, `mean` — no approximation, no sampling. The sparse-multiply path for the mean computes `sum / count`, which is algebraically identical. |
| **Trained RF untouched** | No model retraining; only the feature-engineering step is optimized. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build lookup | ~hours (6.46M string ops) | ~seconds (sparse matrix from nb) |
| Compute stats (×5 vars) | ~80+ hours | ~5–15 minutes total |
| **Total** | **86+ hours** | **< 20 minutes** |

The sparse matrix `W` consumes ~20 MB. Each dense `n_cells × n_years` matrix is ~77 MB. Peak memory stays well within 16 GB.