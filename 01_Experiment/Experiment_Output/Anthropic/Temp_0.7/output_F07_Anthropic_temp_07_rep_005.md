 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It constructs a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookups** — `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Named character vector lookup in R is O(n) hash probing per call, but the real cost is the **6.46M iterations of interpreted R code**, each allocating small character vectors and performing multiple lookups.

2. **Redundant expansion across years** — The neighbor *topology* is fixed across all 28 years (rook neighbors don't change over time), yet the code re-discovers neighbor row indices year by year for every cell-year row. This means the spatial adjacency (344K cells × ~4 neighbors each ≈ 1.37M directed edges) is being re-resolved 28 times, producing ~38.4M individual key lookups inside the loop.

3. **`compute_neighbor_stats` is also slow** — It iterates over 6.46M list elements in interpreted R, calling `max`, `min`, `mean` on small vectors. This is repeated for each of 5 variables (32.3M R-level function calls).

4. **Memory pressure** — The 6.46M-element list of integer vectors (`neighbor_lookup`) itself is large due to R's per-object overhead (~200+ bytes per list element → ~1.3 GB just for the list skeleton).

**Estimated wall-clock**: ~86 hours is consistent with ~6.46M slow R iterations in `build_neighbor_lookup` plus ~32.3M iterations in `compute_neighbor_stats`.

---

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook-neighbor graph is **time-invariant**. Instead of building a 6.46M-element row-level lookup, we:

1. **Work in two layers**: a spatial layer (344K cells) and a time layer (28 years).
2. **Build a sparse adjacency matrix** once from `rook_neighbors_unique` (344K × 344K, ~1.37M nonzeros). This is a standard `dgCMatrix`.
3. **Reshape each variable into a 344K × 28 matrix** (cells × years).
4. **Compute neighbor stats via sparse matrix–dense matrix multiplication** and analogous sparse operations — fully vectorized in compiled C code (Matrix package), no R-level loops.

### Specific Operations

For a variable matrix **V** (344K × 28) and binary adjacency matrix **A** (344K × 344K, with `A[i,j] = 1` iff j is a rook neighbor of i):

- **Neighbor sum** = `A %*% V` (sparse × dense, runs in C)
- **Neighbor count** = `A %*% (!is.na(V))` (to handle NAs correctly)
- **Neighbor mean** = `neighbor_sum / neighbor_count`
- **Neighbor max / min** — cannot be done by matrix multiplication directly, but can be done via a **single pass over the sparse matrix's column-compressed structure** using `data.table` or vectorized indexing, avoiding any R-level per-cell loop.

### Expected Speedup

| Component | Before | After |
|---|---|---|
| `build_neighbor_lookup` | ~60+ hrs (6.46M R iterations) | ~2 sec (sparse matrix construction) |
| `compute_neighbor_stats` (×5 vars) | ~25+ hrs (32.3M R iterations) | ~30 sec (vectorized sparse ops) |
| **Total** | **~86 hrs** | **< 2 minutes** |

Memory: sparse matrix ~20 MB; variable matrices ~70 MB each; well within 16 GB.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Drop-in replacement — preserves the original numerical estimand exactly.
# Requires: Matrix, data.table (both are lightweight, commonly available)
# =============================================================================

library(Matrix)
library(data.table)

#' Build a sparse binary rook-adjacency matrix from an nb object.
#' @param id_order Integer vector of cell IDs in the order used by the nb object.
#' @param nb_obj   An spdep::nb object (list of integer index vectors).
#' @return A dgCMatrix of dimension n x n, where n = length(id_order).
build_adjacency_matrix <- function(id_order, nb_obj) {
  n <- length(id_order)
  stopifnot(length(nb_obj) == n)

  # Expand to edge list (i -> j directed pairs)
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep "0 = no neighbors" sentinel

  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]

  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n),
               dimnames = list(as.character(id_order),
                               as.character(id_order)))
}

#' Compute neighbor max, min, mean for one variable across all cell-years.
#' Exactly reproduces the original compute_neighbor_stats logic:
#'   - Only non-NA neighbor values are considered.
#'   - If a cell has zero non-NA neighbors in a year, all three stats are NA.
#'
#' @param dt        data.table with columns: id, year, and `var_name`.
#' @param var_name  Character: name of the variable column.
#' @param A         Sparse adjacency matrix (dgCMatrix), rows/cols named by cell id.
#' @param id_order  Character vector of cell IDs matching A's row/col names.
#' @param years     Sorted integer vector of unique years.
#' @return data.table with columns: id, year, nb_max, nb_min, nb_mean.
compute_neighbor_stats_fast <- function(dt, var_name, A, id_order, years) {


  n_cells <- length(id_order)
  n_years <- length(years)

  # --- 1. Reshape variable into a cells x years matrix ----------------------
  # Create a fast lookup: id -> row index in matrix

  id_idx  <- setNames(seq_along(id_order), id_order)
  yr_idx  <- setNames(seq_along(years), as.character(years))

  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  row_i <- id_idx[as.character(dt$id)]
  col_j <- yr_idx[as.character(dt$year)]
  V[cbind(row_i, col_j)] <- dt[[var_name]]

  # --- 2. Neighbor MEAN via sparse matrix multiplication --------------------
  # Replace NA with 0 for summation; track non-NA counts separately.
  V_nona      <- V
  V_nona[is.na(V_nona)] <- 0
  not_na_mask <- (!is.na(V)) * 1  # 1 where non-NA, 0 where NA

  nb_sum   <- as.matrix(A %*% V_nona)       # n_cells x n_years

  nb_count <- as.matrix(A %*% not_na_mask)   # n_cells x n_years

  nb_mean <- nb_sum / nb_count               # NaN where nb_count == 0
  nb_mean[nb_count == 0] <- NA_real_

  # --- 3. Neighbor MAX and MIN via vectorized sparse-edge expansion ---------
  # Extract edge list from sparse matrix (CSC format)
  # A is dgCMatrix: slots @i (0-based row indices), @p (column pointers), @x
  At <- as(A, "dgTMatrix")  # triplet form for easy extraction
  edge_from <- At@i + 1L    # 1-based: "from" cell (the cell whose neighbor list we're filling)
  edge_to   <- At@j + 1L    # 1-based: "to" cell (the neighbor)

  n_edges <- length(edge_from)

  # For each year, gather neighbor values along edges, then group-aggregate.
  # We vectorize across all edges × years simultaneously.
  edge_from_rep <- rep(edge_from, times = n_years)
  edge_to_rep   <- rep(edge_to,   times = n_years)
  year_col_rep  <- rep(seq_len(n_years), each = n_edges)

  # Fetch neighbor values: V[to, year]
  neighbor_vals <- V[cbind(edge_to_rep, year_col_rep)]

  # Build a data.table for grouped max/min (data.table is extremely fast at this)
  edge_dt <- data.table(
    cell_row = edge_from_rep,
    year_col = year_col_rep,
    val      = neighbor_vals
  )

  # Remove edges where the neighbor value is NA (matches original logic)
  edge_dt <- edge_dt[!is.na(val)]

  agg <- edge_dt[, .(nb_max = max(val), nb_min = min(val)),
                 keyby = .(cell_row, year_col)]

  # Initialize result matrices
  nb_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  nb_max_mat[cbind(agg$cell_row, agg$year_col)] <- agg$nb_max
  nb_min_mat[cbind(agg$cell_row, agg$year_col)] <- agg$nb_min

  # --- 4. Flatten back to the original row order of dt ----------------------
  out_idx <- cbind(row_i, col_j)

  data.table(
    id      = dt$id,
    year    = dt$year,
    nb_max  = nb_max_mat[out_idx],
    nb_min  = nb_min_mat[out_idx],
    nb_mean = nb_mean[out_idx]
  )
}

#' Top-level driver: add neighbor features for all source variables.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data  data.frame/data.table with columns: id, year, and all
#'                   neighbor_source_vars columns.
#' @param id_order   Integer vector of cell IDs in nb-object order.
#' @param rook_neighbors_unique  spdep::nb object.
#' @return cell_data with new columns appended (same row order preserved).
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique) {

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # Convert to data.table for speed (non-destructive copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }

  # Build sparse adjacency matrix once (~2 seconds, ~20 MB)
  message("Building sparse adjacency matrix...")
  A <- build_adjacency_matrix(id_order, rook_neighbors_unique)

  id_order_chr <- as.character(id_order)
  years <- sort(unique(cell_data$year))

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    stats_dt <- compute_neighbor_stats_fast(
      dt       = cell_data,
      var_name = var_name,
      A        = A,
      id_order = id_order_chr,
      years    = years
    )

    # Name columns to match original pipeline's naming convention
    # Adjust these suffixes if your compute_and_add_neighbor_features used
    # different names — the numerical values are identical.
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    cell_data[, (max_col)  := stats_dt$nb_max]
    cell_data[, (min_col)  := stats_dt$nb_min]
    cell_data[, (mean_col) := stats_dt$nb_mean]
  }

  if (was_df) cell_data <- as.data.frame(cell_data)

  message("Done. All neighbor features added.")
  cell_data
}

# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================
# cell_data <- add_all_neighbor_features(cell_data,
#                                        id_order,
#                                        rook_neighbors_unique)
#
# # Then proceed directly to prediction with the already-trained RF model:
# # predictions <- predict(rf_model, newdata = cell_data)
# =============================================================================
```

---

## Memory Note for 16 GB Laptops

The edge expansion in step 3 creates ~`1.37M edges × 28 years ≈ 38.4M` rows in `edge_dt`. At ~24 bytes/row this is ~920 MB, which fits comfortably. If memory is tighter, you can loop over years in small batches:

```r
# Optional: chunked year loop to reduce peak memory in step 3
# Replace the single edge_dt construction with:
for (yr_block in split(seq_len(n_years), ceiling(seq_len(n_years) / 7))) {
  # ... build edge_dt only for years in yr_block, aggregate, fill matrices ...
}
```

This trades a trivial amount of speed for ~4× lower peak memory.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Data structure | 6.46M-element R list | Sparse matrix + dense matrix |
| Neighbor mean | R-level loop | `A %*% V` (BLAS/C) |
| Neighbor max/min | R-level loop | Vectorized edge expansion + `data.table` grouped aggregation |
| Estimated runtime | ~86 hours | **< 2 minutes** |
| Peak RAM | ~3–5 GB (list overhead) | ~1.5 GB |
| Numerical result | Baseline | **Identical** (same max, min, mean over same non-NA neighbor values) |
| Trained RF model | Untouched | **Untouched** |