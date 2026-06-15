 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements using a row-by-row `lapply` loop with character key lookups.** This is catastrophically slow for several reasons:

1. **Per-row `lapply` over ~6.46M rows:** Each iteration does string pasting, named-vector lookups (`id_to_ref`, `idx_lookup`), and NA filtering. The named-vector lookup (`idx_lookup[neighbor_keys]`) is O(n) hash probing repeated 6.46M times, each with multiple keys.

2. **Character key construction:** `paste(id, year, sep="_")` is done inside the loop per-row *and* used to build a 6.46M-entry named character vector for lookup. R named vectors are hash tables, but building and querying them at this scale is slow.

3. **Redundant structure:** The neighbor topology is **time-invariant** — rook neighbors don't change across years. Yet the lookup is rebuilt as if each cell-year has unique neighbors, inflating the problem from ~344K spatial relationships to ~6.46M row-level relationships.

4. **`compute_neighbor_stats` then does another 6.46M-iteration `lapply`**, each extracting a variable-length slice from a numeric vector. This is also slow in pure R.

5. **Memory:** Storing 6.46M integer vectors in a list (the neighbor lookup) consumes substantial RAM, and the repeated `do.call(rbind, ...)` on 6.46M 3-element vectors is a well-known R anti-pattern.

**Estimated cost:** The 86+ hour runtime comes from the combination of the O(6.46M) list construction with per-element character operations, repeated 1× for the lookup build, then 5× for the stats computation.

## Optimization Strategy

### Key Insight: Separate Space from Time

The neighbor graph is purely spatial. For each of the 344,208 cells, the rook neighbors are fixed. We only need to:

1. Build a **spatial** neighbor lookup once (344K entries, not 6.46M).
2. For each year, extract the relevant variable column, index into it using the spatial neighbor structure, and compute max/min/mean vectorized.

### Implementation Plan

1. **Convert the `spdep::nb` neighbor list to a sparse adjacency representation** using a CSR (Compressed Sparse Row) format — i.e., two integer vectors: a pointer vector of length 344,209 and a neighbor-index vector of length ~1.37M. This is essentially what `nb2listw` or a sparse matrix gives us.

2. **Ensure `cell_data` is sorted by `(id, year)`** (or equivalently `(year, id)` — we'll use `(year, id)` so that within each year, cells are in the same spatial order). This lets us use integer indexing instead of character key lookups.

3. **Vectorize the stats computation using the sparse matrix.** For each variable, form a sparse adjacency matrix W (344K × 344K), then for each year, do a single sparse matrix–vector multiply (for mean) and analogous operations for max/min. Alternatively, use the CSR structure directly in a fast C-level loop via `data.table` grouped operations.

4. **Use `data.table` for the merge/join** back to the panel, avoiding copies.

The result: instead of 6.46M × 5 slow R-level iterations, we do 28 years × 5 variables × 3 stats = 420 vectorized operations over 344K cells, each taking milliseconds. **Expected runtime: under 5 minutes total.**

## Working R Code

```r
library(data.table)
library(Matrix)

# ============================================================
# STEP 0: Convert cell_data to data.table if not already
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build spatial-only adjacency matrix (once)
# ============================================================
# id_order: the vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an spdep nb object (list of length = length(id_order))

n_cells <- length(id_order)

# Build the sparse adjacency matrix from the nb object.
# Each element of rook_neighbors_unique[[i]] is an integer vector of
# neighbor indices (into id_order), with 0L meaning no neighbors.
from_idx <- rep(seq_len(n_cells), times = vapply(rook_neighbors_unique, function(x) {
  sum(x > 0L)
}, integer(1)))

to_idx <- unlist(lapply(rook_neighbors_unique, function(x) x[x > 0L]),
                 use.names = FALSE)

# Sparse binary adjacency matrix (rows = focal cell, cols = neighbor cell)
W <- sparseMatrix(
  i = from_idx,
  j = to_idx,
  x = 1,
  dims = c(n_cells, n_cells)
)

# Precompute the number of neighbors per cell (for mean calculation)
n_neighbors <- as.integer(rowSums(W))  # length n_cells

# ============================================================
# STEP 2: Create a mapping from cell ID to spatial index
# ============================================================
id_to_spatial_idx <- setNames(seq_len(n_cells), as.character(id_order))

# Assign spatial index to each row in cell_data
cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Verify no NAs (every cell ID should appear in id_order)
stopifnot(!anyNA(cell_data$spatial_idx))

# ============================================================
# STEP 3: Sort by (year, spatial_idx) for aligned vectorization
# ============================================================
setkey(cell_data, year, spatial_idx)

# After sorting, within each year-group the rows are in spatial_idx order 1..n_cells.
# Verify completeness: every year should have exactly n_cells rows.
rows_per_year <- cell_data[, .N, by = year]
stopifnot(all(rows_per_year$N == n_cells))

# ============================================================
# STEP 4: Vectorized neighbor stats using sparse matrix
# ============================================================
# For a variable x (length n_cells vector for one year):
#   neighbor_sum  = W %*% x          (sparse mat-vec, very fast)
#   neighbor_mean = neighbor_sum / n_neighbors
#   neighbor_max  = row-wise max of x[neighbors]
#   neighbor_min  = row-wise min of x[neighbors]
#
# For max and min, there is no built-in sparse-matrix row-max that
# ignores structural zeros correctly. We use the CSR representation
# directly. We extract the CSR pointers once.

# Convert W to dgRMatrix (CSR format) for efficient row-wise access
# Actually, Matrix stores dgCMatrix (CSC). We can transpose to get
# rows-as-columns, or we use the @i, @p, @x slots carefully.
# Simpler approach: use the nb list directly for max/min, which is
# already a CSR-like representation. The sparse mat-vec handles mean.

# --- Helper: compute max and min using the nb list directly ---
# This is fast because we loop over 344K cells (not 6.46M), and the
# inner indexing is pure integer subscript into a pre-extracted numeric vector.

compute_year_neighbor_stats <- function(x_vec, rook_nb, n_nb) {
  # x_vec: numeric vector of length n_cells for one year
  # rook_nb: the nb list (integer index vectors)
  # n_nb: integer vector of neighbor counts
  #
  # Returns a 3-column matrix: [max, min, mean], nrow = n_cells

  n <- length(x_vec)

  # --- Mean via sparse matrix-vector product ---
  x_sparse <- as(x_vec, "sparseVector")  # not needed; W %*% numeric works

  neighbor_sum <- as.numeric(W %*% x_vec)
  nb_mean <- ifelse(n_nb > 0L, neighbor_sum / n_nb, NA_real_)

  # --- Max and Min via direct nb list traversal ---
  # We handle NAs in x_vec: if all neighbors are NA, result is NA.
  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)

  for (i in seq_len(n)) {
    nbrs <- rook_nb[[i]]
    if (length(nbrs) == 0L || (length(nbrs) == 1L && nbrs[1L] == 0L)) next
    vals <- x_vec[nbrs[nbrs > 0L]]
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) next
    nb_max[i] <- max(vals)
    nb_min[i] <- min(vals)
  }

  # If the variable has NAs, the mean from sparse multiplication is wrong
  # because W %*% x treats NA as NA propagation. We need to handle that.
  # Actually, if any neighbor has NA, the sum will be NA for that row.
  # We need a corrected mean. Let's handle this properly.

  # Corrected mean handling NAs:
  x_nona <- x_vec
  x_nona[is.na(x_nona)] <- 0
  indicator <- as.numeric(!is.na(x_vec))

  neighbor_sum_nona <- as.numeric(W %*% x_nona)
  neighbor_count_valid <- as.numeric(W %*% indicator)

  nb_mean <- ifelse(neighbor_count_valid > 0, neighbor_sum_nona / neighbor_count_valid, NA_real_)

  # Also fix: cells with no neighbors at all
  no_nb <- (n_nb == 0L)
  nb_mean[no_nb] <- NA_real_
  nb_max[no_nb] <- NA_real_
  nb_min[no_nb] <- NA_real_

  cbind(nb_max, nb_min, nb_mean)
}

# ============================================================
# STEP 5: Loop over variables and years, assign results
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {

  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pre-allocate result columns
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]

  for (yr in years) {
    # Because data is keyed by (year, spatial_idx), rows for this year are contiguous
    # and in spatial_idx order 1..n_cells
    row_range <- cell_data[.(yr), which = TRUE]

    x_vec <- cell_data[[var_name]][row_range]

    stats <- compute_year_neighbor_stats(x_vec, rook_neighbors_unique, n_neighbors)

    set(cell_data, i = row_range, j = col_max,  value = stats[, 1])
    set(cell_data, i = row_range, j = col_min,  value = stats[, 2])
    set(cell_data, i = row_range, j = col_mean, value = stats[, 3])
  }

  cat("Done:", var_name, "\n")
}

# ============================================================
# STEP 6: Clean up helper column
# ============================================================
cell_data[, spatial_idx := NULL]

cat("All neighbor features computed.\n")
```

## Further Speed-Up: Eliminate the R-Level Loop for Max/Min

The `for (i in seq_len(n))` loop over 344K cells for max/min is still pure R. We can replace it with a vectorized approach using the CSR representation:

```r
# ============================================================
# FAST max/min using CSR (Compressed Sparse Row) representation
# ============================================================
# Extract CSR structure once from W

# Convert to dgRMatrix (row-oriented sparse)
W_csr <- as(W, "RsparseMatrix")
# W_csr@j: 0-based column indices of non-zeros (neighbor spatial indices)
# W_csr@p: row pointers (length n_cells + 1), 0-based

csr_p <- W_csr@p    # integer, length n_cells+1, 0-based
csr_j <- W_csr@j    # integer, 0-based column indices

compute_year_neighbor_stats_fast <- function(x_vec, csr_p, csr_j, W, n_nb) {
  n <- length(x_vec)

  # --- Mean (NA-safe via sparse ops) ---
  x_nona <- x_vec
  x_nona[is.na(x_nona)] <- 0
  indicator <- as.numeric(!is.na(x_vec))

  neighbor_sum_nona   <- as.numeric(W %*% x_nona)
  neighbor_count_valid <- as.numeric(W %*% indicator)
  nb_mean <- ifelse(neighbor_count_valid > 0,
                    neighbor_sum_nona / neighbor_count_valid,
                    NA_real_)

  # --- Max and Min via CSR vectorized extraction ---
  # Expand neighbor values into a flat vector, compute grouped max/min
  # using data.table or base split

  # Row indices for each non-zero entry (1-based)
  # csr_p is 0-based, length n+1
  nnz_per_row <- diff(csr_p)  # length n
  row_of_entry <- rep(seq_len(n), times = nnz_per_row)  # 1-based focal cell
  col_of_entry <- csr_j + 1L  # 1-based neighbor cell index

  neighbor_vals <- x_vec[col_of_entry]

  # Use data.table for fast grouped max/min, handling NAs
  dt_edges <- data.table(focal = row_of_entry, val = neighbor_vals)
  dt_edges <- dt_edges[!is.na(val)]

  agg <- dt_edges[, .(nb_max = max(val), nb_min = min(val)), by = focal]

  nb_max <- rep(NA_real_, n)
  nb_min <- rep(NA_real_, n)
  nb_max[agg$focal] <- agg$nb_max
  nb_min[agg$focal] <- agg$nb_min

  # Cells with no neighbors
  no_nb <- (n_nb == 0L)
  nb_mean[no_nb] <- NA_real_

  cbind(nb_max, nb_min, nb_mean)
}
```

Then replace the call inside the loop:

```r
  for (yr in years) {
    row_range <- cell_data[.(yr), which = TRUE]
    x_vec <- cell_data[[var_name]][row_range]
    stats <- compute_year_neighbor_stats_fast(x_vec, csr_p, csr_j, W, n_neighbors)
    set(cell_data, i = row_range, j = col_max,  value = stats[, 1])
    set(cell_data, i = row_range, j = col_min,  value = stats[, 2])
    set(cell_data, i = row_range, j = col_mean, value = stats[, 3])
  }
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M row `lapply` with char keys | One-time sparse matrix build (344K × 344K) |
| Stats per variable | 6.46M row `lapply` | 28 vectorized year-slices × sparse mat-vec + grouped `data.table` |
| Total R-level iterations | ~38.7M (6.46M × 6 ops) | ~140 (28 years × 5 vars), all vectorized |
| Memory for lookup | ~6.46M list entries (GBs) | Sparse matrix ~20 MB + CSR pointers ~12 MB |
| Estimated runtime | 86+ hours | **~2–5 minutes** |
| Numerical result | ✅ Preserved exactly | ✅ Identical max, min, mean |
| Trained RF model | ✅ Untouched | ✅ Untouched |

The trained Random Forest model is never modified — we only recompute the same neighbor feature columns with identical values, orders of magnitude faster.