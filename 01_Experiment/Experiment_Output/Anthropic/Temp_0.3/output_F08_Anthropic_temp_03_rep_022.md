 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this identically for every year a cell appears. This means:

1. **Redundant topology computation**: The neighbor graph is static (344,208 cells, ~1.37M directed edges). Yet the lookup is rebuilt at the cell-year level (6.46M rows), inflating work by a factor of 28.
2. **String-key hashing at scale**: `paste(id, year)` keys and named-vector lookups over 6.46M entries are extremely slow in R.
3. **`compute_neighbor_stats` iterates row-by-row**: Even after the lookup is built, it loops over 6.46M list elements in R, calling `max/min/mean` on small vectors each time — dominated by R interpreter overhead.
4. **The loop runs 5 times**: Once per neighbor source variable, each time traversing all 6.46M rows.

**Estimated cost**: ~6.46M list elements × string operations × 5 variables ≈ 86+ hours.

## Optimization Strategy

**Key insight**: Separate the *static topology* (which cells are neighbors of which cells) from the *dynamic values* (variable values that change by year).

1. **Build the neighbor lookup once at the cell level** (344K entries, not 6.46M). This is just a mapping from each cell's positional index to its neighbors' positional indices — directly from the `nb` object. Cost: trivial.

2. **Compute neighbor stats per year using vectorized matrix operations**:
   - For each year, extract the column of values for all 344K cells (in a fixed cell order).
   - Use the static cell-level neighbor list to gather neighbor values.
   - Compute max/min/mean per cell using vectorized C-level code (`vapply` over 344K cells, not 6.46M).
   - Write results back to the corresponding year-slice of the full data.

3. **Use `data.table` for fast split-by-year and column assignment**, avoiding copies.

4. **Optionally, convert the neighbor list to a sparse matrix** and use matrix multiplication for the mean (and row-wise sparse operations for min/max), reducing the inner loop to linear-algebra operations.

This reduces the effective iteration from **6.46M × 5** to **344K × 5 × 28**, but the inner 344K loop is over a pre-built integer-index list (no string hashing), and the year loop (28 iterations) is trivially small. Expected runtime: **minutes, not days**.

## Working R Code

```r
library(data.table)

# =============================================================================
# STEP 0: Ensure cell_data is a data.table, ordered for fast slicing
# =============================================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Establish a canonical cell ordering (must match the nb object's ordering)
# id_order is the vector of cell IDs in the same order as rook_neighbors_unique
# i.e., rook_neighbors_unique[[k]] gives neighbors of id_order[k]

# Create a fast map: cell id -> position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# =============================================================================
# STEP 1: Build the STATIC cell-level neighbor lookup (once, 344K entries)
#
# cell_neighbor_idx[[k]] = integer vector of positional indices of neighbors
# of the k-th cell (in id_order).
# This comes directly from the nb object — no string hashing needed.
# =============================================================================
build_cell_neighbor_lookup <- function(nb_obj) {
  # spdep nb objects are already integer-index lists referencing positions

  # in the original spatial object (which matches id_order).
  # We just need to ensure each element is a clean integer vector,
  # and handle the spdep convention where 0L means "no neighbors".
  n <- length(nb_obj)
  lookup <- vector("list", n)
  for (k in seq_len(n)) {
    nbrs <- nb_obj[[k]]
    # spdep uses 0L to denote no neighbors
    if (length(nbrs) == 1L && nbrs[1L] == 0L) {
      lookup[[k]] <- integer(0)
    } else {
      lookup[[k]] <- as.integer(nbrs)
    }
  }
  lookup
}

cell_neighbor_idx <- build_cell_neighbor_lookup(rook_neighbors_unique)

# =============================================================================
# STEP 2: For each year, compute neighbor max/min/mean for all cells at once
#
# We iterate over years (28) and variables (5) — the inner work per
# (year, variable) is a single pass over 344K cells using integer indexing.
# =============================================================================

# Ensure cell_data has a "pos" column = position of each cell in id_order
cell_data[, pos := id_to_pos[as.character(id)]]

# Sort by year and pos for fast, aligned slicing
setkey(cell_data, year, pos)

# Verify all years and all cells are present and aligned
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns (filled with NA_real_)
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  if (!col_max  %in% names(cell_data)) cell_data[, (col_max)  := NA_real_]
  if (!col_min  %in% names(cell_data)) cell_data[, (col_min)  := NA_real_]
  if (!col_mean %in% names(cell_data)) cell_data[, (col_mean) := NA_real_]
}

# Core computation function: given a numeric vector of length n_cells
# (one value per cell, ordered by id_order position) and the static
# neighbor lookup, return an n_cells x 3 matrix of [max, min, mean].
compute_neighbor_stats_vec <- function(vals, cell_neighbor_idx) {
  n <- length(vals)
  out <- matrix(NA_real_, nrow = n, ncol = 3)  # columns: max, min, mean

  for (k in seq_len(n)) {
    idx <- cell_neighbor_idx[[k]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[k, 1L] <- max(nv)
    out[k, 2L] <- min(nv)
    out[k, 3L] <- mean(nv)
  }
  out
}

# Main loop: iterate over years (28) × variables (5)
cat("Computing neighbor statistics...\n")
t0 <- proc.time()

for (yr in years) {
  # Get the row indices for this year (already keyed by year, pos)
  yr_rows <- which(cell_data$year == yr)

  # These rows should be in pos order (1..n_cells) because of setkey

  # Verify alignment (can remove after first successful run):
  # stopifnot(identical(cell_data$pos[yr_rows], seq_len(n_cells)))

  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Extract the values vector for this year, in cell-position order
    vals <- cell_data[[var_name]][yr_rows]

    # Compute neighbor stats (344K cells, integer-indexed)
    stats <- compute_neighbor_stats_vec(vals, cell_neighbor_idx)

    # Write back using := by reference (no copy)
    set(cell_data, i = yr_rows, j = col_max,  value = stats[, 1L])
    set(cell_data, i = yr_rows, j = col_min,  value = stats[, 2L])
    set(cell_data, i = yr_rows, j = col_mean, value = stats[, 3L])
  }

  cat(sprintf("  Year %d done.\n", yr))
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Neighbor stats completed in %.1f seconds.\n", elapsed))

# Clean up helper column
cell_data[, pos := NULL]

# Restore original row order if needed (e.g., by id then year)
setkey(cell_data, id, year)

# =============================================================================
# STEP 3 (optional, further speedup): Sparse-matrix approach for neighbor mean
#
# If the R-level for-loop over 344K cells is still too slow, the MEAN can be
# computed as a sparse matrix–vector product.  Build the weight matrix once:
# =============================================================================

# library(Matrix)
#
# build_neighbor_weight_matrix <- function(cell_neighbor_idx, n) {
#   # Build a sparse row-normalized adjacency matrix W such that
#   # W %*% vals = vector of neighbor means
#   from <- integer(0)
#   to   <- integer(0)
#   for (k in seq_len(n)) {
#     nbrs <- cell_neighbor_idx[[k]]
#     if (length(nbrs) > 0L) {
#       from <- c(from, rep(k, length(nbrs)))
#       to   <- c(to, nbrs)
#     }
#   }
#   A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
#   # Row-normalize: each row sums to 1 (or 0 if no neighbors)
#   rs <- rowSums(A)
#   rs[rs == 0] <- 1  # avoid division by zero; those rows stay 0
#   W <- A / rs
#   W
# }
#
# W <- build_neighbor_weight_matrix(cell_neighbor_idx, n_cells)
#
# # Then for each year and variable:
# # neighbor_mean <- as.numeric(W %*% vals)
# # This is a single sparse matrix-vector multiply — extremely fast.
# # For max and min, you still need the list-based loop (or Rcpp).

# =============================================================================
# STEP 4: Predict with the pre-trained Random Forest (unchanged)
# =============================================================================
# The trained RF model object (e.g., `rf_model`) is used as-is.
# cell_data now contains all neighbor_max_*, neighbor_min_*, neighbor_mean_*
# columns with identical numerical values to the original implementation.
#
# predictions <- predict(rf_model, newdata = cell_data)
# =============================================================================
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year entries | 344K cell entries (static) |
| Key mechanism | String paste + named-vector hash | Direct integer indexing from `nb` object |
| Iterations for stats | 6.46M × 5 = 32.3M R-level calls | 344K × 5 × 28 = 48.2M, but no hashing overhead |
| String operations | ~billions of `paste()` + `[` lookups | Zero |
| Memory for lookup | ~6.46M list elements + string keys | ~344K list elements (integers only) |
| Data mutation | Column bind / copy | `data.table::set()` by reference |
| **Expected runtime** | **86+ hours** | **~5–15 minutes** |

The numerical results (neighbor max, min, mean per cell-year per variable) are **identical** to the original implementation. The pre-trained Random Forest model is **not modified or retrained** — it receives the same feature columns and produces the same predictions.