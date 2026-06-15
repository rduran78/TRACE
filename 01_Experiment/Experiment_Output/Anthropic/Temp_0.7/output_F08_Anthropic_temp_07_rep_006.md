 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The root cause of the ~86-hour runtime is that `build_neighbor_lookup` constructs a **row-level** lookup for all ~6.46 million cell-year rows, even though the neighbor *topology* is purely spatial and identical across all 28 years. Specifically:

1. **Redundant topology recomputation**: The function creates a lookup keyed by `(id, year)`, producing ~6.46M list entries. But the neighbor graph is defined over 344,208 cells and doesn't change year-to-year. The same neighbor cell IDs are re-resolved 28 times each.

2. **Expensive string-key lookups**: `paste(id, year, sep="_")` and named-vector indexing (`idx_lookup[neighbor_keys]`) on ~6.46M keys is extremely slow in R—O(n) hash lookups repeated inside a `lapply` over 6.46M rows.

3. **Column-at-a-time stat computation is fine in principle**, but it operates on the bloated 6.46M-entry `neighbor_lookup`, multiplying the cost by 28× compared to what's necessary.

**In summary**: The neighbor *structure* (which cells are neighbors of which) is static. Only the *values* attached to cells change by year. The current code fails to exploit this separation, doing 28× the necessary work for topology and then 28× the necessary work for value lookups.

## Optimization Strategy

**Separate the static topology from the dynamic values:**

1. **Build a cell-level neighbor lookup once** over the 344,208 unique cell IDs — not over 6.46M cell-year rows. This is a simple list: for each cell index `i`, store the vector of neighbor cell indices. This is essentially just a cleaned-up version of `rook_neighbors_unique` (the `nb` object) and takes milliseconds.

2. **For each year and each variable**, extract the values vector (length 344,208), then compute neighbor max/min/mean using the static cell-level neighbor list. This turns the inner loop into 28 passes × 5 variables = 140 vectorized passes over 344,208 cells instead of 5 passes over 6.46M cells with expensive string lookups.

3. **Use `data.table` for fast split-by-year and column assignment**, avoiding copies.

4. **Optionally vectorize** the neighbor-stat computation with C++-speed via a small `vapply` over the 344,208-length neighbor list, or use matrix operations.

**Complexity reduction**: From ~6.46M × (string hashing + list indexing) to ~344K × 28 × (integer indexing), a ~28× structural speedup plus elimination of string overhead, yielding an estimated ~100-500× wall-clock improvement (minutes instead of days).

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Prepare data as data.table, sorted by (year, id) for alignment
# ==============================================================================
setDT(cell_data)

# Ensure a canonical ordering of cell IDs (must match rook_neighbors_unique / id_order)
# id_order: vector of 344,208 cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of length 344,208), each element is
#   an integer vector of indices into id_order

# Build a fast map from cell ID -> position in id_order
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Assign each row its spatial position index (done once)
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Sort by year then cell_pos so that within each year, row order = cell_pos order
setkey(cell_data, year, cell_pos)

# Verify alignment: within each year-block, row i corresponds to cell_pos i
# (This is guaranteed by the setkey above as long as every cell appears in every year)
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
stopifnot(nrow(cell_data) == n_cells * length(years))

# ==============================================================================
# STEP 1: Build STATIC cell-level neighbor index list (once, ~344K entries)
# ==============================================================================
# rook_neighbors_unique is an nb object: list of integer vectors
# Convert to a clean list of integer vectors (remove 0L entries that spdep uses
# to signal no-neighbor cells)
cell_neighbor_idx <- lapply(rook_neighbors_unique, function(nb) {
  nb <- as.integer(nb)
  nb[nb > 0L]
})
# cell_neighbor_idx[[i]] gives the positions (in id_order) of neighbors of cell i

# ==============================================================================
# STEP 2: Fast neighbor stat computation per year-variable combination
# ==============================================================================
# For a single variable and a single year's values vector (length n_cells, ordered
# by cell_pos), compute neighbor max, min, mean for every cell.

compute_neighbor_stats_fast <- function(vals, cell_neighbor_idx) {
  # vals: numeric vector of length n_cells, aligned to id_order
  # Returns: matrix of dim (n_cells, 3) — columns: max, min, mean
  n <- length(cell_neighbor_idx)
  out <- matrix(NA_real_, nrow = n, ncol = 3L)

  for (i in seq_len(n)) {
    nb <- cell_neighbor_idx[[i]]
    if (length(nb) == 0L) next
    nv <- vals[nb]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1L] <- max(nv)
    out[i, 2L] <- min(nv)
    out[i, 3L] <- mean(nv)
  }
  out
}

# ==============================================================================
# STEP 3: Loop over variables and years, assign columns back into cell_data
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output column names and initialize them
for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
}

# Because cell_data is keyed by (year, cell_pos), each year occupies a
# contiguous block of n_cells rows in cell_pos order.
# Row indices for year y: ((y_idx - 1) * n_cells + 1) : (y_idx * n_cells)

for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  message(sprintf("Computing neighbor stats for: %s", var_name))

  for (y_idx in seq_along(years)) {
    yr <- years[y_idx]
    row_start <- (y_idx - 1L) * n_cells + 1L
    row_end   <- y_idx * n_cells
    row_range <- row_start:row_end

    # Extract this year's values for the variable, in cell_pos order
    vals <- cell_data[[var_name]][row_range]

    # Compute neighbor stats using the STATIC topology
    stats <- compute_neighbor_stats_fast(vals, cell_neighbor_idx)

    # Write back in-place (data.table set() avoids copies)
    set(cell_data, i = row_range, j = max_col,  value = stats[, 1L])
    set(cell_data, i = row_range, j = min_col,  value = stats[, 2L])
    set(cell_data, i = row_range, j = mean_col, value = stats[, 3L])
  }
}

# ==============================================================================
# STEP 4: (Optional) Restore original row order if downstream code expects it
# ==============================================================================
setkey(cell_data, id, year)  # or whatever the original ordering was

# ==============================================================================
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model object and predict call remain exactly as before.
# Example:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Same numerical estimand** | `max`, `min`, `mean` are computed over exactly the same neighbor value sets as the original code. The only change is *how* we find those neighbors (integer index into a year-slice vs. string-key lookup into the full panel). |
| **Trained RF model untouched** | No retraining occurs. The same column names (`neighbor_max_ntl`, etc.) are produced with identical values, so `predict(rf_model, newdata = cell_data)` yields identical results. |
| **Static topology exploited** | `cell_neighbor_idx` is built once from the `nb` object. Within each year, the values vector is aligned to the same cell ordering, so `vals[nb]` correctly retrieves neighbor values. |

## Expected Performance

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-keyed entries | ~344K integer lists (reused) |
| Inner stat computation | 5 vars × 6.46M rows | 5 vars × 28 years × 344K cells |
| String operations | ~19M `paste()` + hash lookups | Zero |
| Estimated wall-clock | 86+ hours | **~10–30 minutes** |
| Peak RAM | Multiple copies of 6.46M-row lookup | One 344K-entry list + year-slice vectors |

The key insight: **factor the computation into (static topology) × (dynamic values)** instead of materializing the full Cartesian product. This eliminates the dominant cost entirely.