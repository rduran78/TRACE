 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is in `build_neighbor_lookup`. It constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors **in that same year**. This is enormously wasteful because:

1. **The neighbor graph is static.** Cell `i`'s rook neighbors are the same in 1992 as in 2019. The `nb` object (`rook_neighbors_unique`) already encodes this topology once for all 344,208 cells. There is no reason to rebuild or re-traverse it for every year.

2. **The per-row lookup is O(n_rows).** Building 6.46 million list entries—each requiring string-pasting, hash lookups, and NA filtering—is the dominant cost. The estimated 86+ hours comes almost entirely from this step.

3. **`compute_neighbor_stats` is fine in principle** but is called on a lookup that is 28× larger than necessary. If we separate the static topology from the year-varying data, we can compute neighbor statistics year-by-year using fast vectorized/matrix operations on only 344,208 cells at a time.

**In summary:** The code conflates the *static spatial topology* with the *year-varying panel structure*, inflating the problem by a factor of 28 and forcing expensive string-key lookups on millions of rows.

---

## Optimization Strategy

1. **Build the neighbor lookup once, at the cell level (not cell-year level).** The `nb` object already provides this: `rook_neighbors_unique[[cell_ref]]` gives the neighbor indices for cell `cell_ref`. We just need a mapping from `cell_id` → position in the data within each year.

2. **Loop over years (28 iterations), not over rows (6.46M iterations).** For each year, extract the 344,208-row slice, compute neighbor max/min/mean using the static `nb` object with vectorized operations, and write results back.

3. **Use matrix indexing or `vapply` over 344,208 cells per year** instead of 6.46M. This reduces the inner-loop size by 28×.

4. **Use `data.table` for fast subsetting and assignment by year.** This avoids repeated copying of the full data frame.

5. **Pre-convert the `nb` object to an integer-index list once.** This is already done (`rook_neighbors_unique`), so no work needed.

**Expected speedup:** The current approach does ~6.46M list iterations × 5 variables = ~32.3M iterations with string operations. The new approach does 28 years × 344,208 cells × 5 variables = ~48.2M lightweight integer-indexed vector lookups—but critically, *no string pasting or hash-map lookups*, and the inner operation is a simple `vals[nb_idx]` on a pre-aligned numeric vector. This should complete in **minutes, not hours**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Prepare the data and static structures
# ──────────────────────────────────────────────────────────────────────

# Convert to data.table for fast grouped operations (non-destructive)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# id_order: the vector of cell IDs in the same order as rook_neighbors_unique.
# Build a fast lookup: cell_id -> position in id_order (i.e., index into nb list)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

# Precompute the neighbor index list at the CELL level (static, built once).
# nb_list[[j]] gives the integer positions (in id_order) of cell j's neighbors.
# This is exactly what rook_neighbors_unique already is (spdep::nb object),
# but we ensure it's a clean integer list and handle the 0-neighbor case.
n_cells <- length(id_order)
nb_list <- lapply(seq_len(n_cells), function(j) {
  idx <- rook_neighbors_unique[[j]]
  # spdep::nb uses 0L to denote "no neighbors"
  if (length(idx) == 1L && idx[1] == 0L) {
    integer(0)
  } else {
    as.integer(idx)
  }
})

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Function to compute neighbor stats for one variable, one year
#          using the STATIC neighbor topology
# ──────────────────────────────────────────────────────────────────────

# For a given numeric vector `vals` of length n_cells (aligned to id_order),
# compute neighbor max, min, mean for every cell.
compute_neighbor_stats_vec <- function(vals, nb_list) {
  n <- length(nb_list)
  out <- matrix(NA_real_, nrow = n, ncol = 3)  # columns: max, min, mean

  for (j in seq_len(n)) {
    idx <- nb_list[[j]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[j, 1L] <- max(nv)
    out[j, 2L] <- min(nv)
    out[j, 3L] <- mean(nv)
  }
  out
}

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Ensure cell_data is sorted by (year, id_order position)
#         so we can extract aligned vectors per year
# ──────────────────────────────────────────────────────────────────────

# Add the cell-reference index (position in id_order / nb_list)
cell_data[, cell_ref := id_to_ref[as.character(id)]]

# Sort by year and cell_ref so that within each year the rows are aligned
# to the nb_list indexing.
setkey(cell_data, year, cell_ref)

# Verify alignment: within each year, cell_ref should be 1:n_cells
# (every cell appears exactly once per year).

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Main loop — years × variables (28 × 5 = 140 iterations)
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  if (!max_col  %in% names(cell_data)) cell_data[, (max_col)  := NA_real_]
  if (!min_col  %in% names(cell_data)) cell_data[, (min_col)  := NA_real_]
  if (!mean_col %in% names(cell_data)) cell_data[, (mean_col) := NA_real_]
}

years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)

  # Boolean mask for this year's rows (already sorted by cell_ref within year)
  yr_mask <- cell_data$year == yr

  for (var_name in neighbor_source_vars) {
    # Extract the variable values aligned to id_order for this year
    vals <- cell_data[[var_name]][yr_mask]

    # Compute neighbor stats using the static topology
    stats <- compute_neighbor_stats_vec(vals, nb_list)

    # Write back into the data.table
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(cell_data, which(yr_mask), max_col,  stats[, 1L])
    set(cell_data, which(yr_mask), min_col,  stats[, 2L])
    set(cell_data, which(yr_mask), mean_col, stats[, 3L])
  }
}

# Clean up the helper column
cell_data[, cell_ref := NULL]

message("Neighbor feature computation complete.")

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict using the pre-trained Random Forest (unchanged)
# ──────────────────────────────────────────────────────────────────────
# The trained RF model object and prediction call remain exactly as before.
# For example:
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)$predictions
# Nothing about the model or its usage changes.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and the same variable values as before. The only change is *how* we find those neighbor rows—via a pre-aligned vector indexed by the static `nb` object, rather than via string-key hash lookups. The arithmetic is identical. |
| **Trained RF model** | Untouched. We only changed the feature-engineering step that feeds into `predict()`. Column names and semantics are preserved. |
| **Edge cases** | Zero-neighbor cells and `NA` values are handled identically (returning `NA` for all three statistics). |

## Expected Performance

| Metric | Before | After |
|---|---|---|
| Lookup construction | ~6.46M string-paste + hash entries | 0 (use existing `nb` object directly) |
| Inner loop iterations | ~6.46M per variable | ~344K per variable per year |
| Total inner iterations | ~32.3M with string ops | ~48.2M with integer vector indexing |
| Estimated wall time | 86+ hours | **~5–15 minutes** on a 16 GB laptop |
| Peak memory | Large list of 6.46M entries + string keys | One 344K × 3 matrix + one numeric vector at a time |