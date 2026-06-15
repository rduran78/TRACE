 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is **`build_neighbor_lookup`**, which constructs a lookup of ~6.46 million row-indices by iterating over every cell-year row. For each of the ~6.46M rows, it:

1. Finds the cell's spatial neighbors from the `nb` object.
2. Constructs string keys by pasting neighbor IDs with the current row's year.
3. Looks up those keys in a named vector of ~6.46M entries.

This produces a **list of 6.46M elements**, each containing integer row indices into the full panel. The fundamental inefficiency is that **the neighbor topology is purely spatial and identical across all 28 years**, yet the lookup is rebuilt redundantly for every year. The string-pasting and named-vector lookup over millions of keys is extremely slow in R.

**Key insight:** The neighbor graph is a property of the 344,208 cells, not of the 6.46M cell-years. Only the *variable values* change by year. Therefore:

- The **neighbor structure** (which cells are neighbors of which) should be built **once** over the 344,208 unique cells.
- The **neighbor statistics** (max, min, mean of neighbor values) should be computed **per year**, by slicing the data by year, mapping cell IDs to positions within that year-slice, and using the static neighbor list to pull values.

This reduces the lookup construction from O(6.46M) to O(344K), and the per-variable stats computation becomes a simple year-loop over 28 slices of ~230K rows each, using integer indexing rather than string hashing.

## Optimization Strategy

1. **Build a static cell-level neighbor lookup once** — a named list mapping each cell's position (in `id_order`) to the positions of its rook neighbors. This is O(344K) and done once.

2. **Sort/index the data by year** so that each year-slice can be extracted cheaply (or use `split()`).

3. **For each year-slice**, create a fast mapping from cell ID → row position within that slice. Then for each cell, gather neighbor variable values using the static neighbor list and the within-year position map. Compute max, min, mean via vectorized operations.

4. **Use `data.table`** for efficient split-by-year, column assignment, and memory-friendly operations.

5. **Vectorize the inner loop** using matrix operations: for each year, arrange values in cell-order, build a neighbor-value matrix, and compute row-wise max/min/mean.

This brings the estimated runtime from 86+ hours down to **minutes**.

## Working R Code

```r
library(data.table)

#' Redesigned pipeline: separate static topology from dynamic variable computation.
#' Preserves the trained Random Forest model and the original numerical estimand.

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Ensure data.table format
# ──────────────────────────────────────────────────────────────────────
if (!is.data.table(cell_data)) {

  cell_data <- as.data.table(cell_data)
}

# Preserve original row order so final output aligns with any downstream use
cell_data[, .row_order := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build STATIC cell-level neighbor structure (done ONCE)
#
#   rook_neighbors_unique : spdep nb object, length = length(id_order)
#   id_order              : vector of cell IDs in the order matching the nb object
#
#   We produce:
#     neighbor_mat  — a matrix (n_cells x max_k) of neighbor *positions* in id_order
#     neighbor_k    — integer vector, number of neighbors per cell
#   Padded columns beyond a cell's actual neighbor count are set to NA.
# ──────────────────────────────────────────────────────────────────────

build_static_neighbor_structure <- function(id_order, nb_obj) {
  n_cells <- length(id_order)
  stopifnot(length(nb_obj) == n_cells)

  # Number of neighbors per cell
  k <- vapply(nb_obj, function(x) {
    # spdep nb encodes "no neighbours" as a single 0L
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1))

  max_k <- max(k)

  # Build padded matrix of neighbor positions (indices into id_order)
  mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
  for (i in seq_len(n_cells)) {
    ki <- k[i]
    if (ki > 0L) {
      mat[i, seq_len(ki)] <- nb_obj[[i]]
    }
  }

  list(neighbor_mat = mat, neighbor_k = k, max_k = max_k)
}

message("Building static neighbor structure …")
nb_struct <- build_static_neighbor_structure(id_order, rook_neighbors_unique)
neighbor_mat <- nb_struct$neighbor_mat   # (344208 x max_k) integer matrix
neighbor_k   <- nb_struct$neighbor_k
max_k        <- nb_struct$max_k
n_cells      <- length(id_order)

# Fast lookup: cell_id -> position in id_order (integer)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# ──────────────────────────────────────────────────────────────────────
# STEP 2: For each YEAR, compute neighbor stats for all source variables
#
#   Strategy per year:
#     - Extract the year-slice (≈230–345K rows).
#     - Map each row's cell id to its position in id_order.
#     - For each source variable, arrange values into a vector aligned
#       with id_order (cells not present in this year get NA).
#     - Use the static neighbor_mat to gather neighbor values into a matrix,
#       then compute row-wise max, min, mean (vectorised).
#     - Write results back to the data.table.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns in cell_data
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  if (!col_max  %in% names(cell_data)) cell_data[, (col_max)  := NA_real_]
  if (!col_min  %in% names(cell_data)) cell_data[, (col_min)  := NA_real_]
  if (!col_mean %in% names(cell_data)) cell_data[, (col_mean) := NA_real_]
}

# Key the table for fast subsetting
setkey(cell_data, year)
years <- sort(unique(cell_data$year))

message("Computing neighbor statistics by year …")

for (yr in years) {
  message("  year = ", yr)

  # Extract year-slice row indices in cell_data
  idx_yr <- cell_data[.(yr), which = TRUE]
  n_yr   <- length(idx_yr)

  # Map each row's cell id to its position in id_order
  cell_ids_yr <- cell_data$id[idx_yr]
  pos_yr      <- id_to_pos[as.character(cell_ids_yr)]  # position in id_order

  # Build reverse map: for each id_order position, what is the index

  # *within this year-slice* (1..n_yr)?  NA if cell absent this year.
  pos_to_local <- rep(NA_integer_, n_cells)
  pos_to_local[pos_yr] <- seq_len(n_yr)

  # For each cell present this year, gather the local indices of its neighbors

  # neighbor_mat[pos_yr, ] gives neighbor positions in id_order;
  # we then translate to local indices via pos_to_local.

  # Gather neighbor id_order positions for present cells  (n_yr x max_k)
  nb_positions <- neighbor_mat[pos_yr, , drop = FALSE]  # id_order positions

  # Translate to local year-slice indices
  # (vectorised lookup; NAs propagate correctly)
  nb_local <- matrix(pos_to_local[nb_positions],
                     nrow = n_yr, ncol = max_k)

  for (var_name in neighbor_source_vars) {
    # Values for this variable in this year-slice
    vals <- cell_data[[var_name]][idx_yr]   # length n_yr

    # Gather neighbor values into matrix (n_yr x max_k)
    # Cells with no neighbor at a column get NA
    nb_vals <- matrix(vals[nb_local], nrow = n_yr, ncol = max_k)

    # Compute row-wise stats using matrixStats for speed if available,
    # otherwise base R.  We handle all-NA rows gracefully.

    # Count non-NA per row (to detect rows with zero valid neighbors)
    not_na   <- !is.na(nb_vals)
    row_nna  <- rowSums(not_na)

    # Replace NAs with +/-Inf for max/min computation, then fix all-NA rows
    nb_for_max <- nb_vals
    nb_for_max[!not_na] <- -Inf
    row_max <- do.call(pmax, c(as.data.frame(nb_for_max), na.rm = FALSE))

    nb_for_min <- nb_vals
    nb_for_min[!not_na] <- Inf
    row_min <- do.call(pmin, c(as.data.frame(nb_for_min), na.rm = FALSE))

    # Mean: sum of non-NA values / count
    nb_for_sum <- nb_vals
    nb_for_sum[!not_na] <- 0
    row_sum  <- rowSums(nb_for_sum)
    row_mean <- row_sum / row_nna

    # Set all-NA rows to NA
    all_na <- row_nna == 0L
    row_max[all_na]  <- NA_real_
    row_min[all_na]  <- NA_real_
    row_mean[all_na] <- NA_real_

    # Write back to cell_data
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    set(cell_data, i = idx_yr, j = col_max,  value = row_max)
    set(cell_data, i = idx_yr, j = col_min,  value = row_min)
    set(cell_data, i = idx_yr, j = col_mean, value = row_mean)
  }
}

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Restore original row order & proceed to prediction
# ──────────────────────────────────────────────────────────────────────
setorder(cell_data, .row_order)
cell_data[, .row_order := NULL]

message("Neighbor features computed. Ready for Random Forest prediction.")

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Predict with the pre-trained Random Forest (UNCHANGED)
#
#   The trained model object (e.g., `rf_model`) and the predict() call
#   remain exactly as before.  No retraining occurs.
#
#   Example (adjust object/column names to match your pipeline):
#
#   cell_data$predicted <- predict(rf_model, newdata = cell_data)
# ──────────────────────────────────────────────────────────────────────
```

## Summary of Improvements

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup construction** | 6.46M string-key lookups in a named vector | 344K-element static integer matrix, built once |
| **Per-variable stats** | `lapply` over 6.46M list elements per variable | Vectorised matrix ops over 28 year-slices × ~230K rows |
| **Memory** | 6.46M-element list of integer vectors + 6.46M-entry named character vector | One (344K × max_k) integer matrix + year-local temporaries |
| **String operations** | ~6.46M `paste()` calls per build | Zero |
| **Estimated runtime** | 86+ hours | ~5–15 minutes |
| **Numerical result** | — | Identical (same max/min/mean over same rook neighbors) |
| **Trained RF model** | — | Untouched; no retraining |