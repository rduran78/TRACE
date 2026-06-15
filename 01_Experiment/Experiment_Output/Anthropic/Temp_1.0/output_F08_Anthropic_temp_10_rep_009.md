 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It produces ~6.46 million list entries, one per cell-year row. But the neighbor *relationships* among cells are purely spatial and never change across years. There are only 344,208 unique cells. The function does redundant work by a factor of 28 (the number of years).

2. **The lookup is keyed by `paste(id, year)` strings.** This creates ~6.46 million string keys and performs named-vector lookups (linear scans in R) inside a `lapply` over 6.46 million rows. String concatenation, hashing, and named-vector indexing at this scale is catastrophically slow.

3. **`compute_neighbor_stats` iterates over 6.46 million list elements**, each time subsetting and computing `max`/`min`/`mean` in pure R. This is repeated 5 times (once per source variable), yielding ~32.3 million R-level function invocations with per-element allocation overhead.

4. **Memory pressure.** Storing 6.46 million list entries (each a vector of neighbor row indices that changes per year) consumes significant RAM and stresses the garbage collector on a 16 GB laptop.

### The Key Insight

> **Neighbor topology is static; only the variable values change by year.**

The neighbor list is a property of the 344,208 cells, not of the 6.46 million cell-year rows. We should build the neighbor structure *once* over cells, then compute neighbor statistics *per year* using fast vectorized/matrix operations, slicing the data by year and indexing into a compact cell-level neighbor structure.

---

## Optimization Strategy

### 1. Build a cell-level neighbor lookup once (344K entries, not 6.46M)

Create a list of length 344,208 where element `i` contains the integer indices (into the canonical cell ordering) of cell `i`'s rook neighbors. This is built once from `rook_neighbors_unique` and reused forever.

### 2. Process year-by-year using vectorized matrix indexing

For each year:
- Extract the subset of rows for that year.
- For each source variable, build a values vector indexed by cell position.
- Use the static neighbor list to gather neighbor values via `vapply` over only 344K cells (not 6.46M rows), or better yet, use a sparse-matrix multiplication / `data.table` approach.

### 3. Use a CSR-like (Compressed Sparse Row) approach with vectorized R

Convert the neighbor list into two flat vectors (`neighbor_idx`, `cell_ptr`) and use `cumsum`-based group operations. This avoids all `lapply` overhead and enables fully vectorized `max`/`min`/`mean` via `fmin`/`fmax`/`fmean` from the `collapse` package (or `data.table` grouping).

### Complexity Reduction

| Aspect | Before | After |
|---|---|---|
| Neighbor lookup entries | 6.46M | 344K (built once) |
| String key operations | ~6.46M `paste` + named lookups | 0 |
| Stats computation loops | 6.46M × 5 vars | 344K × 5 vars × 28 years (vectorized) |
| Estimated time | 86+ hours | **~2–10 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) data
# =============================================================================

library(data.table)

# ---- Step 0: Ensure cell_data is a data.table for performance ----
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build the STATIC cell-level neighbor structure (once) ---------
# id_order: vector of 344,208 cell IDs in the canonical order matching
#           rook_neighbors_unique (the spdep nb object).
# rook_neighbors_unique: nb object, list of length 344,208; each element is
#           an integer vector of neighbor positions (indices into id_order),
#           with 0L meaning no neighbors.

build_static_neighbor_structure <- function(id_order, neighbors_nb) {
  # id_order[i] is the cell ID at position i
  # neighbors_nb[[i]] gives the positions (in id_order) of neighbors of cell i
  # We convert this to a CSR-like flat representation for vectorized ops.

  n_cells <- length(id_order)

  # Clean: in spdep nb objects, a single 0L means "no neighbors"
  neighbor_list <- lapply(seq_len(n_cells), function(i) {
    nb <- neighbors_nb[[i]]
    if (length(nb) == 1L && nb[0 + 1] == 0L) integer(0) else as.integer(nb)
    # spdep uses 0L to denote no neighbors; check properly:
  })
  # Actually spdep encodes no-neighbor as integer(0) or 0L depending on version

  neighbor_list <- lapply(neighbors_nb, function(nb) {
    nb <- as.integer(nb)
    nb[nb != 0L]
  })

  # Build CSR representation
  lengths_vec <- vapply(neighbor_list, length, integer(1))
  flat_neighbors <- unlist(neighbor_list, use.names = FALSE)
  # cell_ptr: cumulative pointer; cell i's neighbors are in
  # flat_neighbors[(cell_ptr[i]+1):cell_ptr[i+1]]
  cell_ptr <- c(0L, cumsum(lengths_vec))

  # Also build a map from cell ID -> position index
  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  list(
    id_order       = id_order,
    id_to_pos      = id_to_pos,
    n_cells        = n_cells,
    flat_neighbors = flat_neighbors,
    cell_ptr       = cell_ptr,
    n_neighbors    = lengths_vec
  )
}

cat("Building static neighbor structure...\n")
nb_struct <- build_static_neighbor_structure(id_order, rook_neighbors_unique)
cat("  Done. Cells:", nb_struct$n_cells,
    " Total directed edges:", length(nb_struct$flat_neighbors), "\n")


# ---- Step 2: Vectorized neighbor stats using CSR + grouping ---------------

compute_neighbor_stats_vectorized <- function(values_by_pos, nb_struct) {
  # values_by_pos: numeric vector of length n_cells, indexed by cell position.
  #   values_by_pos[i] = value for the cell at position i in id_order.
  #   NA is allowed.
  #
  # Returns: matrix of (n_cells x 3): columns = max, min, mean

  n_cells        <- nb_struct$n_cells
  flat_neighbors <- nb_struct$flat_neighbors
  cell_ptr       <- nb_struct$cell_ptr
  n_neighbors    <- nb_struct$n_neighbors

  # Gather all neighbor values in one vectorized step
  neighbor_vals <- values_by_pos[flat_neighbors]  # length = total edges

  # Create a group ID for each edge (which cell does it belong to?)
  # cell i owns edges from (cell_ptr[i]+1) to cell_ptr[i+1]
  group_id <- rep.int(seq_len(n_cells), times = n_neighbors)

  # Handle NAs: mark NA values so they are excluded from aggregation
  valid <- !is.na(neighbor_vals)
  neighbor_vals_valid <- neighbor_vals[valid]
  group_id_valid      <- group_id[valid]

  # Compute aggregates using data.table for speed
  if (length(neighbor_vals_valid) == 0) {
    return(matrix(NA_real_, nrow = n_cells, ncol = 3,
                  dimnames = list(NULL, c("max", "min", "mean"))))
  }

  dt <- data.table(g = group_id_valid, v = neighbor_vals_valid)
  agg <- dt[, .(nb_max = max(v), nb_min = min(v), nb_mean = mean(v)), by = g]

  # Map back to full n_cells vector
  result <- matrix(NA_real_, nrow = n_cells, ncol = 3)
  result[agg$g, 1] <- agg$nb_max
  result[agg$g, 2] <- agg$nb_min
  result[agg$g, 3] <- agg$nb_mean

  result
}


# ---- Step 3: Process all years × all variables ----------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate new columns
for (var_name in neighbor_source_vars) {
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")
  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]
}

# We need a mapping from cell ID -> position for row matching
# Ensure cell_data has id and year columns
# Process year by year

years <- sort(unique(cell_data$year))
cat("Processing", length(years), "years x", length(neighbor_source_vars),
    "variables...\n")

# Create a position column for each row (cell position in id_order)
cell_data[, cell_pos := nb_struct$id_to_pos[as.character(id)]]

for (yr in years) {
  cat("  Year:", yr, "\n")

  # Get row indices for this year
  yr_rows <- which(cell_data$year == yr)

  # Get the cell positions for these rows (which position in id_order)
  positions <- cell_data$cell_pos[yr_rows]

  for (var_name in neighbor_source_vars) {
    # Build a values vector indexed by cell position for this year
    # Initialize with NA
    values_by_pos <- rep(NA_real_, nb_struct$n_cells)
    values_by_pos[positions] <- cell_data[[var_name]][yr_rows]

    # Compute neighbor stats (vectorized over all 344K cells)
    stats <- compute_neighbor_stats_vectorized(values_by_pos, nb_struct)

    # Write results back to the correct rows
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    set(cell_data, i = yr_rows, j = max_col,  value = stats[positions, 1])
    set(cell_data, i = yr_rows, j = min_col,  value = stats[positions, 2])
    set(cell_data, i = yr_rows, j = mean_col, value = stats[positions, 3])
  }
}

# Clean up helper column
cell_data[, cell_pos := NULL]

cat("Neighbor feature computation complete.\n")

# ---- Step 4: Prediction with the pre-trained Random Forest (unchanged) -----
# The trained RF model is preserved exactly as-is. The computed neighbor
# features are numerically identical to the original implementation
# (same max, min, mean over the same neighbor sets), so predictions
# will be identical.
#
# Example (unchanged from original pipeline):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

1. **Same neighbor sets.** The static cell-level neighbor structure encodes exactly the same rook-contiguity relationships as the original `rook_neighbors_unique` nb object. The per-year slicing reproduces the same `paste(id, year)` matching—cell `i`'s neighbors in year `t` are looked up in year `t`'s data only.

2. **Same aggregation functions.** `max`, `min`, and `mean` are computed over the identical set of non-NA neighbor values. The output columns have the same names (`{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean`).

3. **Trained RF model untouched.** No retraining occurs. The model object is used as-is with `predict()`.

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-keyed list entries | 344K-cell CSR structure (built once) |
| Stats computation | `lapply` over 6.46M entries × 5 vars | Vectorized gather + `data.table` grouping, 28 year-slices × 5 vars × 344K cells |
| String operations | ~19M `paste()` + named vector lookups | Zero |
| Estimated wall time | 86+ hours | ~2–10 minutes |
| Peak RAM | High (6.46M list + string keys) | Moderate (~flat vectors, year slices) |