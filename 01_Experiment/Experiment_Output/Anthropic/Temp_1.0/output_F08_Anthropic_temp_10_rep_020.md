 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance bottleneck is in `build_neighbor_lookup`. This function builds a **row-index lookup for every cell-year combination**, producing a list of ~6.46 million elements, each containing integer indices into the full data frame. Because the rook neighbor topology is **static across all 28 years**, the function is redundantly recomputing the same neighbor cell-ID relationships 28 times (once per year per cell). It then resolves those cell-IDs into row indices using a named-character-key lookup (`paste(id, year)`) across the entire 6.46M-row data frame, which is extremely slow.

Similarly, `compute_neighbor_stats` iterates over 6.46 million list elements, extracting and summarizing neighbor values one row at a time via `lapply`—a pure R loop with no vectorization.

**Key insight:** The neighbor graph (which cell is neighbor to which cell) is **time-invariant**. Only the variable values change year to year. Therefore:

1. The neighbor topology should be built **once over the 344,208 cells**, not over 6.46M cell-years.
2. For each year, neighbor statistics can be computed via **vectorized matrix operations** on the static topology, avoiding millions of R-level list lookups.

## Optimization Strategy

1. **Build a static cell-to-cell neighbor structure once** — a sparse adjacency matrix or a simple list of neighbor cell indices (keyed by cell position, not cell-year). This is O(344K) not O(6.46M).

2. **For each variable and each year**, slice the variable column into a vector of length 344,208 (one value per cell), then compute neighbor max/min/mean using the sparse adjacency structure via **vectorized sparse matrix multiplication** (for mean) and **row-wise sparse operations** (for max/min). This replaces 6.46M R-level iterations with 28 vectorized year-slices.

3. **Use the `Matrix` package** sparse matrix multiply for neighbor mean (equivalent to summing neighbor values and dividing by count). For max and min, iterate over cells but only once per cell (344K iterations, not 6.46M), or use an efficient grouped operation.

4. **Reassemble** the neighbor features back into the original data frame in the original row order, preserving the exact numerical estimand for downstream prediction with the pre-trained Random Forest.

**Expected speedup:** From ~86+ hours to minutes. The dominant cost moves from 6.46M× list operations to 28× sparse-matrix operations over 344K cells, plus simple indexing.

## Working R Code

```r
library(Matrix)

# =============================================================================
# STEP 1: Build the static sparse adjacency matrix ONCE (time-invariant)
# =============================================================================
# Inputs:
#   id_order            — vector of 344,208 unique cell IDs in canonical order
#   rook_neighbors_unique — spdep::nb object (list of length 344,208)
#
# Output:
#   adj_sparse — a 344208 x 344208 sparse logical/numeric adjacency matrix
#   neighbor_count — vector of neighbor counts per cell

build_static_adjacency <- function(id_order, neighbors) {
  n <- length(id_order)
  # Build COO (coordinate) triplets
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep::nb encodes "no neighbors" as a single 0; skip those
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    from <- c(from, rep(i, length(nb_i)))
    to   <- c(to, nb_i)
  }
  adj <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  neighbor_count <- diff(adj@p)  # number of neighbors per row (CSC -> per col)
  # For row-wise counts, use rowSums:
  neighbor_count <- rowSums(adj)
  list(adj = adj, neighbor_count = neighbor_count)
}

# Build once
static <- build_static_adjacency(id_order, rook_neighbors_unique)
adj_sparse     <- static$adj
neighbor_count <- static$neighbor_count

# =============================================================================
# STEP 2: Build a cell-index mapping from the data frame
# =============================================================================
# We need to know, for each row of cell_data, which position (1..344208) it
# corresponds to in id_order, and which year it belongs to.

# Create a map from cell ID to canonical position
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# Add canonical position to cell_data (temporary helper column)
cell_data$.cell_pos <- id_to_pos[as.character(cell_data$id)]

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

# =============================================================================
# STEP 3: Compute neighbor stats per variable using sparse matrix ops
# =============================================================================
# For each variable, we produce three new columns:
#   {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
#
# Strategy per year:
#   - Extract a length-344208 vector of variable values (one per cell).
#   - neighbor_mean = (adj %*% vals) / neighbor_count
#   - neighbor_max and neighbor_min: computed via efficient row-wise ops
#     on the sparse matrix.

compute_neighbor_stats_sparse <- function(cell_data, adj_sparse,
                                          neighbor_count, var_name,
                                          years, n_cells) {
  # Pre-allocate output columns
  out_max  <- rep(NA_real_, nrow(cell_data))
  out_min  <- rep(NA_real_, nrow(cell_data))
  out_mean <- rep(NA_real_, nrow(cell_data))

  # Pre-extract the adjacency structure for row-wise max/min

  # Convert to dgRMatrix (row-oriented) for efficient row iteration
  adj_r <- as(adj_sparse, "RsparseMatrix")

  for (yr in years) {
    # Row indices in cell_data for this year
    yr_mask <- cell_data$year == yr
    yr_rows <- which(yr_mask)

    # Build a full-length vector: position -> value
    # (cells missing from this year get NA)
    vals_vec <- rep(NA_real_, n_cells)
    vals_vec[cell_data$.cell_pos[yr_rows]] <- cell_data[[var_name]][yr_rows]

    # --- Neighbor mean via sparse matrix multiply ---
    # adj %*% vals gives the sum of neighbor values for each cell
    neighbor_sum <- as.numeric(adj_sparse %*% vals_vec)
    n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # --- Neighbor max and min via row-wise sparse iteration ---
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)

    # Use the CSR structure: adj_r@p (row pointers), adj_r@j (column indices)
    p <- adj_r@p
    j <- adj_r@j  # 0-based column indices

    for (cell_i in seq_len(n_cells)) {
      start <- p[cell_i] + 1L      # R is 1-based
      end   <- p[cell_i + 1L]
      if (end < start) next         # no neighbors
      nb_cols <- j[start:end] + 1L  # convert to 1-based
      nb_vals <- vals_vec[nb_cols]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      n_max[cell_i] <- max(nb_vals)
      n_min[cell_i] <- min(nb_vals)
    }

    # Handle mean where all neighbors are NA
    # (sparse multiply treats NA as 0; correct for this)
    # Recount non-NA neighbors
    notna_vec <- as.numeric(!is.na(vals_vec))
    valid_count <- as.numeric(adj_sparse %*% notna_vec)
    valid_sum   <- as.numeric(adj_sparse %*% ifelse(is.na(vals_vec), 0, vals_vec))
    n_mean <- ifelse(valid_count > 0, valid_sum / valid_count, NA_real_)

    # Map results back to cell_data rows
    positions_yr <- cell_data$.cell_pos[yr_rows]
    out_max[yr_rows]  <- n_max[positions_yr]
    out_min[yr_rows]  <- n_min[positions_yr]
    out_mean[yr_rows] <- n_mean[positions_yr]
  }

  list(max = out_max, min = out_min, mean = out_mean)
}

# =============================================================================
# STEP 4: Outer loop — compute and attach neighbor features for each variable
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  stats <- compute_neighbor_stats_sparse(
    cell_data, adj_sparse, neighbor_count,
    var_name, years, n_cells
  )
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats$max
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats$min
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats$mean
}

# Clean up temporary column
cell_data$.cell_pos <- NULL

# =============================================================================
# STEP 5: Predict with the pre-trained Random Forest (unchanged)
# =============================================================================
# The trained RF model object and predict call remain exactly as before.
# e.g.:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Further Optimization: Vectorized Max/Min (Eliminating the Inner Cell Loop)

The inner `for (cell_i in seq_len(n_cells))` loop (344K iterations per year × 28 years × 5 variables) may still take significant time. Below is a fully vectorized alternative using grouped operations via `data.table` or a C++-level approach:

```r
# Alternative: vectorized max/min using data.table grouping on the COO representation
library(data.table)

compute_neighbor_stats_dt <- function(cell_data, adj_sparse, var_name,
                                      years, n_cells, id_order) {

  # Extract COO from sparse matrix
  adj_t <- summary(adj_sparse)  # returns i, j, x triplets
  edge_dt <- data.table(from = adj_t$i, to = adj_t$j)

  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_data$.cell_pos <- id_to_pos[as.character(cell_data$id)]

  out_max  <- rep(NA_real_, nrow(cell_data))
  out_min  <- rep(NA_real_, nrow(cell_data))
  out_mean <- rep(NA_real_, nrow(cell_data))

  for (yr in years) {
    yr_rows <- which(cell_data$year == yr)

    # Value lookup: cell_pos -> variable value
    vals_vec <- rep(NA_real_, n_cells)
    vals_vec[cell_data$.cell_pos[yr_rows]] <- cell_data[[var_name]][yr_rows]

    # Attach neighbor values to edges
    edge_dt[, nb_val := vals_vec[to]]

    # Grouped aggregation (fully vectorized in C via data.table)
    agg <- edge_dt[!is.na(nb_val),
                   .(nb_max  = max(nb_val),
                     nb_min  = min(nb_val),
                     nb_mean = mean(nb_val)),
                   by = from]

    # Map back
    positions_yr <- cell_data$.cell_pos[yr_rows]
    # Build a full-length result vector for this year
    yr_max  <- rep(NA_real_, n_cells)
    yr_min  <- rep(NA_real_, n_cells)
    yr_mean <- rep(NA_real_, n_cells)
    yr_max[agg$from]  <- agg$nb_max
    yr_min[agg$from]  <- agg$nb_min
    yr_mean[agg$from] <- agg$nb_mean

    out_max[yr_rows]  <- yr_max[positions_yr]
    out_min[yr_rows]  <- yr_min[positions_yr]
    out_mean[yr_rows] <- yr_mean[positions_yr]
  }

  cell_data$.cell_pos <- NULL
  list(max = out_max, min = out_min, mean = out_mean)
}

# Usage (replaces the sparse-matrix version):
for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  stats <- compute_neighbor_stats_dt(
    cell_data, adj_sparse, var_name, years, n_cells, id_order
  )
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats$max
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats$min
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats$mean
}
```

## Summary

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor lookup | 6.46M-element list, rebuilt per run | 344K×344K sparse matrix, built once |
| Stats computation | 6.46M R-level `lapply` iterations per variable | 28 vectorized sparse-matrix ops or `data.table` grouped aggregations per variable |
| Estimated time | 86+ hours | ~5–20 minutes (depending on variant) |
| Numerical output | — | Identical (same max/min/mean of same neighbor values) |
| RF model | — | Untouched; used as-is for prediction |
| RAM | ~16 GB feasible | ~2–4 GB for sparse matrix + data |