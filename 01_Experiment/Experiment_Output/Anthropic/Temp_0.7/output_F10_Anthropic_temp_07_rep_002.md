 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three critical bottlenecks:

1. **O(n²)-scale neighbor lookup construction**: `build_neighbor_lookup` creates a per-row lookup by pasting string keys and matching them in a named vector for every single row (~6.46M rows). The `paste`/`match` on character keys is extremely slow at this scale.

2. **Row-level R `lapply` over 6.46M rows in `compute_neighbor_stats`**: Each call to the anonymous function has R-interpreter overhead (function dispatch, subsetting, `is.na` checks, `max/min/mean`). Over 6.46M iterations × 5 variables = ~32.3M R-level loop iterations.

3. **Topology is entangled with time**: The neighbor graph is purely spatial (rook contiguity among 344,208 cells) but the lookup is rebuilt as if it's a 6.46M-node graph. The same spatial adjacency is redundantly expanded across all 28 years.

**Key insight**: The rook neighbor graph has ~1.37M directed edges among 344,208 spatial nodes. This topology is *identical* for every year. The task is simply: for each year and each variable, gather neighbor values and compute `max`, `min`, `mean`. This is a **sparse matrix–vector product** (for mean) and analogous sparse aggregation (for max/min), repeated 28 × 5 = 140 times — trivially fast if done correctly.

---

## Optimization Strategy

| Principle | Implementation |
|---|---|
| **Separate topology from time** | Build a 344,208 × 344,208 sparse adjacency matrix once from the `nb` object. |
| **Vectorized sparse aggregation** | Use `Matrix` package sparse matrix operations for `mean`. Use `data.table` grouped operations for `max` and `min`. |
| **Avoid character key matching** | Use integer indexing throughout. Map cell IDs to integer positions once. |
| **Process year-by-year in a matrix** | Reshape each variable to a 344,208 × 28 matrix, apply sparse aggregation column-wise. |
| **Minimize memory copies** | Use `data.table` set-by-reference to attach new columns. |

**Expected speedup**: From 86+ hours to **minutes** (typically 5–15 minutes on a 16 GB laptop).

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

optimize_neighbor_pipeline <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                 "def", "usd_est_n2")) {

  # ---------------------------------------------------------------
  # 0. Convert to data.table for efficient column operations
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))

  # ---------------------------------------------------------------
  # 1. Build sparse adjacency matrix ONCE (topology only)
  #    rook_neighbors_unique is an nb object: a list of length n_cells

  #    where element i contains integer indices of neighbors of cell i
  #    (indices into id_order).
  # ---------------------------------------------------------------
  cat("Building sparse adjacency matrix...\n")

  # Build COO (coordinate) representation
  from_list <- lapply(seq_len(n_cells), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) == 0L) return(NULL)
    list(i = rep.int(i, length(nb_i)), j = nb_i)
  })

  ii <- unlist(lapply(from_list, `[[`, "i"), use.names = FALSE)
  jj <- unlist(lapply(from_list, `[[`, "j"), use.names = FALSE)

  # Binary adjacency matrix: A[i,j] = 1 if j is a rook neighbor of i
  A <- sparseMatrix(i = ii, j = jj, x = 1, dims = c(n_cells, n_cells))

  # Degree vector (number of neighbors per cell) for computing mean
  degree <- as.numeric(rowSums(A))  # length n_cells

  cat(sprintf("Adjacency matrix: %d non-zeros (directed edges)\n", length(ii)))

  # ---------------------------------------------------------------
  # 2. Build integer mappings: cell_id -> position, year -> position
  # ---------------------------------------------------------------
  id_to_pos   <- setNames(seq_len(n_cells), as.character(id_order))
  year_to_col <- setNames(seq_len(n_years), as.character(years))

  # Map each row of cell_data to (cell_position, year_position)
  cell_data[, c("cell_pos__", "year_pos__") := list(
    id_to_pos[as.character(id)],
    year_to_col[as.character(year)]
  )]

  # Ensure data is sorted by (cell_pos, year_pos) for matrix filling
  setorder(cell_data, cell_pos__, year_pos__)

  # ---------------------------------------------------------------
  # 3. For each source variable, compute neighbor max, min, mean
  #    Strategy:
  #      - Reshape variable to n_cells x n_years matrix V
  #      - MEAN:  (A %*% V) / degree  (sparse mat-mat multiply)
  #      - MAX/MIN: use the COO edges + data.table grouped aggregation
  # ---------------------------------------------------------------

  # Pre-extract edge list as data.table for max/min computation
  edge_dt <- data.table(from = ii, to = jj)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))
    t0 <- proc.time()

    # 3a. Build n_cells x n_years matrix V from cell_data
    vals <- cell_data[[var_name]]
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(cell_data$cell_pos__, cell_data$year_pos__)] <- vals

    # 3b. MEAN via sparse matrix multiplication
    #     sum_neighbors = A %*% V   (n_cells x n_years)
    #     mean_neighbors = sum_neighbors / degree
    #     Where degree == 0, result is NaN -> convert to NA
    sum_V <- as.matrix(A %*% V)  # dense result, n_cells x n_years
    mean_V <- sum_V / degree     # vectorized division (recycles by column)
    mean_V[degree == 0, ] <- NA_real_
    # Also: if a neighbor exists but its value is NA, the sparse multiply
    # treats it as 0. We need to handle NAs properly.
    # Fix: count non-NA neighbors and non-NA sums separately.

    # --- Correct NA-aware mean ---
    # Replace NA with 0 in V for summation, and create indicator matrix
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0
    V_ind <- (!is.na(V)) * 1  # indicator: 1 if not NA, 0 if NA

    sum_V    <- as.matrix(A %*% V_nona)       # sum of non-NA neighbor values
    count_V  <- as.matrix(A %*% V_ind)        # count of non-NA neighbors
    mean_V   <- sum_V / count_V               # NA-aware mean
    mean_V[count_V == 0] <- NA_real_          # no valid neighbors -> NA

    # 3c. MAX and MIN via edge list + data.table grouped aggregation
    #     For each year, look up neighbor values, group by 'from', compute max/min
    max_V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (yr_idx in seq_len(n_years)) {
      # Get the column of values for this year
      v_yr <- V[, yr_idx]

      # Look up neighbor values: for each edge (from, to), value = v_yr[to]
      edge_dt[, val := v_yr[to]]

      # Remove edges where neighbor value is NA
      valid <- edge_dt[!is.na(val)]

      if (nrow(valid) > 0L) {
        agg <- valid[, .(mx = max(val), mn = min(val)), by = from]
        max_V[agg$from, yr_idx] <- agg$mx
        min_V[agg$from, yr_idx] <- agg$mn
      }
    }

    # 3d. Map results back to cell_data row order
    idx_mat <- cbind(cell_data$cell_pos__, cell_data$year_pos__)

    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    set(cell_data, j = col_max,  value = max_V[idx_mat])
    set(cell_data, j = col_min,  value = min_V[idx_mat])
    set(cell_data, j = col_mean, value = mean_V[idx_mat])

    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("    Done in %.1f seconds\n", elapsed))
  }

  # ---------------------------------------------------------------
  # 4. Clean up temporary columns
  # ---------------------------------------------------------------
  cell_data[, c("cell_pos__", "year_pos__") := NULL]

  # Remove temporary column from edge_dt
  edge_dt[, val := NULL]

  cat("Neighbor feature computation complete.\n")
  return(cell_data)
}


# ===================================================================
# USAGE
# ===================================================================
#
# # Load data (already in memory or from disk)
# # cell_data: data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# # id_order: vector of 344,208 cell IDs matching the nb object indexing
# # rook_neighbors_unique: spdep nb object (loaded via readRDS or similar)
# # rf_model: pre-trained Random Forest (loaded via readRDS)
#
# cell_data <- optimize_neighbor_pipeline(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Now cell_data has 15 new columns:
# #   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
# #   neighbor_max_ec, neighbor_min_ec, neighbor_mean_ec,
# #   ... etc.
#
# # Apply the pre-trained Random Forest (DO NOT retrain)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **max** | `max(vals[neighbor_indices])` after removing NAs | `data.table` grouped `max` over the same edge set, same NA removal | ✅ Identical |
| **min** | `min(vals[neighbor_indices])` after removing NAs | `data.table` grouped `min` over the same edge set, same NA removal | ✅ Identical |
| **mean** | `mean(vals[neighbor_indices])` after removing NAs | `sum(non-NA values) / count(non-NA values)` via sparse matrix ops with NA-indicator correction | ✅ Identical (IEEE 754 floating point sum order may differ by ≤ machine epsilon; practically identical) |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M string paste+match | Sparse matrix built once from integer COO |
| Mean computation | 6.46M R-level loops per variable | Single sparse matrix multiply (`A %*% V`) per variable |
| Max/Min computation | 6.46M R-level loops per variable | 28 `data.table` grouped aggregations per variable (~1.37M edges each) |
| Total loop iterations | ~32.3M R function calls | ~140 vectorized operations |
| Estimated time | **86+ hours** | **5–15 minutes** |
| RAM peak | Moderate (list of 6.46M vectors) | ~1–2 GB (sparse matrix + dense n_cells×n_years matrices) |
| RF model | Untouched | Untouched ✅ |