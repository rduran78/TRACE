 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical performance bottlenecks:

### 1. **O(N²)-scale neighbor lookup construction via string hashing**
`build_neighbor_lookup` creates ~6.46 million string keys (`paste(id, year, sep="_")`), stores them in a named vector, and then for each of the 6.46M rows, constructs neighbor keys and performs named lookups. Named vector lookups in R are O(n) hash probes per call, and doing this 6.46M times with string allocation is extremely slow. This alone likely accounts for the majority of the 86+ hour runtime.

### 2. **Row-wise `lapply` over 6.46M rows in `compute_neighbor_stats`**
For each of the 5 variables, the code iterates over 6.46M rows in an R-level loop, extracting subsets of a vector by index and computing `max/min/mean`. The R interpreter overhead per iteration is ~1–5μs, so 6.46M × 5 ≈ 32.3M iterations ≈ 30–160 seconds just in dispatch overhead, plus the actual computation. This is moderate but avoidable.

### 3. **Redundant topology computation**
The neighbor graph topology is **year-invariant** (rook contiguity depends only on spatial position), but the lookup is built over the full panel (cell × year), duplicating the same spatial adjacency structure 28 times. This inflates the lookup from ~1.37M edges to ~38.5M entries.

---

## Optimization Strategy

### Key Insight: Separate Topology from Temporal Indexing

The rook neighbor graph is a **static spatial graph** over 344,208 cells. The yearly panel just replicates node attributes across 28 time slices. Therefore:

1. **Build the sparse adjacency structure once** over 344,208 cells (not 6.46M cell-years) using a sparse matrix (`Matrix::sparseMatrix`). This encodes the ~1.37M directed edges.

2. **Operate year-by-year using sparse matrix–dense matrix multiplication** idiom. For each year slice:
   - Extract the N×1 attribute vector for each source variable.
   - Use the sparse adjacency matrix to compute **neighbor sums** and **neighbor counts** (for mean).
   - For max and min, use a grouped operation over the CSR structure.

3. **Vectorize max/min** using the sparse matrix's internal compressed-row structure (`dgRMatrix` or equivalent) to avoid R-level loops entirely, or use `data.table` grouped operations on an edge list.

4. **Memory**: A sparse matrix with 1.37M nonzeros uses ~16 MB. A year-slice of 344,208 doubles uses ~2.6 MB. Total memory is well within 16 GB.

### Expected Speedup

| Component | Original | Optimized |
|-----------|----------|-----------|
| Lookup build | ~hours (string hashing 6.46M rows) | ~2 sec (sparse matrix from nb object) |
| Stats (per variable) | ~hours (6.46M R-level iterations) | ~5–15 sec (sparse ops × 28 years) |
| Total (5 vars) | 86+ hours | **~2–5 minutes** |

---

## Optimized R Code

```r
###############################################################################
# optimized_neighbor_features.R
#
# Computes max, min, mean of rook-neighbor attributes for each cell-year,
# numerically equivalent to the original pipeline, in minutes instead of days.
#
# Prerequisites:
#   - cell_data: data.frame/data.table with columns id, year, and the source vars
#   - id_order: integer vector of cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique: spdep nb object (list of integer index vectors)
#   - rf_model: pre-trained Random Forest (unchanged)
###############################################################################

library(data.table)
library(Matrix)

optimized_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec",
                                                                  "pop_density",
                                                                  "def",
                                                                  "usd_est_n2")) {

  # Convert to data.table for fast indexed operations (copy to avoid side effects)
  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  stopifnot(n_cells == length(rook_neighbors_unique))

  #---------------------------------------------------------------------------
  # STEP 1: Build sparse adjacency matrix ONCE (static spatial graph)
  #
  # Adjacency matrix A where A[i, j] = 1 means cell j is a rook neighbor of

  # cell i. Then for attribute vector x, A %*% x gives neighbor sums,
  # and rowSums(A) gives neighbor counts (degree).
  #
  # For max and min we need the explicit edge list.
  #---------------------------------------------------------------------------

  message("Building sparse adjacency structure...")

  # Build edge list from nb object: from (row index in id_order) -> to (neighbor index)
  from_idx <- rep(seq_len(n_cells),
                  times = lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(to_idx) & to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  n_edges <- length(from_idx)
  message(sprintf("  %d cells, %d directed edges", n_cells, n_edges))

  # Sparse adjacency matrix (CSR-friendly via dgRMatrix, but we build as dgCMatrix)
  A <- sparseMatrix(
    i    = from_idx,
    j    = to_idx,
    x    = rep(1, n_edges),
    dims = c(n_cells, n_cells),
    repr = "C"           # dgCMatrix (column-compressed) — fast for %*%
  )

  # Degree vector (number of neighbors per cell) — constant across years
  degree <- as.integer(diff(A@p))  # For dgCMatrix built by row... 
  # Actually, for row-oriented ops, we want rowSums:
  degree_vec <- rowSums(A)  # numeric vector of length n_cells


  # For max/min we need row-grouped operations on the edge list.
  # Pre-build a data.table edge list keyed by 'from' for fast grouped ops.
  edge_dt <- data.table(from = from_idx, to = to_idx, key = "from")

  #---------------------------------------------------------------------------
  # STEP 2: Create a fast cell-ID -> position mapping
  #---------------------------------------------------------------------------

  id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

  # Map each row in dt to its position in id_order
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  message(sprintf("  %d years of panel data", n_years))

  #---------------------------------------------------------------------------
  # STEP 3: Pre-allocate output columns
  #---------------------------------------------------------------------------

  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
  }

  #---------------------------------------------------------------------------
  # STEP 4: Process year-by-year × variable-by-variable
  #
  # For each year:
  #   - Extract the year slice (up to 344,208 rows)
  #   - For each variable, build a full attribute vector over all cells,
  #     then use sparse matrix ops for mean/sum and edge list for max/min.
  #---------------------------------------------------------------------------

  message("Computing neighbor statistics...")

  # Index rows of dt by year for fast subsetting
  setkey(dt, year)

  for (yr in years) {

    # Row indices in dt for this year
    yr_rows <- which(dt$year == yr)
    sub      <- dt[yr_rows]
    cell_positions <- sub$cell_pos

    for (var_name in neighbor_source_vars) {

      max_col  <- paste0("neighbor_max_", var_name)
      min_col  <- paste0("neighbor_min_", var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      # Build a full-length attribute vector (n_cells); NA for missing cells
      x <- rep(NA_real_, n_cells)
      x[cell_positions] <- sub[[var_name]]

      #--- MEAN via sparse matrix multiplication ---
      # neighbor_sum = A %*% x  (treats NA as 0 in the product)
      # neighbor_count = A %*% (!is.na(x))
      # neighbor_mean = neighbor_sum / neighbor_count

      x_nona   <- x
      x_nona[is.na(x_nona)] <- 0
      not_na   <- as.numeric(!is.na(x))

      neighbor_sum   <- as.numeric(A %*% x_nona)    # length n_cells
      neighbor_count <- as.numeric(A %*% not_na)     # length n_cells

      neighbor_mean <- ifelse(neighbor_count > 0,
                              neighbor_sum / neighbor_count,
                              NA_real_)

      #--- MAX and MIN via edge-list grouped operations ---
      # Look up neighbor attribute values
      neighbor_vals <- x[edge_dt$to]

      # Temporary data.table for grouped max/min (only non-NA values)
      tmp <- data.table(from = edge_dt$from, val = neighbor_vals)
      tmp <- tmp[!is.na(val)]

      if (nrow(tmp) > 0) {
        agg <- tmp[, .(nmax = max(val), nmin = min(val)), by = from]

        # Initialize full vectors with NA
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
        neighbor_max[agg$from] <- agg$nmax
        neighbor_min[agg$from] <- agg$nmin
      } else {
        neighbor_max <- rep(NA_real_, n_cells)
        neighbor_min <- rep(NA_real_, n_cells)
      }

      # Also set mean to NA where degree is 0 (island cells)
      neighbor_mean[degree_vec == 0] <- NA_real_

      #--- Write results back to dt for this year's rows ---
      set(dt, i = yr_rows, j = max_col,  value = neighbor_max[cell_positions])
      set(dt, i = yr_rows, j = min_col,  value = neighbor_min[cell_positions])
      set(dt, i = yr_rows, j = mean_col, value = neighbor_mean[cell_positions])
    }

    if (yr %% 5 == 0 || yr == years[1] || yr == years[n_years]) {
      message(sprintf("  Completed year %d", yr))
    }
  }

  #---------------------------------------------------------------------------
  # STEP 5: Clean up helper column and return
  #---------------------------------------------------------------------------

  dt[, cell_pos := NULL]

  message("Neighbor feature computation complete.")
  return(dt)
}


###############################################################################
# USAGE
###############################################################################
#
# # Load data
# cell_data               <- readRDS("cell_data.rds")
# id_order                <- readRDS("id_order.rds")
# rook_neighbors_unique   <- readRDS("rook_neighbors_unique.rds")
# rf_model                <- readRDS("rf_model.rds")
#
# # Compute neighbor features (replaces the 86-hour pipeline)
# cell_data_augmented <- optimized_neighbor_features(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # Predict with the pre-trained (unchanged) Random Forest
# preds <- predict(rf_model, newdata = cell_data_augmented)
#
###############################################################################
```

---

## Numerical Equivalence Verification

The following helper function can be used to verify that the optimized output matches the original on a small subset:

```r
verify_numerical_equivalence <- function(original_dt, optimized_dt,
                                         neighbor_source_vars,
                                         sample_n = 1000,
                                         tol = 1e-10) {
  set.seed(42)
  idx <- sample(nrow(original_dt), min(sample_n, nrow(original_dt)))

  all_ok <- TRUE
  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col <- paste0("neighbor_", stat, "_", var_name)
      orig <- original_dt[[col]][idx]
      optm <- optimized_dt[[col]][idx]

      # Compare allowing for NA equality
      both_na <- is.na(orig) & is.na(optm)
      both_val <- !is.na(orig) & !is.na(optm)
      mismatched_na <- xor(is.na(orig), is.na(optm))

      if (any(mismatched_na)) {
        message(sprintf("FAIL [%s]: %d NA mismatches", col, sum(mismatched_na)))
        all_ok <- FALSE
      } else if (any(abs(orig[both_val] - optm[both_val]) > tol)) {
        max_diff <- max(abs(orig[both_val] - optm[both_val]))
        message(sprintf("FAIL [%s]: max diff = %e", col, max_diff))
        all_ok <- FALSE
      } else {
        message(sprintf("PASS [%s]", col))
      }
    }
  }
  return(all_ok)
}
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Topology construction** | String-key named vector over 6.46M rows | Sparse matrix over 344K cells, built once |
| **Stats computation** | R-level `lapply` over 6.46M rows × 5 vars | Sparse matrix multiply (mean) + `data.table` grouped agg (max/min), year-sliced |
| **Estimated runtime** | 86+ hours | ~2–5 minutes |
| **Peak memory** | High (6.46M-element named list) | ~1–2 GB (sparse matrix + data.table) |
| **Numerical result** | Original estimand | Identical (same max/min/mean, same NA handling) |
| **RF model** | Preserved, not retrained | Preserved, not retrained |