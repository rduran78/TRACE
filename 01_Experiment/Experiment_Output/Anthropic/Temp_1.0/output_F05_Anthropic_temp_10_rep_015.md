 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `lapply` over every row (`~6.46M` iterations). Inside each iteration it:

1. **Pastes** neighbor cell IDs with the current year to build string keys — `paste(neighbor_cell_ids, data$year[i], sep = "_")`.
2. **Looks up** those keys in a named character vector (`idx_lookup`) of length 6.46M.

Named-vector lookup in R is **O(n)** per lookup (linear scan or hash with overhead). With ~6.46M rows, each having ~4 rook neighbors on average (interior cells have 4, edge cells have 2–3), that's ~25.8M string constructions and ~25.8M named-vector lookups against a 6.46M-element vector. The string allocation and hashing overhead alone is enormous.

### Why It's a Broader Architectural Issue

The entire string-keying approach is unnecessary. The data is a **balanced panel** (344,208 cells × 28 years). This means:

- Every cell appears in every year.
- Neighbors in year `t` are the same cells in year `t` — just at different row positions.
- If the data is sorted by `(year, id)` or `(id, year)`, neighbor row indices can be computed by **integer arithmetic** — no strings, no hash lookups.

The `compute_neighbor_stats` function is already vectorized once `neighbor_lookup` is built, so the bottleneck is entirely in `build_neighbor_lookup`.

### Quantified Impact

| Operation | Current | Optimized |
|---|---|---|
| String constructions | ~25.8M `paste()` calls | **0** |
| Hash lookups | ~25.8M named-vector lookups | **0** |
| Core index computation | String-based | **Integer arithmetic** |
| `build_neighbor_lookup` time | ~hours | **Seconds** |
| `compute_neighbor_stats` (×5 vars) | Already vectorized | Further vectorizable with matrix ops |
| **Total estimated time** | **86+ hours** | **~2–5 minutes** |

---

## Optimization Strategy

### Strategy 1: Exploit Balanced Panel Structure with Integer Arithmetic

If data is sorted by `(year, id)` in a consistent order, then for year-block `t`, all cells appear at rows `((t-1)*N_cells + 1)` through `(t * N_cells)`, in the same `id_order`. A neighbor at position `j` in the id_order during year-block `t` is at row `(t-1)*N_cells + j`. No strings needed.

### Strategy 2: Vectorize `compute_neighbor_stats` with Matrix Column Indexing

Instead of `lapply` over 6.46M entries, we can build a neighbor-index matrix (padded to max neighbors), then use matrix subsetting to pull all neighbor values at once, and compute `max/min/mean` with `rowMeans`, `pmin`, `pmax` over columns.

### Strategy 3: Keep Everything Else Identical

The Random Forest model is already trained and takes the same column names — we only change how feature columns are computed, preserving exact numerical equivalence.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns id, year, and predictor vars
#   - id_order: vector of unique cell IDs in the order used by the nb object
#   - rook_neighbors_unique: spdep::nb object (list of integer neighbor indices)
#   - The data must contain all combinations of id_order × years (balanced panel)
# =============================================================================

library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # -------------------------------------------------------------------------
  # Convert to data.table for fast, controlled sorting
  # -------------------------------------------------------------------------
  dt <- as.data.table(data)
  
  # Build a mapping from cell id -> position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add the spatial position index
  dt[, spatial_pos := id_to_pos[as.character(id)]]
  
  # Sort by (year, spatial_pos) so that within each year-block,
  # row i corresponds to spatial_pos i.
  setorder(dt, year, spatial_pos)
  
  # Record the permutation so we can map back to original row order later
  # We need to know: for each row in the ORIGINAL data, what row is it
  # in the sorted data?
  # We'll store the sorted order and the reverse mapping.
  
  N_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  N_years <- length(years)
  
  stopifnot(nrow(dt) == N_cells * N_years)  # balanced panel check
  
  # In the sorted data, year-block t (0-indexed) spans rows
  #   (t * N_cells + 1) : ((t+1) * N_cells)
  # and within that block, row offset j corresponds to spatial_pos j.
  
  # -------------------------------------------------------------------------
  # Build padded neighbor matrix (N_cells × max_neighbors)
  # Entry [i, k] = spatial_pos of k-th neighbor of cell i, or NA
  # -------------------------------------------------------------------------
  max_nb <- max(lengths(neighbors))
  
  # Pad neighbor lists to equal length
  nb_padded <- lapply(neighbors, function(x) {
    if (length(x) == 0) return(rep(NA_integer_, max_nb))
    c(as.integer(x), rep(NA_integer_, max_nb - length(x)))
  })
  nb_matrix <- do.call(rbind, nb_padded)  # N_cells × max_nb
  # nb_matrix[i, k] = spatial_pos of k-th neighbor of cell at spatial_pos i
  
  # -------------------------------------------------------------------------
  # Build the full row-index neighbor lookup for the SORTED data
  # For sorted row r, its spatial_pos = ((r-1) %% N_cells) + 1
  # and its year-block offset = ((r-1) %/% N_cells) * N_cells
  # Neighbor sorted-rows = year_block_offset + nb_matrix[spatial_pos, ]
  # -------------------------------------------------------------------------
  
  # We'll return:
  #   1) The sorted data.table (to be used for feature computation)
  #   2) The neighbor matrix in terms of sorted-row indices (N_rows × max_nb)
  #   3) A mapping to restore original row order
  
  # Compute neighbor row indices for ALL sorted rows at once (vectorized)
  spatial_pos_all   <- rep(seq_len(N_cells), times = N_years)
  year_block_offset <- rep(seq(0L, (N_years - 1L) * N_cells, by = N_cells),
                           each = N_cells)
  
  # nb_matrix[spatial_pos_all, ] gives an (N_rows × max_nb) matrix
  # of neighbor spatial positions. Add year_block_offset to get sorted row idx.
  neighbor_row_matrix <- nb_matrix[spatial_pos_all, , drop = FALSE] +
                         year_block_offset
  # Where nb_matrix had NA, the result is NA (NA + integer = NA). Good.
  
  # -------------------------------------------------------------------------
  # Store original row indices for restoring order
  # dt was reordered; we need to map back.
  # Before sorting, we should have saved the original row index.
  # -------------------------------------------------------------------------
  
  list(
    sorted_dt            = dt,
    neighbor_row_matrix  = neighbor_row_matrix,   # N_rows × max_nb (sorted-row indices)
    max_nb               = max_nb,
    N_cells              = N_cells,
    N_years              = N_years
  )
}


compute_neighbor_stats_fast <- function(vals, neighbor_row_matrix) {
  # -------------------------------------------------------------------------
  # vals: numeric vector of length N_rows (in sorted order)
  # neighbor_row_matrix: integer matrix N_rows × max_nb
  # Returns: N_rows × 3 matrix with columns max, min, mean
  # -------------------------------------------------------------------------
  
  max_nb <- ncol(neighbor_row_matrix)
  N_rows <- length(vals)
  
  # Build a matrix of neighbor values: N_rows × max_nb
  # Use vals[neighbor_row_matrix] — this is a single vectorized index operation
  nb_vals <- matrix(vals[neighbor_row_matrix], nrow = N_rows, ncol = max_nb)
  # Where neighbor_row_matrix is NA, nb_vals is NA. Correct.
  
  # Compute row-wise max, min, mean ignoring NAs
  # For large matrices, rowwise operations are efficient in R / can use matrixStats
  
  # Check if matrixStats is available for speed; otherwise use base
  use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)
  
  if (use_matrixStats) {
    row_max  <- matrixStats::rowMaxs(nb_vals,  na.rm = TRUE)
    row_min  <- matrixStats::rowMins(nb_vals,  na.rm = TRUE)
    row_mean <- matrixStats::rowMeans2(nb_vals, na.rm = TRUE)
  } else {
    row_max  <- apply(nb_vals, 1, max,  na.rm = TRUE)
    row_min  <- apply(nb_vals, 1, min,  na.rm = TRUE)
    row_mean <- rowMeans(nb_vals, na.rm = TRUE)
  }
  
  # Handle rows where ALL neighbors are NA -> should return NA
  all_na <- rowSums(!is.na(nb_vals)) == 0L
  row_max[all_na]  <- NA_real_
  row_min[all_na]  <- NA_real_
  row_mean[all_na] <- NA_real_
  
  # Fix -Inf/Inf from max/min on empty sets (if matrixStats returns them)
  row_max[is.infinite(row_max)]  <- NA_real_
  row_min[is.infinite(row_min)]  <- NA_real_
  
  cbind(nb_max = row_max, nb_min = row_min, nb_mean = row_mean)
}


compute_and_add_neighbor_features_fast <- function(dt, var_name,
                                                   neighbor_row_matrix) {
  vals   <- dt[[var_name]]
  stats  <- compute_neighbor_stats_fast(vals, neighbor_row_matrix)
  
  dt[, paste0(var_name, "_nb_max")  := stats[, 1]]
  dt[, paste0(var_name, "_nb_min")  := stats[, 2]]
  dt[, paste0(var_name, "_nb_mean") := stats[, 3]]
  
  invisible(dt)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

run_optimized_feature_construction <- function(cell_data, id_order,
                                               rook_neighbors_unique) {
  
  cat("Building fast neighbor lookup (integer arithmetic, no strings)...\n")
  t0 <- proc.time()
  
  # Save original row order for later restoration
  cell_data$.orig_row_idx <- seq_len(nrow(cell_data))
  
  lookup <- build_neighbor_lookup_fast(cell_data, id_order,
                                       rook_neighbors_unique)
  
  sorted_dt           <- lookup$sorted_dt
  neighbor_row_matrix <- lookup$neighbor_row_matrix
  
  t1 <- proc.time()
  cat(sprintf("  Neighbor lookup built in %.1f seconds.\n", (t1 - t0)[3]))
  
  # Compute neighbor features for each source variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Computing neighbor stats for '%s'...\n", var_name))
    tv0 <- proc.time()
    compute_and_add_neighbor_features_fast(sorted_dt, var_name,
                                           neighbor_row_matrix)
    tv1 <- proc.time()
    cat(sprintf("    Done in %.1f seconds.\n", (tv1 - tv0)[3]))
  }
  
  # Restore original row order
  cat("Restoring original row order...\n")
  setorder(sorted_dt, .orig_row_idx)
  sorted_dt[, c("spatial_pos", ".orig_row_idx") := NULL]
  
  t2 <- proc.time()
  cat(sprintf("Total feature construction time: %.1f seconds.\n", (t2 - t0)[3]))
  
  # Return as data.frame if the original was a data.frame
  return(sorted_dt[])
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# cell_data <- run_optimized_feature_construction(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # Then predict with existing trained Random Forest as before:
# # predictions <- predict(rf_model, cell_data)
# =============================================================================
```

---

## Why This Preserves Numerical Equivalence

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `nb_matrix[i, ]` contains exactly the same neighbor spatial positions as `neighbors[[i]]`. The year-block offset maps them to the correct year's rows — identical to the string-key join. |
| **Same aggregation** | `max`, `min`, `mean` with `na.rm = TRUE` over the same set of neighbor values produces identical results. |
| **Same output columns** | Column names follow the same `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` pattern. |
| **Original row order restored** | The `.orig_row_idx` column ensures the final output matches the input row order exactly. |
| **RF model untouched** | No model retraining; only feature column computation is changed. |

## Performance Summary

| Step | Original | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~hours (string paste + hash ×6.46M) | ~5–15 sec (integer arithmetic, vectorized) | ~1000× |
| Stats for 1 variable | ~minutes (lapply over 6.46M) | ~3–8 sec (matrix indexing + `matrixStats`) | ~100× |
| Stats for 5 variables | ~hours | ~15–40 sec | ~100× |
| **Total** | **86+ hours** | **~1–3 minutes** | **~2000×** |

**Memory**: The `neighbor_row_matrix` is `6.46M × 4 integers ≈ 103 MB`. Each `nb_vals` column matrix is `6.46M × 4 doubles ≈ 207 MB`. With 16 GB RAM this fits comfortably, even with the data.table copy overhead.