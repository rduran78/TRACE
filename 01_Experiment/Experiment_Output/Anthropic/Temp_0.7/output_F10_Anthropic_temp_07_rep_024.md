 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). String operations on millions of keys are extremely slow in R. The lookup is O(n) per row with expensive character matching.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements** in a serial `lapply`, extracting subsets of a numeric vector for each node-year. This is repeated 5 times (once per variable), totaling ~32.3 million list iterations.

3. **The neighbor topology is year-invariant but the lookup conflates topology and time.** Every cell has the same rook neighbors every year, yet the lookup rebuilds index references per cell-year row, inflating the problem by 28×.

**Memory:** Storing 6.46M list entries (each a small integer vector) consumes substantial RAM and causes GC pressure. String keys (`paste(id, year)`) add further overhead.

**Net effect:** ~86+ hours on a 16 GB laptop.

## Optimization Strategy

1. **Separate topology from time.** Build a sparse adjacency structure once over 344,208 cells (not 6.46M cell-years). Rook neighbors don't change across years.

2. **Use sparse matrix–dense matrix multiplication for aggregation.** Construct a row-normalized (or raw) sparse adjacency matrix `A` (344,208 × 344,208) with ~1.37M non-zero entries. For each year, extract the variable column as a dense vector, then:
   - `A %*% x` gives neighbor sums; dividing by neighbor counts gives **mean**.
   - For **max** and **min**, use grouped operations via `data.table` with the edge list.

3. **Use `data.table` throughout** for fast grouped operations, avoiding `lapply` over millions of rows.

4. **Process year-by-year within the sparse matrix framework** to keep memory bounded, or process all years at once via a reshaped matrix if RAM allows.

5. **Preserve numerical equivalence:** The same neighbors, same values, same `max`/`min`/`mean` — just computed via vectorized/matrix operations.

## Optimized R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_pipeline <- function(cell_data_df, id_order, rook_neighbors_unique,
                                        neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                  "def", "usd_est_n2")) {

  # ──────────────────────────────────────────────────────────────

  # 1. Convert to data.table and set up cell/year indexing

  # ──────────────────────────────────────────────────────────────
  cat("Step 1: Setting up data.table and indexing...\n")
  dt <- as.data.table(cell_data_df)

  # Ensure id_order is a plain vector
  id_order <- as.vector(id_order)
  n_cells  <- length(id_order)

  # Map cell IDs to integer indices 1..n_cells
  id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))

  # Add cell index to dt
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Sort by cell_idx and year for fast matrix reshaping
  setkey(dt, cell_idx, year)

  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))

  cat(sprintf("  %d cells, %d years, %d rows\n", n_cells, n_years, nrow(dt)))

  # ──────────────────────────────────────────────────────────────
  # 2. Build sparse adjacency edge list from nb object (once)
  # ──────────────────────────────────────────────────────────────
  cat("Step 2: Building edge list from nb object...\n")

  # rook_neighbors_unique is a list of length n_cells;

  # rook_neighbors_unique[[i]] contains integer indices (into id_order)
  # of neighbors of cell i.
  from_vec <- integer(0)
  to_vec   <- integer(0)

  # Pre-count edges for allocation
  edge_counts <- vapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb != 0L]  # spdep nb uses 0 for no-neighbor regions
    length(nb)
  }, integer(1))
  total_edges <- sum(edge_counts)

  cat(sprintf("  Total directed edges: %d\n", total_edges))

  from_vec <- integer(total_edges)
  to_vec   <- integer(total_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- rook_neighbors_unique[[i]]
    nb <- nb[nb != 0L]
    k  <- length(nb)
    if (k > 0L) {
      from_vec[pos:(pos + k - 1L)] <- i
      to_vec[pos:(pos + k - 1L)]   <- nb
      pos <- pos + k
    }
  }

  edge_dt <- data.table(from = from_vec, to = to_vec)
  rm(from_vec, to_vec)

  # ──────────────────────────────────────────────────────────────
  # 3. Build sparse matrix for mean (neighbor sum / neighbor count)
  # ──────────────────────────────────────────────────────────────
  cat("Step 3: Building sparse adjacency matrix...\n")

  # Sparse adjacency matrix: A[i,j] = 1 if j is a neighbor of i
  A <- sparseMatrix(
    i = edge_dt$from,
    j = edge_dt$to,
    x = rep(1, nrow(edge_dt)),
    dims = c(n_cells, n_cells)
  )

  # Neighbor counts per cell (for mean calculation)
  neighbor_counts <- as.vector(rowSums(A))  # integer-valued

  # ──────────────────────────────────────────────────────────────
  # 4. For each variable, compute max/min/mean across neighbors
  #    Strategy:
  #      - Reshape variable into n_cells × n_years matrix
  #      - MEAN: sparse mat-mul A %*% X / counts
  #      - MAX/MIN: grouped operations on edge_dt
  # ──────────────────────────────────────────────────────────────

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Step 4: Processing variable '%s'...\n", var_name))

    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # 4a. Build n_cells × n_years matrix of variable values
    #     dt is keyed by (cell_idx, year), so we can reshape efficiently
    vals <- dt[[var_name]]

    # Create matrix: rows = cell_idx, cols = year index
    # dt is sorted by (cell_idx, year), so if balanced panel:
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[cbind(dt$cell_idx, year_to_col[as.character(dt$year)])] <- vals

    # 4b. MEAN via sparse matrix multiplication
    #     neighbor_sum = A %*% X  (n_cells × n_years)
    #     neighbor_mean = neighbor_sum / neighbor_counts
    cat("    Computing mean via sparse mat-mul...\n")
    neighbor_sum <- as.matrix(A %*% X)  # n_cells × n_years dense

    # Handle NA propagation: we need mean of non-NA neighbors
    # Count non-NA neighbors per cell-year
    X_notna <- !is.na(X)
    X_zero  <- X
    X_zero[is.na(X_zero)] <- 0

    neighbor_sum_nona  <- as.matrix(A %*% X_zero)
    neighbor_count_nona <- as.matrix(A %*% (X_notna * 1.0))

    mean_mat <- neighbor_sum_nona / neighbor_count_nona
    mean_mat[neighbor_count_nona == 0] <- NA_real_

    # 4c. MAX and MIN via grouped edge-list operations
    cat("    Computing max/min via edge-list grouping...\n")

    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    # Process year by year to keep memory bounded
    for (yi in seq_along(years)) {
      # Values of neighbors for this year: for each edge (from, to),
      # the neighbor value is X[to, yi]
      neighbor_vals <- X[edge_dt$to, yi]

      # Build temporary data.table for grouped max/min
      tmp <- data.table(from = edge_dt$from, val = neighbor_vals)
      tmp <- tmp[!is.na(val)]

      if (nrow(tmp) > 0L) {
        agg <- tmp[, .(mx = max(val), mn = min(val)), by = from]
        max_mat[agg$from, yi] <- agg$mx
        min_mat[agg$from, yi] <- agg$mn
      }
    }

    # 4d. Write results back to dt
    cat("    Writing results back...\n")
    dt[, (max_col)  := max_mat[cbind(cell_idx, year_to_col[as.character(year)])]]
    dt[, (min_col)  := min_mat[cbind(cell_idx, year_to_col[as.character(year)])]]
    dt[, (mean_col) := mean_mat[cbind(cell_idx, year_to_col[as.character(year)])]]

    # Free per-variable matrices
    rm(X, X_zero, X_notna, neighbor_sum, neighbor_sum_nona,
       neighbor_count_nona, mean_mat, max_mat, min_mat)
    gc()
  }

  # ──────────────────────────────────────────────────────────────
  # 5. Clean up and return
  # ──────────────────────────────────────────────────────────────
  dt[, cell_idx := NULL]
  cat("Done.\n")
  return(dt)
}


# ════════════════════════════════════════════════════════════════
# USAGE
# ════════════════════════════════════════════════════════════════

# # Load data
# load("cell_data.RData")        # -> cell_data (data.frame)
# load("rook_neighbors.RData")   # -> rook_neighbors_unique (nb list), id_order
# load("rf_model.RData")         # -> rf_model (pre-trained randomForest)
#
# # Run optimized pipeline
# cell_data_dt <- optimize_neighbor_pipeline(
#   cell_data_df          = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars  = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Predict with the pre-trained RF (no retraining)
# # Ensure column order matches training data
# pred_vars <- names(rf_model$forest$xlevels)  # or stored feature names
# # If rf_model was trained via randomForest:
# # pred_vars can be extracted from rf_model$importance rownames
# pred_vars <- rownames(rf_model$importance)
#
# cell_data_dt[, prediction := predict(rf_model, newdata = .SD), .SDcols = pred_vars]
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | One sparse matrix build (344K × 344K, 1.37M nnz) |
| **Mean computation** | 6.46M `lapply` iterations per variable | Single sparse `A %*% X` matrix multiply per variable |
| **Max/Min computation** | 6.46M `lapply` with subsetting per variable | 28 `data.table` grouped aggregations per variable (1.37M rows each) |
| **Total variable passes** | 5 × 6.46M = 32.3M list iterations | 5 × (1 matmul + 28 grouped aggs) |
| **Memory pattern** | ~6.46M list entries + string keys (~4–8 GB) | Sparse matrix (~30 MB) + dense 344K×28 matrices (~77 MB each) |
| **Estimated time** | 86+ hours | **~5–15 minutes** |

### Why This Is Numerically Equivalent

- **Mean:** `A %*% X_zero / A %*% notNA(X)` computes exactly `sum(non-NA neighbor values) / count(non-NA neighbor values)` — identical to the original `mean(neighbor_vals)` after NA removal.
- **Max/Min:** The edge-list grouped aggregation computes `max`/`min` over exactly the same set of neighbor values (excluding NAs) as the original code.
- **Topology:** The sparse matrix and edge list are built from the same `rook_neighbors_unique` nb object with the same `id_order` mapping.
- **The pre-trained Random Forest is loaded and used for `predict()` only — never retrained.**