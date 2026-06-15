 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookups in R are O(n) hash probes per call, and doing this 6.46M times with string concatenation is extremely slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting and aggregating neighbor values one node-year at a time. This is pure R-level looping with no vectorization.

3. **The topology is year-invariant but is being resolved per cell-year row.** Every cell has the same rook neighbors every year, so the 344,208-cell adjacency structure is needlessly replicated 28 times into a 6.46M-element lookup.

**Root cause:** The design entangles the spatial topology (344K nodes, ~1.37M edges) with the temporal panel (28 years), producing a 6.46M-node "graph" when the actual graph has only 344K nodes. All neighbor aggregation can be expressed as sparse matrix–vector products, which are highly optimized in C via the `Matrix` package.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 × 344,208, ~1.37M nonzeros). This is the graph topology.
2. **Reshape each variable into a 344,208 × 28 matrix** (rows = cells, columns = years).
3. **Compute neighbor aggregates via sparse matrix operations:**
   - **Mean:** `A_row_normalized %*% X` (one sparse matrix multiply per variable).
   - **Max / Min:** Use a CSC/CSR walk or `data.table` grouped operations on the edge list — but the most efficient pure-R approach is to convert the sparse matrix to an edge list once and use `data.table` grouping.
4. **Flatten results back** into the long panel and bind columns.
5. **Predict** with the pre-trained Random Forest.

This reduces runtime from 86+ hours to **minutes** by eliminating all R-level per-row iteration and leveraging compiled sparse linear algebra and `data.table` grouped aggregation.

## Working R Code

```r
# =============================================================================
# Optimized spatial‐neighbor feature engineering
# Preserves numerical equivalence with the original pipeline
# =============================================================================

library(Matrix)
library(data.table)
library(spdep)      # for nb2listw if needed

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars, rf_model = NULL) {

  # -----------------------------------------------------------
  # 0. Convert to data.table for speed; keep original row order
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))

  # -----------------------------------------------------------
  # 1. Build sparse adjacency matrix ONCE (344,208 x 344,208)
  #    from the nb object. This encodes the rook graph topology.
  # -----------------------------------------------------------
  cat("Building sparse adjacency matrix from nb object...\n")

  # Map cell IDs to integer indices 1..n_cells
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Build COO (coordinate) representation of adjacency
  from_list <- vector("list", n_cells)
  to_list   <- vector("list", n_cells)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[0] == 0L)) {
      # nb objects use 0 to indicate no neighbors
      nb_i <- nb_i[nb_i != 0L]
      if (length(nb_i) > 0) {
        from_list[[i]] <- rep.int(i, length(nb_i))
        to_list[[i]]   <- nb_i
      }
    }
  }
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)
  rm(from_list, to_list)

  n_edges <- length(from_vec)
  cat(sprintf("Directed edges in adjacency: %d\n", n_edges))

  # Sparse adjacency matrix (binary)
  A <- sparseMatrix(
    i    = from_vec,
    j    = to_vec,
    x    = rep.int(1, n_edges),
    dims = c(n_cells, n_cells)
  )

  # Row-degree for computing mean (number of neighbors per cell)
  deg <- rowSums(A)  # integer vector length n_cells

  # Row-normalized adjacency for mean computation
  # D^{-1} A where D = diag(deg); handle zero-degree nodes
  deg_inv <- ifelse(deg > 0, 1 / deg, 0)
  A_mean  <- Diagonal(x = deg_inv) %*% A   # still sparse

  # -----------------------------------------------------------
  # 2. Build edge-list data.table for max/min (grouped aggregation)
  # -----------------------------------------------------------
  edge_dt <- data.table(from = from_vec, to = to_vec)
  rm(from_vec, to_vec)
  gc()

  # -----------------------------------------------------------
  # 3. Create cell-index column in dt for fast matrix indexing
  #    Map each row's cell ID to its position in id_order
  # -----------------------------------------------------------
  dt[, cell_idx := id_to_idx[as.character(id)]]

  # Ensure data is sorted by (cell_idx, year) for matrix reshaping

  setkey(dt, cell_idx, year)

  # -----------------------------------------------------------
  # 4. For each variable, compute neighbor max, min, mean
  # -----------------------------------------------------------
  cat("Computing neighbor statistics...\n")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))

    # 4a. Pivot to matrix: rows = cells (1..n_cells), cols = years
    #     Some cells may be missing for some years; handle via NA
    X_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    year_to_col <- setNames(seq_along(years), as.character(years))

    # Vectorized fill
    row_indices <- dt$cell_idx
    col_indices <- year_to_col[as.character(dt$year)]
    X_mat[cbind(row_indices, col_indices)] <- dt[[var_name]]

    # ---------------------------------------------------------
    # 4b. MEAN via sparse matrix multiply: M = A_mean %*% X_mat
    #     This gives the mean of neighbor values for each cell-year.
    #     Cells with zero neighbors get 0; we'll fix to NA below.
    # ---------------------------------------------------------
    # Replace NA with 0 for matrix multiply, but track NA counts
    X_notna  <- ifelse(is.na(X_mat), 0, 1)
    X_filled <- ifelse(is.na(X_mat), 0, X_mat)

    # Number of non-NA neighbors per cell-year
    N_valid <- as.matrix(A %*% X_notna)   # n_cells x n_years
    # Sum of non-NA neighbor values
    S_vals  <- as.matrix(A %*% X_filled)  # n_cells x n_years

    # Mean = sum / count (NA where count == 0)
    mean_mat <- ifelse(N_valid > 0, S_vals / N_valid, NA_real_)

    rm(S_vals)

    # ---------------------------------------------------------
    # 4c. MAX and MIN via edge-list grouped aggregation
    #     For each year-column, look up neighbor values and aggregate.
    #     This is vectorized over all edges simultaneously.
    # ---------------------------------------------------------
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (yy in seq_len(n_years)) {
      # Get values for all "to" (neighbor) nodes in this year
      neighbor_vals <- X_mat[edge_dt$to, yy]

      # Build temporary DT with from-node and neighbor value
      tmp <- data.table(
        from = edge_dt$from,
        val  = neighbor_vals
      )
      # Remove edges where neighbor value is NA
      tmp <- tmp[!is.na(val)]

      if (nrow(tmp) > 0) {
        agg <- tmp[, .(mx = max(val), mn = min(val)), by = from]
        max_mat[agg$from, yy] <- agg$mx
        min_mat[agg$from, yy] <- agg$mn
      }
    }

    rm(X_mat, X_filled, X_notna, N_valid)

    # ---------------------------------------------------------
    # 4d. Flatten matrices back to long panel and attach to dt
    # ---------------------------------------------------------
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)

    dt[, (max_col_name)  := max_mat[cbind(cell_idx, year_to_col[as.character(year)])]]
    dt[, (min_col_name)  := min_mat[cbind(cell_idx, year_to_col[as.character(year)])]]
    dt[, (mean_col_name) := mean_mat[cbind(cell_idx, year_to_col[as.character(year)])]]

    rm(max_mat, min_mat, mean_mat)
    gc()
  }

  # -----------------------------------------------------------
  # 5. Restore original row order and clean up
  # -----------------------------------------------------------
  setorder(dt, .row_order)
  dt[, c("cell_idx", ".row_order") := NULL]

  cat("Neighbor feature computation complete.\n")

  # -----------------------------------------------------------
  # 6. Predict with pre-trained Random Forest (if provided)
  # -----------------------------------------------------------
  if (!is.null(rf_model)) {
    cat("Generating predictions with pre-trained Random Forest...\n")
    dt[, prediction := predict(rf_model, newdata = dt)]
  }

  return(as.data.frame(dt))
}

# =============================================================================
# Usage
# =============================================================================
# result <- optimize_neighbor_features(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   rf_model               = trained_rf_model
# )
```

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `max(vals[neighbor_indices])` | `data.table` grouped `max` over edge list per year | Identical: same set of non-NA neighbor values, same `max` function |
| `min(vals[neighbor_indices])` | `data.table` grouped `min` over edge list per year | Identical |
| `mean(vals[neighbor_indices])` | `sum(non-NA neighbor vals) / count(non-NA neighbor vals)` via sparse matrix multiply | Identical: `A %*% X_filled / A %*% X_notna` equals `mean` of non-NA neighbors |
| NA handling: skip NA neighbors, return NA if zero valid neighbors | Replicated exactly via `N_valid > 0` guard and `!is.na(val)` filter | Identical |

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| Topology resolution | 6.46M string-paste + named lookups | One sparse matrix build (344K × 344K, 1.37M nnz) |
| Mean computation | 6.46M R-level `lapply` iterations | 5 sparse matrix multiplies (~0.5s each) |
| Max/Min computation | 6.46M R-level loops | 5 vars × 28 years = 140 `data.table` grouped aggregations on ~1.37M-row edge list |
| Memory | ~6.46M-element list of integer vectors | ~50 MB sparse matrix + ~77 MB per variable matrix (344K × 28) |
| **Estimated time** | **86+ hours** | **~5–15 minutes** |

The key insight: **the graph topology is time-invariant**. By separating the spatial adjacency (sparse matrix, built once) from the temporal panel (matrix columns), we convert millions of R-level iterations into a handful of sparse linear algebra operations and vectorized `data.table` aggregations — all executed in compiled C/C++ code.