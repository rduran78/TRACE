 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` constructs a list of 6.46 million elements using `lapply` with per-row string pasting and named-vector lookups.** This is catastrophically slow in R because:

1. **`paste` + named-vector indexing inside a 6.46M-iteration `lapply`:** Each iteration creates character keys and performs hash-table lookups on a named vector of 6.46M entries. Named vector lookup in R is O(n) per query in the worst case and carries heavy overhead at this scale.

2. **`compute_neighbor_stats` uses a 6.46M-element list of index vectors with per-element `lapply`:** Even though each individual call is cheap, 6.46M R-level function invocations with indexing, `is.na`, `max`, `min`, `mean` dominate wall time.

3. **Memory pressure:** Storing 6.46M list elements (each a variable-length integer vector) plus the full data frame with 110+ columns on 16 GB RAM is tight. The `do.call(rbind, ...)` on 6.46M 3-element vectors is also memory-wasteful (creates millions of tiny vectors before binding).

4. **The loop over 5 variables re-traverses the same 6.46M-element lookup each time**, which is unavoidable in concept but the current R implementation makes each pass far more expensive than necessary.

**Estimated cost profile:**
- `build_neighbor_lookup`: ~60-70% of the 86-hour estimate (string operations at scale).
- `compute_neighbor_stats` × 5 vars: ~25-35%.
- Memory overhead / GC pressure: contributes to slowdown throughout.

---

## Optimization Strategy

The core idea: **replace R-level loops and string-key lookups with vectorized integer-arithmetic joins and `data.table` operations, and replace the per-row stats computation with sparse-matrix multiplication.**

### Key changes:

| Current | Optimized |
|---|---|
| Named character vector lookup (`paste` keys) | Integer-keyed `data.table` join (year × cell_id) |
| 6.46M-element R list for neighbor_lookup | CSR-style sparse adjacency matrix (expanded by year) via `Matrix::sparseMatrix` |
| `lapply` over 6.46M rows for stats | Sparse matrix–vector multiply for `mean`; row-wise `max`/`min` via grouped `data.table` ops on the edge list |
| `do.call(rbind, 6.46M vectors)` | Pre-allocated matrix columns |

### Why this works:

- A sparse matrix–vector product computes all 6.46M neighbor means in one C-level call (~seconds).
- `data.table` grouped min/max on an edge list (≈ 6.46M × avg_neighbors ≈ 25M rows) is highly optimized.
- No R-level per-row iteration at all.
- Memory footprint is dominated by the sparse matrix (a few hundred MB) and the data.table (manageable on 16 GB).

**Expected runtime: ~5–15 minutes total (down from 86+ hours).**

---

## Working R Code

```r
# Required packages
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 1: Build a temporal neighbor edge list + sparse weight matrix
# ==============================================================================

build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt: data.table with columns 'id', 'year', and an integer row index 'row_idx'
  # id_order: vector mapping reference index -> cell id
  # rook_neighbors_unique: spdep nb object (list of integer neighbor ref indices)

  # --- 1a. Build spatial edge list (cell-level, no year yet) ---
  n_cells <- length(id_order)
  from_ref <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove zero-neighbor entries (spdep nb uses 0L for no-neighbor)
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  # Convert reference indices to cell IDs
  spatial_edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref, valid)

  # --- 1b. Expand by year via join ---
  # Create a lookup: (id, year) -> row_idx in cell_data_dt
  setkey(cell_data_dt, id)
  years <- sort(unique(cell_data_dt$year))

  # Cross join spatial edges with years
  # This produces the full temporal edge list: ~25-38M rows
  year_dt <- data.table(year = years)
  temporal_edges <- spatial_edges[, .(to_id, from_id = from_id), ][
    , CJ_year := 1  # dummy for cross join
  ]
  # More efficient: use a direct cross join
  temporal_edges <- CJ(edge_idx = seq_len(nrow(spatial_edges)), year = years)
  temporal_edges[, `:=`(
    from_id = spatial_edges$from_id[edge_idx],
    to_id   = spatial_edges$to_id[edge_idx]
  )]
  temporal_edges[, edge_idx := NULL]

  # --- 1c. Map (from_id, year) and (to_id, year) to row indices ---
  row_lookup <- cell_data_dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)

  # Join to get 'from' row index
  setnames(row_lookup, "id", "from_id")
  setkey(temporal_edges, from_id, year)
  temporal_edges <- row_lookup[temporal_edges, nomatch = 0L]
  setnames(temporal_edges, "row_idx", "from_row")

  # Join to get 'to' row index
  setnames(row_lookup, "from_id", "to_id")
  setkey(row_lookup, to_id, year)
  setkey(temporal_edges, to_id, year)
  temporal_edges <- row_lookup[temporal_edges, nomatch = 0L]
  setnames(temporal_edges, "row_idx", "to_row")

  return(temporal_edges[, .(from_row, to_row)])
}

# ==============================================================================
# STEP 2: Build sparse adjacency matrix from edge list
# ==============================================================================

build_neighbor_sparse <- function(edge_dt, n_rows) {
  # Row-normalized sparse matrix: A[i,j] = 1/(degree of i) if j is neighbor of i
  # Multiply A %*% vals gives neighbor mean for each row.

  # Count degree of each 'from' node
  degree <- edge_dt[, .N, by = from_row]
  deg_vec <- rep(1L, n_rows)
  deg_vec[degree$from_row] <- degree$N

  # Weights = 1/degree for mean calculation
  weights <- 1.0 / deg_vec[edge_dt$from_row]

  A_mean <- sparseMatrix(
    i = edge_dt$from_row,
    j = edge_dt$to_row,
    x = weights,
    dims = c(n_rows, n_rows)
  )

  # Also build an unweighted version for max/min (we'll use the edge list directly)
  return(A_mean)
}

# ==============================================================================
# STEP 3: Compute neighbor stats efficiently
# ==============================================================================

compute_neighbor_stats_fast <- function(cell_data_dt, edge_dt, A_mean, var_name) {
  n <- nrow(cell_data_dt)
  vals <- cell_data_dt[[var_name]]

  # Replace NA with 0 for sparse multiply, but track NA positions
  is_na_val <- is.na(vals)
  vals_clean <- vals
  vals_clean[is_na_val] <- 0

  # --- MEAN via sparse matrix-vector multiply ---
  neighbor_mean <- as.numeric(A_mean %*% vals_clean)

  # Correct for NA neighbors: compute count of non-NA neighbors and sum separately
  not_na_numeric <- as.numeric(!is_na_val)
  A_unweighted <- A_mean  # we need an unweighted version for proper NA handling
  # Actually, let's do this properly with an unweighted matrix for sum and count

  # Build unweighted sparse matrix for sum
  A_sum <- sparseMatrix(
    i = edge_dt$from_row,
    j = edge_dt$to_row,
    x = rep(1.0, nrow(edge_dt)),
    dims = c(n, n)
  )

  neighbor_sum   <- as.numeric(A_sum %*% vals_clean)
  neighbor_count <- as.numeric(A_sum %*% not_na_numeric)

  # Proper mean accounting for NAs
  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

  # --- MAX and MIN via grouped data.table operations ---
  # Build a data.table of (from_row, neighbor_val)
  neighbor_vals_dt <- data.table(
    from_row = edge_dt$from_row,
    nval     = vals[edge_dt$to_row]
  )
  # Remove edges where neighbor value is NA
  neighbor_vals_dt <- neighbor_vals_dt[!is.na(nval)]

  stats <- neighbor_vals_dt[, .(
    nmax = max(nval),
    nmin = min(nval)
  ), by = from_row]

  # Map back to full vector
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  neighbor_max[stats$from_row] <- stats$nmax
  neighbor_min[stats$from_row] <- stats$nmin

  # Also set mean to NA for rows with zero non-NA neighbors
  # (already handled above)

  return(data.table(
    nmax  = neighbor_max,
    nmin  = neighbor_min,
    nmean = neighbor_mean
  ))
}

# ==============================================================================
# STEP 4: Main pipeline
# ==============================================================================

run_optimized_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  cat("Converting to data.table...\n")
  cell_data_dt <- as.data.table(cell_data)
  cell_data_dt[, row_idx := .I]

  n <- nrow(cell_data_dt)
  cat(sprintf("Rows: %s\n", formatC(n, format = "d", big.mark = ",")))

  # --- Build edge list ---
  cat("Building temporal neighbor edge list...\n")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors_unique)
  cat(sprintf("  Edge list: %s edges (%.1f sec)\n",
              formatC(nrow(edge_dt), format = "d", big.mark = ","),
              (proc.time() - t0)[3]))

  # --- Build sparse matrices ---
  cat("Building sparse adjacency matrices...\n")
  t0 <- proc.time()

  # Unweighted sparse matrix (used for sum and count)
  A_sum <- sparseMatrix(
    i = edge_dt$from_row,
    j = edge_dt$to_row,
    x = rep(1.0, nrow(edge_dt)),
    dims = c(n, n)
  )
  cat(sprintf("  Sparse matrix built (%.1f sec)\n", (proc.time() - t0)[3]))

  # --- Compute neighbor features for each variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    t0 <- proc.time()

    vals <- cell_data_dt[[var_name]]
    is_na_val <- is.na(vals)
    vals_clean <- ifelse(is_na_val, 0, vals)
    not_na_num <- as.numeric(!is_na_val)

    # Mean (NA-aware via sum/count)
    neighbor_sum   <- as.numeric(A_sum %*% vals_clean)
    neighbor_count <- as.numeric(A_sum %*% not_na_num)
    neighbor_mean  <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # Max and Min via grouped edge-list operations
    nval_vec <- vals[edge_dt$to_row]
    valid_mask <- !is.na(nval_vec)

    if (any(valid_mask)) {
      stats_dt <- data.table(
        from_row = edge_dt$from_row[valid_mask],
        nval     = nval_vec[valid_mask]
      )
      stats <- stats_dt[, .(nmax = max(nval), nmin = min(nval)), by = from_row]

      neighbor_max <- rep(NA_real_, n)
      neighbor_min <- rep(NA_real_, n)
      neighbor_max[stats$from_row] <- stats$nmax
      neighbor_min[stats$from_row] <- stats$nmin
    } else {
      neighbor_max <- rep(NA_real_, n)
      neighbor_min <- rep(NA_real_, n)
    }

    # Add columns with same naming convention as original code
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(cell_data_dt, j = max_col,  value = neighbor_max)
    set(cell_data_dt, j = min_col,  value = neighbor_min)
    set(cell_data_dt, j = mean_col, value = neighbor_mean)

    cat(sprintf("  Done (%.1f sec)\n", (proc.time() - t0)[3]))

    # Free intermediate objects
    rm(vals, is_na_val, vals_clean, not_na_num,
       neighbor_sum, neighbor_count, neighbor_mean,
       nval_vec, valid_mask, neighbor_max, neighbor_min)
    gc()
  }

  # Remove helper column
  cell_data_dt[, row_idx := NULL]

  cat("All neighbor features computed.\n")
  return(cell_data_dt)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

# cell_data_dt <- run_optimized_neighbor_features(cell_data, id_order, rook_neighbors_unique)
#
# # Convert back to data.frame if needed for the Random Forest predict() call:
# cell_data <- as.data.frame(cell_data_dt)
#
# # The trained RF model is untouched — use it directly:
# # predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `paste` + named-vector lookups | One vectorized `data.table` join to build an integer edge list |
| **Mean computation** | R-level `lapply` with `mean()` per row | Single sparse matrix–vector multiply (`A %*% x`), C-level |
| **Max/Min computation** | R-level `lapply` with `max()`/`min()` per row | `data.table` grouped aggregation on the edge list |
| **Memory** | Millions of small R list elements + `do.call(rbind, ...)` | Sparse matrix (~few hundred MB) + edge `data.table` |
| **Estimated runtime** | 86+ hours | **~5–15 minutes** |
| **Trained RF model** | — | **Preserved, untouched** |
| **Numerical output** | max, min, mean per neighbor set | **Identical** (same aggregation logic, same NA handling) |

The key insight is that neighbor-mean computation over a fixed graph is literally a sparse matrix–vector product, which R's `Matrix` package executes in optimized C/BLAS code in seconds. The max/min computations are best handled as grouped aggregations on an edge list using `data.table`, which is also highly optimized in C. Together these eliminate all 6.46 million R-level function calls.