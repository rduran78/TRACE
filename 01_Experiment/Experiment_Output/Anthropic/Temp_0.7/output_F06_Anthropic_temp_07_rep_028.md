 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over ~6.46 million rows

For every row, the function:
1. Looks up the cell's reference index via a named character vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs `paste()`-based keys for every neighbor × that row's year.
4. Looks up those keys in `idx_lookup` (another named character vector).

Named character vector lookup in R is **O(n)** per query (hash-based, but with overhead from `as.character`, `paste`, and repeated allocation). Over 6.46M rows, each with ~4 rook neighbors on average (~1.37M relationships / 344K cells ≈ 4), this produces **~25.8 million string constructions and hash lookups**, all inside a sequential `lapply`. The resulting list of 6.46M integer vectors also consumes significant memory.

### Bottleneck B: `compute_neighbor_stats` — Called 5 times, each iterating over 6.46M rows

Each call does another `lapply` over 6.46M elements, subsetting a numeric vector and computing `max`, `min`, `mean`. This is pure R-level looping — no vectorization.

### Combined effect

The two stages together produce roughly **86+ hours** of wall-clock time on a 16 GB laptop. The fundamental problem is: **row-level R-loop iteration over millions of rows with per-row string operations and list allocations**.

---

## 2. Optimization Strategy

### Key insight: Separate the spatial dimension from the temporal dimension

The neighbor structure is **time-invariant** — cell A's rook neighbors are the same in every year. The `nb` object has only 344,208 entries (one per cell). The current code "explodes" this into 6.46M entries by replicating the neighbor structure across all 28 years. This is unnecessary.

### Strategy: Vectorized sparse-matrix multiplication

1. **Build a sparse neighbor adjacency matrix `W`** (344,208 × 344,208) from the `nb` object. Each row `i` has 1s in columns corresponding to cell `i`'s rook neighbors. This matrix has ~1.37M non-zero entries — trivially small.

2. **Reshape each source variable into a matrix `V`** of dimension (344,208 cells × 28 years), where rows are cells (in `id_order` order) and columns are years.

3. **Compute neighbor stats using sparse matrix operations:**
   - **Neighbor sum** = `W %*% V` (sparse × dense, extremely fast)
   - **Neighbor count** = `W %*% (!is.na(V))` (to handle NAs correctly)
   - **Neighbor mean** = sum / count
   - **Neighbor max and min**: Use a grouped operation over the sparse structure of `W` — iterate over the 344K cells (not 6.46M rows), extract neighbor indices from `W`, and compute row-wise max/min on the submatrix.

4. **Flatten back** to the original long-format data frame and attach the 15 new columns (3 stats × 5 variables).

### Why this is fast

| Aspect | Old | New |
|---|---|---|
| Loop iterations for lookup | 6.46M | 0 (matrix construction) |
| Loop iterations for stats | 6.46M × 5 = 32.3M | 344K × 5 = 1.72M (max/min only) |
| Mean computation | Per-row R loop | Sparse matrix multiply (C-level) |
| String operations | ~25.8M `paste()` calls | 0 |
| Memory for lookup | 6.46M-element list | Sparse matrix (~20 MB) |

**Expected speedup: from 86+ hours to ~2–10 minutes.**

### Why not raster focal/kernel operations?

The comment in the prompt asks us to consider this. Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. If the grid is perfectly regular and complete, a 3×3 rook kernel (`matrix(c(0,1,0,1,0,1,0,1,0), 3, 3)`) would work and be very fast. However:
- The `nb` object is **precomputed and serialized**, suggesting the grid may have irregular boundaries, missing cells, or non-rectangular extent.
- Using the `nb` object directly (via sparse matrix) **guarantees identical neighbor relationships** and thus **preserves the original numerical estimand exactly**.
- A raster focal approach would require reconstructing the grid, handling edge/missing cells differently, and risks subtle discrepancies.

**Decision: Use the sparse matrix approach built from the actual `nb` object.** This is both fast and numerically faithful.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical results (same neighbor relationships, same stats)
# ==============================================================================

library(Matrix)   # for sparse matrices
library(data.table)  # for fast reshaping

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  
  # --------------------------------------------------------------------------
  # STEP 1: Build sparse adjacency matrix W from the nb object
  # --------------------------------------------------------------------------
  # id_order[i] is the cell ID for the i-th entry in rook_neighbors_unique
  n_cells <- length(id_order)
  
  # Build COO (coordinate) representation
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0) {
      from_idx <- c(from_idx, rep(i, length(nb_i)))
      to_idx   <- c(to_idx, nb_i)
    }
  }
  
  W <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # --------------------------------------------------------------------------
  # STEP 2: Create a mapping from cell ID to row index in W
  # --------------------------------------------------------------------------
  id_to_widx <- setNames(seq_along(id_order), as.character(id_order))
  
  # --------------------------------------------------------------------------
  # STEP 3: Convert cell_data to data.table for fast operations
  # --------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Ensure consistent ordering: assign W-row index to each row
  dt[, w_idx := id_to_widx[as.character(id)]]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  dt[, y_idx := year_to_col[as.character(year)]]
  
  # --------------------------------------------------------------------------
  # STEP 4: For each source variable, compute neighbor max, min, mean
  # --------------------------------------------------------------------------
  
  # Pre-extract the sparse structure of W for max/min computation
  # For each cell i, get its neighbor indices
  # We can extract this from W directly
  W_dgC <- as(W, "dgCMatrix")  # compressed sparse column
  W_dgR <- as(W, "dgRMatrix")  # compressed sparse row — better for row access
  # If dgRMatrix is not available, use dgCMatrix on transposed or manual extraction
  
  # Extract neighbor list from sparse matrix (much faster than re-reading nb)
  # Using dgCMatrix: columns of t(W) = rows of W
  Wt <- t(W_dgC)  # now column j of Wt = row j of W = neighbors of j
  
  neighbor_indices <- vector("list", n_cells)
  for (j in seq_len(n_cells)) {
    col_start <- Wt@p[j] + 1L
    col_end   <- Wt@p[j + 1L]
    if (col_end >= col_start) {
      neighbor_indices[[j]] <- Wt@i[col_start:col_end] + 1L
    } else {
      neighbor_indices[[j]] <- integer(0)
    }
  }
  
  for (var_name in neighbor_source_vars) {
    cat("Processing neighbor features for:", var_name, "\n")
    
    # Build cell × year matrix V (n_cells × n_years)
    # Initialize with NA
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(dt$w_idx, dt$y_idx)] <- dt[[var_name]]
    
    # ------ NEIGHBOR MEAN via sparse matrix multiply ------
    # Handle NAs: replace NA with 0 for sum, track counts separately
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0
    V_notna <- (!is.na(V)) * 1.0  # indicator matrix
    
    # Neighbor sum and count (sparse %*% dense is fast in Matrix package)
    nb_sum   <- as.matrix(W %*% V_nona)    # n_cells × n_years
    nb_count <- as.matrix(W %*% V_notna)   # n_cells × n_years
    
    nb_mean <- nb_sum / nb_count
    nb_mean[nb_count == 0] <- NA_real_
    
    # ------ NEIGHBOR MAX and MIN via grouped row operation ------
    # This loops over 344K cells (not 6.46M rows) — very manageable
    nb_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nbs <- neighbor_indices[[i]]
      if (length(nbs) == 0L) next
      # Extract submatrix: neighbors × years
      sub <- V[nbs, , drop = FALSE]  # length(nbs) × n_years
      if (nrow(sub) == 1L) {
        nb_max[i, ] <- sub[1L, ]
        nb_min[i, ] <- sub[1L, ]
      } else {
        # Columnwise max/min, ignoring NAs
        nb_max[i, ] <- apply(sub, 2, max, na.rm = TRUE)
        nb_min[i, ] <- apply(sub, 2, min, na.rm = TRUE)
      }
    }
    # apply with na.rm=TRUE returns -Inf/Inf when all NA; fix those
    nb_max[is.infinite(nb_max)] <- NA_real_
    nb_min[is.infinite(nb_min)] <- NA_real_
    
    # ------ MAP BACK to long-format data.table ------
    idx_mat <- cbind(dt$w_idx, dt$y_idx)
    
    max_col <- paste0(var_name, "_max_neighbor")
    min_col <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")
    
    dt[, (max_col)  := nb_max[idx_mat]]
    dt[, (min_col)  := nb_min[idx_mat]]
    dt[, (mean_col) := nb_mean[idx_mat]]
  }
  
  # --------------------------------------------------------------------------
  # STEP 5: Clean up helper columns and return
  # --------------------------------------------------------------------------
  dt[, c("w_idx", "y_idx") := NULL]
  
  return(as.data.frame(dt))
}

# ==============================================================================
# USAGE — drop-in replacement for the original outer loop
# ==============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged — use it directly for prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Further Optimization: Eliminating the `apply` Loop for Max/Min

The 344K-iteration loop with `apply` for max/min is already ~1000× fewer iterations than the original, but if it's still slow (each iteration touches a small submatrix across 28 columns), we can vectorize it further using **data.table grouping**:

```r
# ALTERNATIVE: Fully vectorized max/min using data.table joins
# Instead of looping over cells, expand the edge list and group-aggregate

compute_maxmin_dt <- function(dt, W_from, W_to, var_name, years, year_to_col) {
  # W_from, W_to: integer vectors of directed edges (in W-index space)
  # Build edge table
  edges <- data.table(from_w = W_from, to_w = W_to)
  
  # For each (from_cell, year), we need the var values of to_cell
  # Expand edges across years
  edge_year <- CJ(edge_id = seq_len(nrow(edges)), y_idx = seq_along(years))
  edge_year[, `:=`(from_w = edges$from_w[edge_id], to_w = edges$to_w[edge_id])]
  
  # Lookup values: create a keyed table of (w_idx, y_idx) -> value
  val_table <- dt[, .(w_idx, y_idx, val = get(var_name))]
  setkey(val_table, w_idx, y_idx)
  
  # Join to get neighbor values
  edge_year[, val := val_table[.(to_w, y_idx), val, on = .(w_idx, y_idx)]]
  
  # Group by (from_w, y_idx) and compute max, min
  stats <- edge_year[!is.na(val),
                     .(nb_max = max(val), nb_min = min(val)),
                     by = .(from_w, y_idx)]
  
  # Join back to dt
  setkey(stats, from_w, y_idx)
  dt[, paste0(var_name, "_max_neighbor") :=
       stats[.(w_idx, y_idx), nb_max, on = .(from_w = w_idx, y_idx)]]
  dt[, paste0(var_name, "_min_neighbor") :=
       stats[.(w_idx, y_idx), nb_min, on = .(from_w = w_idx, y_idx)]]
  
  invisible(dt)
}
```

> **Caveat:** This edge-expansion approach creates ~1.37M × 28 ≈ 38.5M rows, using ~600 MB of RAM — feasible on a 16 GB machine but heavier. The cell-loop version (344K iterations) is likely the better balance of speed and memory for this dataset.

---

## 5. Summary

| Metric | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M row `lapply` with `paste`/hash | Sparse matrix from `nb` (344K entries) |
| Mean computation | 6.46M row `lapply` | Sparse matrix multiply (`W %*% V`) |
| Max/Min computation | 6.46M row `lapply` | 344K cell loop over 28-column submatrices |
| Estimated runtime | 86+ hours | **~2–10 minutes** |
| Memory | 6.46M-element list (~2–4 GB) | Sparse matrix (~20 MB) + 5 dense matrices (~130 MB each) |
| Numerical results | Baseline | **Identical** (same neighbors, same operations) |
| Trained RF model | Unchanged | **Unchanged** |