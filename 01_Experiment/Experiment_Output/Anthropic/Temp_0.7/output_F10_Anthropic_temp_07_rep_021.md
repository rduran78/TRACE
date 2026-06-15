 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates a per-row (6.46M) list of neighbor row indices.** It iterates over every row, performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`) for each. This is O(N × avg_neighbors) with very high constant factors due to R's character hashing on millions of keys. The resulting list-of-vectors also consumes enormous memory (~6.46M list elements).

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`.** For each node-year, it subsets a numeric vector by indices, removes NAs, and computes max/min/mean. The per-element R function-call overhead dominates — this is called 5 × 6.46M ≈ 32.3M times.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are spatial — cell *i*'s neighbors are the same cells every year. Yet the code re-resolves neighbor relationships at the row level (cell×year), inflating a ~1.37M-edge spatial graph into a ~6.46M-element row-level lookup. This is a 28× redundancy.

**Root cause:** The code treats a *panel* problem as a flat-table problem, losing the separability of space and time.

---

## Optimization Strategy

**Key insight:** Separate the spatial topology (344K cells, ~1.37M edges, static) from the temporal attributes (28 years, varying). Then:

1. **Build the spatial adjacency structure once** as a sparse matrix or integer-indexed edge list over *cell indices* (not row indices). This is small: 344K nodes, ~1.37M edges.

2. **Reshape each variable into a cell × year matrix** (344,208 rows × 28 columns). This allows vectorized column-wise (i.e., year-wise) operations.

3. **Compute neighbor aggregates via sparse matrix multiplication.** For a variable matrix `V` (cells × years):
   - `neighbor_max[i, t]` = max of `V[j, t]` over neighbors `j` of `i`
   - `neighbor_min[i, t]` = min of `V[j, t]` over neighbors `j` of `i`  
   - `neighbor_mean[i, t]` = mean of `V[j, t]` over neighbors `j` of `i`

   **Mean** is directly computable as a sparse matrix–dense matrix product: `A %*% V / degree_vector`, where `A` is the binary adjacency matrix.

   **Max and min** require a grouped row-wise operation over the sparse structure, but can be done efficiently in C++ via `Rcpp` or by a vectorized edge-list approach: expand all (source, target, year) triples, fetch target values, then `group-by source+year` and aggregate. With `data.table`, this is extremely fast.

4. **Melt results back** to the original long panel format and column-bind.

**Expected speedup:** From 86+ hours to **~2–10 minutes** on 16 GB RAM. The dominant cost becomes a few sparse-matrix multiplications and grouped aggregations over ~1.37M × 28 ≈ 38.4M edge-year records — fully vectorized.

**Numerical equivalence:** Guaranteed — we compute identical max, min, mean over identical neighbor sets.

---

## Optimized R Code

```r
# =============================================================================
# Optimized spatial-panel neighbor feature computation
# =============================================================================
# Requirements: data.table, Matrix, ranger (or randomForest — for prediction)
# 
# Inputs expected:
#   cell_data              — data.frame/data.table with columns: id, year, 
#                            ntl, ec, pop_density, def, usd_est_n2, ...
#   id_order               — integer vector of cell IDs in the order matching
#                            rook_neighbors_unique
#   rook_neighbors_unique  — spdep nb object (list of integer index vectors)
#   rf_model               — pre-trained Random Forest model
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # ----------------------------------------------------------
  # 0. Convert to data.table if needed
  # ----------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))
  
  # ----------------------------------------------------------
  # 1. Build spatial edge list ONCE (cell-index space)
  #    This is the adjacency structure: ~1.37M directed edges
  # ----------------------------------------------------------
  cat("Building spatial edge list...\n")
  
  # rook_neighbors_unique[[i]] gives integer indices (into id_order) of 

  # neighbors of cell at position i in id_order.
  # We need directed edges: from = i, to = each neighbor index.
  
  edge_from <- integer(0)
  edge_to   <- integer(0)
  
  # Pre-allocate by counting total edges
  n_edges <- sum(lengths(rook_neighbors_unique))
  edge_from <- integer(n_edges)
  edge_to   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L for no-neighbor; filter those
    nb_i <- nb_i[nb_i > 0L]
    len  <- length(nb_i)
    if (len > 0L) {
      edge_from[pos:(pos + len - 1L)] <- i
      edge_to[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }
  # Trim if we over-allocated (due to 0-neighbor entries)
  edge_from <- edge_from[1:(pos - 1L)]
  edge_to   <- edge_to[1:(pos - 1L)]
  
  actual_edges <- length(edge_from)
  cat(sprintf("Spatial edges: %d\n", actual_edges))
  
  # ----------------------------------------------------------
  # 2. Build cell-index mapping: id -> cell_idx (position in id_order)
  # ----------------------------------------------------------
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell_idx to cell_data
  cell_data[, cell_idx := id_to_cellidx[as.character(id)]]
  
  # ----------------------------------------------------------
  # 3. Build sparse binary adjacency matrix for MEAN computation
  #    A[i,j] = 1 if j is a rook neighbor of i (i.e., edge from i to j)
  # ----------------------------------------------------------
  cat("Building sparse adjacency matrix...\n")
  
  A <- sparseMatrix(
    i = edge_from,
    j = edge_to,
    x = rep(1, actual_edges),
    dims = c(n_cells, n_cells)
  )
  
  # Degree vector (number of neighbors per cell, accounting for panel edge effects later)
  degree_vec <- diff(A@p)  # For dgCMatrix: column counts; but A is row-oriented here
  # Actually for row sums:
  degree_vec <- as.integer(rowSums(A))
  # Cells with 0 neighbors (boundary/island cells) — guard against division by zero
  degree_vec_safe <- pmax(degree_vec, 1L)
  
  # Normalized adjacency for mean: each row sums to 1
  # D^{-1} A  — row-normalized
  D_inv <- Diagonal(x = 1 / degree_vec_safe)
  A_mean <- D_inv %*% A
  
  # ----------------------------------------------------------
  # 4. Create cell × year matrices for each source variable
  #    and compute neighbor stats via vectorized operations
  # ----------------------------------------------------------
  
  # Ensure cell_data is sorted by (cell_idx, year) for matrix reshaping
  setkey(cell_data, cell_idx, year)
  
  # Year-to-column mapping
  year_to_col <- setNames(seq_along(years), as.character(years))
  cell_data[, year_col := year_to_col[as.character(year)]]
  
  # Edge table for max/min computation (data.table approach)
  # We expand edges × years: for each edge (from, to), for each year,
  # fetch the "to" node's attribute value, then group by (from, year).
  # 
  # But 1.37M edges × 28 years = ~38.4M rows — very manageable.
  
  cat("Building edge-year table...\n")
  edge_dt <- data.table(from_idx = edge_from, to_idx = edge_to)
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))
    
    # ---- Build cell × year matrix for this variable ----
    # V[cell_idx, year_col] = value
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(cell_data$cell_idx, cell_data$year_col)] <- cell_data[[var_name]]
    
    # ---- MEAN via sparse matrix multiplication ----
    # neighbor_mean_mat[i, t] = (1/deg_i) * sum_{j in N(i)} V[j, t]
    # = (A_mean %*% V)[i, t]
    # For cells with 0 neighbors, result will be 0; we'll fix to NA below.
    mean_mat <- as.matrix(A_mean %*% V)
    # Set to NA where degree is 0 or where all neighbor values were NA
    # (sparse matmul treats NA × 0 as 0 in the sum, which is fine since
    #  A has no entry for non-neighbors. But if a neighbor's value is NA,
    #  the sum includes NA. We need careful handling.)
    
    # Actually, standard matrix multiplication propagates NA correctly:
    # if any neighbor has NA, the sum is NA. The original code removes NAs 
    # before computing mean. So we need the NA-robust version.
    #
    # Strategy: use the edge-list approach for all three stats to handle 
    # NA removal identically to the original code.
    
    # ---- ALL STATS via edge-list + data.table grouped aggregation ----
    # This is the safest approach for numerical equivalence with NA handling.
    
    # For each edge (from_idx -> to_idx), for each year, get V[to_idx, year]
    # Then group by (from_idx, year) and compute max, min, mean (na.rm=TRUE)
    
    # Vectorized: expand edges across years
    # V_to[edge, year] = V[to_idx[edge], year]
    # We can do this as a matrix operation: V_to_mat = V[edge_to, ]
    # This is a 1.37M × 28 matrix — about 307 MB for doubles. Fine.
    
    V_to_mat <- V[edge_to, , drop = FALSE]  # (n_edges × n_years)
    
    # Now we need grouped aggregation: for each unique from_idx, 
    # compute columnwise max, min, mean of the corresponding rows of V_to_mat.
    
    # Use the sparse matrix structure to do this efficiently:
    # Group rows of V_to_mat by edge_from.
    
    # --- Approach: data.table melt + grouped agg ---
    # For 38.4M rows this is fast with data.table.
    
    # But melting a 1.37M × 28 matrix creates 38.4M rows — let's do it 
    # column by column (year by year) to save memory.
    
    max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    # Create a data.table with from_idx for grouping (reuse across years)
    agg_dt <- data.table(from_idx = edge_from)
    
    for (yr_col in seq_len(n_years)) {
      set(agg_dt, j = "val", value = V_to_mat[, yr_col])
      
      # Grouped aggregation — extremely fast in data.table
      stats <- agg_dt[!is.na(val), 
                       .(nb_max  = max(val),
                         nb_min  = min(val),
                         nb_mean = mean(val)),
                       by = from_idx]
      
      max_mat[stats$from_idx, yr_col]  <- stats$nb_max
      min_mat[stats$from_idx, yr_col]  <- stats$nb_min
      mean_mat[stats$from_idx, yr_col] <- stats$nb_mean
    }
    
    # ---- Map results back to cell_data rows ----
    row_indices <- cbind(cell_data$cell_idx, cell_data$year_col)
    
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    cell_data[, (max_col)  := max_mat[row_indices]]
    cell_data[, (min_col)  := min_mat[row_indices]]
    cell_data[, (mean_col) := mean_mat[row_indices]]
    
    # Clean up large temporaries
    rm(V, V_to_mat, max_mat, min_mat, mean_mat, stats)
  }
  
  # ----------------------------------------------------------
  # 5. Clean up helper columns
  # ----------------------------------------------------------
  cell_data[, c("cell_idx", "year_col") := NULL]
  
  cat("Neighbor feature computation complete.\n")
  return(cell_data)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# -- Load data (assumed already in environment or loaded from disk) --
# cell_data                <- readRDS("cell_data.rds")
# id_order                 <- readRDS("id_order.rds")
# rook_neighbors_unique    <- readRDS("rook_neighbors_unique.rds")
# rf_model                 <- readRDS("rf_model.rds")

# -- Compute neighbor features --
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique)

# -- Predict with pre-trained Random Forest (no retraining) --
cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
# (If using randomForest instead of ranger:)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `vals[idx]` gathers neighbor values for row `i` | `V[edge_to, yr_col]` gathers the same values via the same spatial neighbor indices, filtered to the same year | Same index set |
| `neighbor_vals[!is.na(neighbor_vals)]` removes NAs | `agg_dt[!is.na(val), ...]` removes NAs before aggregation | Identical NA handling |
| `max(neighbor_vals)` | `max(val)` grouped by `from_idx` | Identical function |
| `min(neighbor_vals)` | `min(val)` grouped by `from_idx` | Identical function |
| `mean(neighbor_vals)` | `mean(val)` grouped by `from_idx` | Identical function |
| No neighbors → `c(NA, NA, NA)` | Cells absent from `stats$from_idx` remain `NA` in the pre-initialized matrix | Identical |

---

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M string-paste + hash lookups | One pass over nb list → integer edge vectors |
| Per-variable computation | 6.46M `lapply` iterations with R function calls | 28 `data.table` grouped aggregations over ~1.37M rows each |
| Total inner iterations | 5 × 6.46M = 32.3M R-level calls | 5 × 28 = 140 vectorized `data.table` calls |
| Memory for lookup | ~6.46M-element list of integer vectors | Two integer vectors of length ~1.37M |
| Expected runtime | 86+ hours | **~2–10 minutes** |
| RAM peak | Potentially >16 GB (string keys) | ~4–6 GB (matrices + edge table) |

The 3–4 orders of magnitude speedup comes from: (1) eliminating per-row R function call overhead, (2) exploiting the space-time separability so the topology is size 1.37M not 6.46M, and (3) using `data.table`'s radix-sort grouped aggregation which runs in optimized C.