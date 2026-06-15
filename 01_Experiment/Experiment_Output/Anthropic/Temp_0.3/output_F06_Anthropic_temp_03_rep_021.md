 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(id, year)`) for each neighbor in that year.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) against a 6.46M-element named character vector.

Named vector lookup in R is **hash-based but slow at scale** — doing ~6.46M lookups, each touching ~4 neighbors on average, against a 6.46M-length named vector is extremely expensive. The `paste`-based key construction also allocates millions of small character vectors.

### Bottleneck B: `compute_neighbor_stats` — repeated `lapply` over 6.46M rows × 5 variables

Even after the lookup is built, `compute_neighbor_stats` runs an `lapply` over 6.46M rows for **each** of the 5 variables, performing subsetting and `max/min/mean` in pure R loops. That's ~32.3M R-level function calls total.

### Why raster focal/kernel operations don't directly apply

Focal operations assume a **complete regular grid** with uniform rectangular neighborhoods. Here:
- The panel is irregular in the sense that not all cells exist in all years (potential NA/missing cells).
- The neighbor structure is precomputed as an `nb` object (not necessarily a perfect rectangular grid after subsetting).
- We must **preserve the exact numerical results** of the rook-neighbor computation.

However, the **conceptual analogy is valid**: we are computing focal statistics (max, min, mean) over a spatial neighborhood. The optimization should use **vectorized sparse-matrix operations** that mimic focal computations without requiring a complete raster grid.

---

## 2. Optimization Strategy

### Strategy: Sparse Adjacency Matrix + Vectorized Column Operations

1. **Replace `build_neighbor_lookup`** with a sparse matrix `W` of dimension `(N_rows × N_rows)` where `N_rows = nrow(cell_data)` (~6.46M). Entry `W[i,j] = 1` if row `j` is a rook neighbor of row `i` in the same year. This matrix is extremely sparse (~4 nonzero entries per row on average).

2. **Replace `compute_neighbor_stats`** with vectorized operations:
   - **Mean**: `W %*% x / W %*% ones` (sparse matrix-vector multiply — blazing fast via `Matrix` package).
   - **Max and Min**: Use a grouped operation via `data.table` keyed joins, or iterate over the sparse matrix in C++ via `Rcpp`, or use the sparse structure directly.

3. **Build the sparse matrix efficiently** using `data.table` for the join (cell ID + year), avoiding `paste` keys entirely.

### Expected speedup

| Step | Current | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~hours (paste + named vector) | ~1-2 min (data.table join + sparse matrix) | ~60-100× |
| Neighbor stats (mean) | ~hours (lapply × 5 vars) | ~seconds (sparse mat-vec) | ~1000× |
| Neighbor stats (max/min) | ~hours (lapply × 5 vars) | ~2-5 min (Rcpp or grouped ops) | ~50-100× |
| **Total** | **86+ hours** | **~5-15 minutes** | **~350-1000×** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Dependencies: data.table, Matrix, Rcpp (optional but recommended for max/min)
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 1: Build sparse adjacency matrix (cell-year level) ----------------

build_sparse_neighbor_matrix <- function(cell_data, id_order, rook_neighbors) {
  # cell_data must have columns: id, year

  # id_order: vector of cell IDs in the order matching rook_neighbors (nb object)
  # rook_neighbors: nb object (list of integer index vectors into id_order)
  
  n_cells <- length(id_order)
  
  # --- 1a. Build spatial edge list (cell-level) ---
  # Each entry in rook_neighbors[[i]] gives indices into id_order
  from_cell <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  to_cell   <- unlist(rook_neighbors, use.names = FALSE)
  
  # Remove zero-neighbor entries (spdep uses integer(0) or 0 for no neighbors)
  valid <- to_cell > 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]
  
  # Convert to actual cell IDs
  from_id <- id_order[from_cell]
  to_id   <- id_order[to_cell]
  
  # Spatial edge table
  edges_spatial <- data.table(from_id = from_id, to_id = to_id)
  
  # --- 1b. Expand to cell-year level via join ---
  # Create a data.table with row indices
  dt <- data.table(
    row_idx = seq_len(nrow(cell_data)),
    id      = cell_data$id,
    year    = cell_data$year
  )
  
  # Join: for each spatial edge (from_id, to_id), find all years where BOTH exist
  # First, join edges with "from" rows
  setkey(dt, id, year)
  
  # from side
  edges_from <- edges_spatial[dt, on = .(from_id = id), 
                               .(to_id, year, from_row = row_idx),
                               nomatch = 0L, allow.cartesian = TRUE]
  
  # to side: join to get the row index of the neighbor in the same year
  edges_full <- edges_from[dt, on = .(to_id = id, year = year),
                            .(from_row, to_row = i.row_idx),
                            nomatch = 0L]
  
  # --- 1c. Build sparse matrix ---
  n <- nrow(cell_data)
  W <- sparseMatrix(
    i = edges_full$from_row,
    j = edges_full$to_row,
    x = 1,
    dims = c(n, n)
  )
  
  return(W)
}

# ---- Step 2: Compute neighbor stats using sparse matrix ---------------------

compute_neighbor_features_sparse <- function(cell_data, W, var_name) {
  # Extracts max, min, mean of var_name across rook neighbors
  # W: sparse adjacency matrix (row i has 1s in columns that are i's neighbors)
  
  x <- cell_data[[var_name]]
  n <- length(x)
  
  # --- Handle NAs: we need to exclude NA neighbors ---
  not_na <- as.numeric(!is.na(x))
  x_safe <- x
  x_safe[is.na(x_safe)] <- 0  # zero out NAs for summation
  
  # Number of non-NA neighbors per row
  neighbor_count <- as.numeric(W %*% not_na)
  
  # Sum of non-NA neighbor values
  neighbor_sum <- as.numeric(W %*% x_safe)
  
  # Mean
  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # --- Max and Min: requires iterating over sparse structure ---
  # We use the sparse matrix column pointers directly for efficiency
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  
  # Convert W to dgTMatrix (triplet form) for easy iteration by row
  # But iterating 6.46M rows in R is slow. Better: use dgCMatrix and Rcpp,
  # or use a data.table grouped approach.
  
  # data.table approach (fast grouped max/min):
  Wt <- as(W, "TsparseMatrix")  # triplet: i, j, x (0-indexed)
  
  edge_dt <- data.table(
    from_row = Wt@i + 1L,   # convert to 1-indexed
    to_row   = Wt@j + 1L
  )
  
  # Attach the variable values of the neighbor (to_row)
  edge_dt[, val := x[to_row]]
  
  # Remove edges where neighbor value is NA
  edge_dt <- edge_dt[!is.na(val)]
  
  # Grouped max and min
  stats <- edge_dt[, .(nb_max = max(val), nb_min = min(val)), by = from_row]
  
  neighbor_max[stats$from_row] <- stats$nb_max
  neighbor_min[stats$from_row] <- stats$nb_min
  
  # --- Assemble output names ---
  prefix <- paste0("nb_", var_name, "_")
  cell_data[[paste0(prefix, "max")]]  <- neighbor_max
  cell_data[[paste0(prefix, "min")]]  <- neighbor_min
  cell_data[[paste0(prefix, "mean")]] <- neighbor_mean
  
  return(cell_data)
}

# ---- Step 3: Main pipeline -------------------------------------------------

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cat("Building sparse neighbor matrix...\n")
  t0 <- Sys.time()
  W <- build_sparse_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)
  cat("  Done in", round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
  cat("  Matrix dimensions:", nrow(W), "x", ncol(W), "\n")
  cat("  Non-zero entries:", nnzero(W), "\n")
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor features for:", var_name, "...\n")
    t1 <- Sys.time()
    cell_data <- compute_neighbor_features_sparse(cell_data, W, var_name)
    cat("  Done in", round(difftime(Sys.time(), t1, units = "mins"), 2), "min\n")
  }
  
  return(cell_data)
}

# ---- Execute ----------------------------------------------------------------
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# Now proceed with prediction using the pre-trained Random Forest model
# (model is NOT retrained — only predict)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## 4. Why This Preserves the Original Numerical Estimand

| Property | Original | Optimized | Match? |
|---|---|---|---|
| Neighbor definition | Rook contiguity from `nb` object | Same `nb` object, same edges | ✅ |
| Year matching | Neighbors matched within same year | Same join logic (`on = year`) | ✅ |
| NA handling | Excluded before `max/min/mean` | Excluded identically | ✅ |
| `max` computation | `max(neighbor_vals)` | `data.table` grouped `max(val)` | ✅ Identical |
| `min` computation | `min(neighbor_vals)` | `data.table` grouped `min(val)` | ✅ Identical |
| `mean` computation | `mean(neighbor_vals)` | `sum / count` (NA-excluded) | ✅ Identical |
| No-neighbor rows | Returns `c(NA, NA, NA)` | Defaults to `NA` | ✅ |
| RF model | Not retrained | Not retrained | ✅ |

The sparse matrix `W` encodes **exactly** the same directed neighbor relationships as the original `neighbor_lookup` list. The `data.table` grouped operations and sparse matrix-vector products compute **algebraically identical** results to the original `max`, `min`, and `mean` calls — just vectorized instead of row-by-row in R.

---

## 5. Memory Estimate

| Object | Size |
|---|---|
| Sparse matrix W (6.46M × 6.46M, ~38.5M nonzeros) | ~600 MB |
| Triplet edge `data.table` (~38.5M rows × 3 cols) | ~900 MB |
| `cell_data` (~6.46M × 125 cols) | ~6-8 GB |
| **Total peak** | **~10-12 GB** (fits in 16 GB) |

If memory is tight, process variables one at a time and drop the triplet `edge_dt` between iterations (it's recreated each call but could be cached). Alternatively, cache `edge_dt` once and reuse:

```r
# Memory-optimized variant: build edge_dt once, reuse for all variables
Wt <- as(W, "TsparseMatrix")
edge_dt_base <- data.table(
  from_row = Wt@i + 1L,
  to_row   = Wt@j + 1L
)
rm(Wt); gc()

for (var_name in neighbor_source_vars) {
  x <- cell_data[[var_name]]
  not_na <- !is.na(x)
  x_safe <- replace(x, !not_na, 0)
  
  # Mean via sparse mat-vec
  nb_count <- as.numeric(W %*% as.numeric(not_na))
  nb_sum   <- as.numeric(W %*% x_safe)
  cell_data[[paste0("nb_", var_name, "_mean")]] <- 
    ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
  
  # Max/Min via data.table
  ed <- copy(edge_dt_base)
  ed[, val := x[to_row]]
  ed <- ed[!is.na(val)]
  stats <- ed[, .(nb_max = max(val), nb_min = min(val)), by = from_row]
  
  nb_max <- rep(NA_real_, length(x))
  nb_min <- rep(NA_real_, length(x))
  nb_max[stats$from_row] <- stats$nb_max
  nb_min[stats$from_row] <- stats$nb_min
  
  cell_data[[paste0("nb_", var_name, "_max")]] <- nb_max
  cell_data[[paste0("nb_", var_name, "_min")]] <- nb_min
  
  rm(ed, stats); gc()
}
```

This should complete the entire neighbor feature computation in **approximately 5–15 minutes** on a standard 16 GB laptop, down from the estimated 86+ hours.