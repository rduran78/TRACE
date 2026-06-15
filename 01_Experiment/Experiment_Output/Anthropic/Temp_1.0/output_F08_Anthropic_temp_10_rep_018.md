 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental architectural mistake: **it conflates the static spatial topology with the dynamic yearly variable values.**

### Specific bottlenecks:

1. **`build_neighbor_lookup` operates on the exploded cell-year panel (6.46M rows).** It creates a lookup keyed by `paste(id, year)` for every row, then for each of the 6.46M rows, it finds neighbors by string-matching into that keyed index. This produces a list of 6.46M elements, each containing integer row indices into the full panel. The string pasting, hashing, and per-row `lapply` over 6.46M rows is enormously expensive in R.

2. **The neighbor topology is year-invariant.** Each cell has the same rook neighbors every year. The `rook_neighbors_unique` nb object already encodes this — it is a list of 344,208 integer vectors mapping each cell to its neighbor *cells*. But `build_neighbor_lookup` redundantly recomputes this mapping for every cell×year combination, inflating the work by a factor of 28.

3. **`compute_neighbor_stats` iterates over the 6.46M-element lookup list** with per-element R-level `lapply` calls and subsetting. This is repeated 5 times (once per source variable), producing 5 × 6.46M = 32.3M R-level function calls.

4. **Memory:** The `neighbor_lookup` list itself stores ~6.46M integer vectors, each a copy of what could be derived from the 344K-element nb object plus a year offset. This wastes substantial RAM.

### Root cause summary:

> The static cell-to-cell neighbor graph (344K cells, ~1.37M edges) is being re-expressed as a dynamic row-to-row neighbor graph (6.46M rows, ~38.4M edges) via expensive string operations, when it should be computed once at the cell level and then applied via vectorized matrix/array operations across years.

---

## Optimization Strategy

**Separate topology (static, cell-level) from data (dynamic, year-level).** Then use vectorized matrix arithmetic instead of row-level R loops.

### Key ideas:

1. **Build a sparse adjacency matrix `W` once** from `rook_neighbors_unique` (344,208 × 344,208). This is the static topology. Use the `Matrix` package.

2. **Reshape each variable into a dense matrix `V`** of dimension 344,208 cells × 28 years. Each column is one year's values.

3. **Compute neighbor stats via sparse matrix multiplication and sparse-matrix operations:**
   - **Neighbor mean:** `W %*% V / degree` (where degree = number of non-NA neighbors per cell, adjusted for NAs).
   - **Neighbor max and min:** These are not expressible as simple matrix products, but can be computed efficiently by iterating over cells (not cell-years) using the nb list directly on the matrix columns — a 344K-element loop instead of a 6.46M-element loop, or better yet, via `data.table` grouped operations.

4. **Flatten results back** into the original panel ordering and attach columns.

### Expected speedup:

| Aspect | Current | Proposed |
|---|---|---|
| Lookup construction | 6.46M string-paste + hash | One-time 344K sparse matrix build |
| Stat computation per variable | 6.46M R-level `lapply` calls | Sparse matrix multiply (mean) + 344K-cell vectorized loop (max/min) |
| Total R-level iterations | ~32.3M | ~1.72M (344K × 5 vars) + vectorized `W %*% V` |
| Estimated time | 86+ hours | ~5–15 minutes |

---

## Working R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic year-varying data.
# Preserves the original numerical estimand exactly.
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# STEP 1: Build the static sparse adjacency matrix (once, from nb object)
# --------------------------------------------------------------------------
build_sparse_adjacency <- function(nb_obj) {
  # nb_obj: a list of length N_cells, each element is an integer vector of

  #         neighbor indices (spdep::nb format, 0 means no neighbors)
  n <- length(nb_obj)
  
  # Build COO (coordinate) triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  
  # spdep nb objects use 0L for "no neighbors" — remove those
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Sparse binary adjacency matrix (row i has 1s in columns that are i's neighbors)
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  W
}

# --------------------------------------------------------------------------
# STEP 2: Reshape a variable from long panel to cell × year matrix
# --------------------------------------------------------------------------
reshape_to_matrix <- function(dt, var_name, cell_idx, year_idx) {
  # dt:       data.table with columns id, year, and var_name
  # cell_idx: named integer vector mapping cell id -> row position (1..N_cells)
  # year_idx: named integer vector mapping year -> column position (1..N_years)
  
  n_cells <- length(cell_idx)
  n_years <- length(year_idx)
  
  V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  ri <- cell_idx[as.character(dt$id)]
  ci <- year_idx[as.character(dt$year)]
  V[cbind(ri, ci)] <- dt[[var_name]]
  
  V
}

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor max, min, mean using static topology + matrix data
# --------------------------------------------------------------------------
compute_neighbor_stats_optimized <- function(nb_obj, V) {
  # nb_obj: spdep::nb list (length = N_cells)
  # V:      matrix N_cells x N_years (one variable's values)
  # Returns: list with three matrices (max, min, mean), each N_cells x N_years
  
  n_cells <- nrow(V)
  n_years <- ncol(V)
  
  nmax <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nmin <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nmen <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    # spdep convention: integer(0) or 0 means no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) next
    
    # Extract submatrix: rows = neighbors, cols = years
    # This is a length(nbrs) x n_years matrix
    sub <- V[nbrs, , drop = FALSE]
    
    # Vectorized across years (column-wise operations)
    # Handle NAs: need to replicate original behavior (na.rm-like)
    # For each year-column, compute max/min/mean of non-NA neighbor values
    # Using colMeans, colMaxs-equivalent, etc.
    
    if (length(nbrs) == 1L) {
      # sub is a 1-row matrix; max = min = mean = value (or NA)
      nmax[i, ] <- sub[1L, ]
      nmin[i, ] <- sub[1L, ]
      nmen[i, ] <- sub[1L, ]
    } else {
      # Use apply only when there are multiple neighbors
      # For speed with many neighbors, use matrixStats if available,
      # but base R apply is fine since the inner dimension (nbrs) is small (≤4 for rook)
      nmax[i, ] <- apply(sub, 2L, max,  na.rm = TRUE)
      nmin[i, ] <- apply(sub, 2L, min,  na.rm = TRUE)
      nmen[i, ] <- apply(sub, 2L, mean, na.rm = TRUE)
    }
  }
  
  # Fix Inf/-Inf from max/min on all-NA columns (replicates original c(NA,NA,NA) behavior)
  nmax[is.infinite(nmax)] <- NA_real_
  nmin[is.infinite(nmin)] <- NA_real_
  
  list(nmax = nmax, nmin = nmin, nmean = nmen)
}

# --------------------------------------------------------------------------
# STEP 3-ALT: Much faster version using matrixStats (recommended)
# --------------------------------------------------------------------------
compute_neighbor_stats_fast <- function(nb_obj, V) {
  # Uses matrixStats::colMaxs/colMins/colMeans2 for speed.
  # If matrixStats is unavailable, falls back to the loop version above.
  
  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    message("matrixStats not available; using loop fallback.")
    return(compute_neighbor_stats_optimized(nb_obj, V))
  }
  
  n_cells <- nrow(V)
  n_years <- ncol(V)
  
  nmax <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nmin <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nmen <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) next
    
    sub <- V[nbrs, , drop = FALSE]
    
    if (length(nbrs) == 1L) {
      nmax[i, ] <- sub[1L, ]
      nmin[i, ] <- sub[1L, ]
      nmen[i, ] <- sub[1L, ]
    } else {
      nmax[i, ] <- matrixStats::colMaxs(sub,  na.rm = TRUE)
      nmin[i, ] <- matrixStats::colMins(sub,  na.rm = TRUE)
      nmen[i, ] <- matrixStats::colMeans2(sub, na.rm = TRUE)
    }
  }
  
  nmax[is.infinite(nmax)] <- NA_real_
  nmin[is.infinite(nmin)] <- NA_real_
  
  list(nmax = nmax, nmin = nmin, nmean = nmen)
}

# --------------------------------------------------------------------------
# STEP 4: Flatten matrix results back into the panel and attach columns
# --------------------------------------------------------------------------
flatten_and_attach <- function(dt, var_name, stats, cell_idx, year_idx) {
  # stats: list with nmax, nmin, nmean matrices (N_cells x N_years)
  # Attaches three new columns to dt: neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
  
  ri <- cell_idx[as.character(dt$id)]
  ci <- year_idx[as.character(dt$year)]
  idx <- cbind(ri, ci)
  
  dt[[paste0("neighbor_max_",  var_name)]] <- stats$nmax[idx]
  dt[[paste0("neighbor_min_",  var_name)]] <- stats$nmin[idx]
  dt[[paste0("neighbor_mean_", var_name)]] <- stats$nmean[idx]
  
  dt
}

# ==========================================================================
# MAIN EXECUTION
# ==========================================================================

# Convert to data.table for efficient column operations (if not already)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# --- Build static index mappings (once) ---
id_order   <- unique(cell_data$id)       # or use the same id_order as before
year_order <- sort(unique(cell_data$year))

cell_idx <- setNames(seq_along(id_order),   as.character(id_order))
year_idx <- setNames(seq_along(year_order),  as.character(year_order))

# --- Neighbor source variables (same as original) ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- Process each variable ---
cat("Processing neighbor features for", length(neighbor_source_vars), "variables...\n")
cat("Grid cells:", length(id_order), " | Years:", length(year_order),
    " | Rows:", nrow(cell_data), "\n")

for (var_name in neighbor_source_vars) {
  cat("  Variable:", var_name, "...")
  t0 <- proc.time()
  
  # Reshape long -> cell x year matrix
  V <- reshape_to_matrix(cell_data, var_name, cell_idx, year_idx)
  
  # Compute neighbor stats (static topology × dynamic values)
  stats <- compute_neighbor_stats_fast(rook_neighbors_unique, V)
  
  # Flatten back and attach to panel
  cell_data <- flatten_and_attach(cell_data, var_name, stats, cell_idx, year_idx)
  
  # Free memory
  rm(V, stats)
  gc()
  
  elapsed <- (proc.time() - t0)[3]
  cat(" done in", round(elapsed, 1), "seconds\n")
}

cat("All neighbor features computed.\n")

# --- Proceed with pre-trained Random Forest prediction (unchanged) ---
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

The computation is mathematically identical:

| Original | Optimized |
|---|---|
| For row `i` in the panel (cell `c`, year `y`), find all rows `j` where `j.id ∈ neighbors(c)` and `j.year == y`. | For cell `c`, extract `V[neighbors(c), y]` from the pre-reshaped matrix. |
| Compute `max`, `min`, `mean` of `var[j]` excluding NAs. | Compute `colMaxs`, `colMins`, `colMeans2` of the neighbor submatrix excluding NAs. |
| Return `NA` if no valid neighbor values. | Same: `Inf`/`-Inf` from empty `na.rm=TRUE` are mapped back to `NA`. |

The same neighbor set is used (same nb object), the same values are accessed (same variable from the same rows), and the same aggregation functions are applied. The Random Forest model is never retrained — only the feature-engineering step is accelerated.

---

## Performance Estimate

| Component | Time estimate |
|---|---|
| `reshape_to_matrix` per variable | ~2–5 seconds |
| `compute_neighbor_stats_fast` per variable (344K cells × ≤4 neighbors × 28 years) | ~2–8 minutes |
| `flatten_and_attach` per variable | ~1–3 seconds |
| **Total for 5 variables** | **~10–40 minutes** |

This is a **~130–500× speedup** over the original 86+ hours, achieved entirely by respecting the static-vs-dynamic distinction that the problem statement identifies.