 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs string keys by pasting cell ID and year.
4. Looks up row indices via a named character vector (`idx_lookup`).

String pasting and named-vector lookups for 6.46M rows, each with ~4 rook neighbors on average (~1.37M directed relationships / 344K cells ≈ 4), means roughly **25+ million string operations and hash lookups**. The `lapply` over 6.46M elements in R is inherently slow — this alone could take many hours.

### Bottleneck 2: `compute_neighbor_stats` — repeated `lapply` over 6.46M rows

For **each** of the 5 variables, another `lapply` iterates over 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing `max`, `min`, `mean`. That's 5 × 6.46M = ~32.3M R-level function calls with per-element vector subsetting.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a regular grid with a fixed kernel. The panel data here is a cell × year long table, and the neighbor structure is an irregular `spdep::nb` object (not all cells have the same number of neighbors — boundary cells have fewer). Focal operations would require reshaping into a raster stack per year and handling boundary irregularities. While conceptually analogous, the **vectorized sparse-matrix approach below is more faithful to the `nb` structure and preserves exact numerical results**.

### Summary

| Component | Calls | Estimated Time |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string ops | ~30–40 hrs |
| `compute_neighbor_stats` × 5 vars | 32.3M R function calls | ~45–50 hrs |
| **Total** | | **~80–90 hrs** |

---

## Optimization Strategy

### Key Insight: Separate the spatial dimension from the temporal dimension

Every cell's rook neighbors are **the same in every year**. The `nb` object defines ~344K spatial relationships. The temporal join (matching neighbors within the same year) is currently done row-by-row via string keys. Instead:

1. **Convert the `nb` object to a sparse adjacency matrix** (344K × 344K) once — this is a standard `spdep` operation.
2. **Reshape each variable into a matrix**: 344K cells × 28 years.
3. **Use sparse matrix multiplication and row-wise operations** to compute neighbor max, min, and mean in fully vectorized C-level code.

This eliminates all `lapply` loops, all string operations, and all per-row R function calls.

### Complexity Reduction

| Step | Before | After |
|---|---|---|
| Neighbor lookup | 6.46M string-paste + hash lookups | One `nb2listw` → sparse matrix conversion |
| Stats computation (per var) | 6.46M `lapply` iterations | 3 sparse matrix operations on 344K × 28 matrices |
| Total R-level iterations | ~38M | ~0 (all vectorized) |

**Expected runtime: 2–10 minutes** on a 16 GB laptop.

### Numerical Equivalence

- The sparse matrix `W` has a 1 in position (i, j) iff cell j is a rook neighbor of cell i — identical to the `nb` object.
- `W %*% X` computes the sum of neighbor values for each cell. Dividing by the number of neighbors (row sums of `W`) gives the **exact same mean**.
- For max and min, we use a loop over the (small) neighbor-count dimension or a grouped operation, since sparse matrix algebra doesn't directly support max/min. However, since the maximum number of rook neighbors is **4**, we can restructure into at most 4 "neighbor-slot" matrices and use `pmax`/`pmin` — fully vectorized.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves exact numerical results of the original implementation.
# =============================================================================

library(Matrix)   # for sparse matrices
library(spdep)    # for nb2listw / nb2mat if needed

# ---- Step 0: Prepare ID-to-index mapping ----
# id_order: vector of cell IDs in the order matching rook_neighbors_unique (the nb object)
# cell_data: data.frame/data.table with columns id, year, and the 5 neighbor source vars

# Ensure cell_data is a data.table for fast operations
if (!requireNamespace("data.table", quietly = TRUE)) install.packages("data.table")
library(data.table)
setDT(cell_data)

# ---- Step 1: Build sparse binary adjacency matrix from nb object ----
build_sparse_adjacency <- function(nb_obj) {
  n <- length(nb_obj)
  # Build COO (coordinate) triplets
  from <- rep(seq_len(n), times = vapply(nb_obj, length, integer(1)))
  to   <- unlist(nb_obj)
  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

W <- build_sparse_adjacency(rook_neighbors_unique)
# W[i,j] = 1 iff cell j is a rook neighbor of cell i
# This is the exact same adjacency as the original nb object.

cat("Adjacency matrix:", nrow(W), "x", ncol(W),
    "with", nnzero(W), "non-zero entries\n")

# ---- Step 2: Determine year range and cell ordering ----
years      <- sort(unique(cell_data$year))
n_years    <- length(years)
n_cells    <- length(id_order)
year_to_col <- setNames(seq_along(years), as.character(years))

# Map each cell ID to its spatial index (matching the nb object order)
id_to_sidx <- setNames(seq_along(id_order), as.character(id_order))

# ---- Step 3: Precompute neighbor-slot structure for max/min ----
# Each cell has at most max_k rook neighbors. We create max_k index vectors.
# For cells with fewer neighbors, we pad with NA.

nb_lengths <- vapply(rook_neighbors_unique, function(x) {
  sum(x > 0L)
}, integer(1))
max_k <- max(nb_lengths)  # Should be 4 for rook contiguity (or less at boundaries)
cat("Max rook neighbors per cell:", max_k, "\n")

# Build neighbor-slot matrix: n_cells x max_k
# neighbor_slots[i, k] = spatial index of the k-th neighbor of cell i, or NA
neighbor_slots <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  nb_i <- nb_i[nb_i > 0L]
  if (length(nb_i) > 0) {
    neighbor_slots[i, seq_along(nb_i)] <- nb_i
  }
}

# Number of neighbors per cell (for mean computation)
n_neighbors <- rowSums(!is.na(neighbor_slots))

# ---- Step 4: Reshape cell_data into cell x year matrices ----
# We need a fast way to go from long format to matrix format.

# Add spatial index and year index columns
cell_data[, sidx := id_to_sidx[as.character(id)]]
cell_data[, yidx := year_to_col[as.character(year)]]

reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- dt[[var_name]]
  sidx <- dt$sidx
  yidx <- dt$yidx
  # Vectorized assignment
  mat[cbind(sidx, yidx)] <- vals
  mat
}

# ---- Step 5: Compute neighbor stats for each variable ----
# For each variable:
#   - Reshape to n_cells x n_years matrix
#   - For MEAN: use sparse matrix multiplication  W %*% X / n_neighbors
#   - For MAX/MIN: use neighbor_slots to gather neighbor values, then pmax/pmin

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns in cell_data
for (var_name in neighbor_source_vars) {
  cell_data[, paste0("n_max_", var_name) := NA_real_]
  cell_data[, paste0("n_min_", var_name) := NA_real_]
  cell_data[, paste0("n_mean_", var_name) := NA_real_]
}

# Linear index helper for fast matrix access
# Given a neighbor_slots matrix (n_cells x max_k) and n_years columns,
# we need to look up X[neighbor_slots[i,k], t] for all i, k, t.

for (var_name in neighbor_source_vars) {
  cat("Processing variable:", var_name, "...\n")
  t0 <- proc.time()

  # Step 5a: Reshape to matrix
  X <- reshape_to_matrix(cell_data, var_name, n_cells, n_years)

  # Step 5b: Compute MEAN via sparse matrix multiplication
  # W %*% X gives sum of neighbor values for each cell and year
  # Divide by number of neighbors to get mean
  neighbor_sum  <- as.matrix(W %*% X)  # n_cells x n_years dense matrix
  neighbor_mean <- neighbor_sum / n_neighbors  # recycling: n_neighbors is length n_cells
  # Cells with 0 neighbors: n_neighbors=0 → Inf or NaN; set to NA
  neighbor_mean[n_neighbors == 0, ] <- NA_real_

  # Step 5c: Compute MAX and MIN via neighbor slots
  # Gather neighbor values into max_k layers, then reduce with pmax/pmin
  # Each "layer" k: a matrix of n_cells x n_years where row i = X[neighbor_slots[i,k], ]

  neighbor_max <- matrix(-Inf, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(Inf,  nrow = n_cells, ncol = n_years)

  for (k in seq_len(max_k)) {
    slot_k <- neighbor_slots[, k]  # length n_cells; NA if cell has < k neighbors
    has_k  <- !is.na(slot_k)

    # Gather: for cells that have a k-th neighbor, pull their values
    # X_k[i, ] = X[slot_k[i], ] for cells where has_k[i] is TRUE
    X_k <- X[slot_k[has_k], , drop = FALSE]  # subset rows of X

    # Update max and min only for cells that have this k-th neighbor
    neighbor_max[has_k, ] <- pmax(neighbor_max[has_k, , drop = FALSE], X_k, na.rm = TRUE)
    neighbor_min[has_k, ] <- pmin(neighbor_min[has_k, , drop = FALSE], X_k, na.rm = TRUE)
  }

  # Cells with 0 neighbors or all-NA neighbors: set to NA
  neighbor_max[n_neighbors == 0, ] <- NA_real_
  neighbor_min[n_neighbors == 0, ] <- NA_real_
  # If all neighbor values were NA, -Inf/Inf remain; fix:
  neighbor_max[is.infinite(neighbor_max)] <- NA_real_
  neighbor_min[is.infinite(neighbor_min)] <- NA_real_

  # Handle the case where MEAN should be NA when all neighbor values are NA
  # (sparse multiplication treats NA as 0 in standard Matrix ops)
  # We need to correct for NAs in the original data.

  # Count non-NA neighbor values per cell-year
  X_notna <- (!is.na(X)) * 1.0  # 1 if not NA, 0 if NA
  neighbor_notna_count <- as.matrix(W %*% X_notna)  # count of non-NA neighbors

  # Replace X NAs with 0 for correct summation, then recompute sum
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0
  neighbor_sum_corrected <- as.matrix(W %*% X_zero)

  # Corrected mean: sum of non-NA values / count of non-NA values
  neighbor_mean_corrected <- neighbor_sum_corrected / neighbor_notna_count
  neighbor_mean_corrected[neighbor_notna_count == 0] <- NA_real_

  # Step 5d: Write results back to cell_data using linear indexing
  # cell_data has sidx and yidx columns for direct mapping
  lin_idx <- cbind(cell_data$sidx, cell_data$yidx)

  cell_data[, paste0("n_max_",  var_name) := neighbor_max[lin_idx]]
  cell_data[, paste0("n_min_",  var_name) := neighbor_min[lin_idx]]
  cell_data[, paste0("n_mean_", var_name) := neighbor_mean_corrected[lin_idx]]

  elapsed <- (proc.time() - t0)["elapsed"]
  cat("  Done in", round(elapsed, 1), "seconds\n")

  # Free memory
  rm(X, X_zero, X_notna, neighbor_sum, neighbor_sum_corrected,
     neighbor_mean, neighbor_mean_corrected, neighbor_notna_count,
     neighbor_max, neighbor_min)
  gc()
}

# ---- Step 6: Clean up helper columns ----
cell_data[, c("sidx", "yidx") := NULL]

# ---- Step 7: Predict with the pre-trained Random Forest ----
# The trained model is NOT retrained. We simply call predict().
# Ensure cell_data has all ~110 predictor columns the model expects.

# predictions <- predict(trained_rf_model, newdata = cell_data)

cat("\nAll neighbor features computed. Ready for prediction.\n")
```

---

## Why This Works and Preserves Exact Results

| Aspect | Original | Optimized | Equivalence |
|---|---|---|---|
| **Neighbor definition** | `nb` object → per-row string lookup | Same `nb` object → sparse matrix `W` | Identical adjacency |
| **Mean** | `mean(vals[idx])` | `(W %*% X_zero) / count_nonNA` | Algebraically identical; NA handling replicated |
| **Max** | `max(vals[idx])` | `pmax` across ≤4 neighbor slots | Identical; same values compared |
| **Min** | `min(vals[idx])` | `pmin` across ≤4 neighbor slots | Identical; same values compared |
| **NA handling** | `neighbor_vals[!is.na()]` then stats, or `c(NA,NA,NA)` if empty | Explicit NA counting and correction | Identical behavior |
| **RF model** | Pre-trained, not retrained | Unchanged — only `predict()` called | Identical predictions |

## Expected Performance

| Step | Time Estimate |
|---|---|
| Sparse matrix construction | ~5 seconds |
| Neighbor-slot matrix | ~2 seconds |
| Per-variable reshape + compute (×5) | ~30–90 seconds each |
| **Total** | **~3–8 minutes** |

This represents a **~600–1700× speedup** over the original 86+ hour estimate, achieved entirely through vectorization and sparse linear algebra while preserving the exact numerical estimand.