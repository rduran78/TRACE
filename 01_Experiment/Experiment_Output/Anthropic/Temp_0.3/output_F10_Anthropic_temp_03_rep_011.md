 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each via an anonymous function call with string-pasting and named-vector lookups. The `paste(id, year)` keying strategy means the lookup table has 6.46M character keys, and for each of the 6.46M rows, it does character matching against this table. This is O(N²)-flavored in practice due to repeated string operations.

2. **`compute_neighbor_stats` iterates row-by-row in pure R** over 6.46M rows, extracting subsets of a numeric vector via list-of-index-vectors. Each call to `max/min/mean` on small vectors has high per-call overhead. With 5 variables × 6.46M rows = 32.3M iterations of the inner function, this dominates runtime.

3. **The neighbor topology is year-invariant but the lookup is rebuilt entangling spatial and temporal dimensions.** Rook neighbors are a fixed spatial property of the grid. The current code re-discovers the same spatial neighbors for every year by embedding year into the key. This inflates the problem from ~344K spatial lookups to ~6.46M spatiotemporal lookups.

**Root cause summary:** The code treats a separable problem (space × time) as a monolithic spatiotemporal problem, and uses pure-R loops with per-element string operations over millions of rows.

---

## Optimization Strategy

### Key Insight: Separate Space from Time

The rook neighbor graph is **purely spatial** — it does not change across years. For any variable `v`, the neighbor statistics for cell `i` in year `t` depend only on the values of `v` for cell `i`'s spatial neighbors in the **same year** `t`. This means:

1. **Build the spatial adjacency structure once** as a sparse matrix (344K × 344K), not a 6.46M-entry list.
2. **Reshape each variable into a matrix** of shape (344K cells × 28 years).
3. **Use sparse matrix–dense matrix multiplication** (`A %*% V`) to compute neighbor sums and neighbor counts in one vectorized operation, then derive mean. For max and min, use grouped operations via the sparse structure.

### Specific Optimizations

| Bottleneck | Solution | Speedup Factor |
|---|---|---|
| String-key lookup over 6.46M rows | Integer-indexed sparse matrix, built once | ~100× |
| Row-by-row `lapply` for stats | Vectorized sparse matrix ops for mean; column-parallel grouped ops for max/min | ~50–200× |
| Redundant per-year neighbor discovery | Year-invariant adjacency matrix reused across all 28 years | 28× |
| 5 variables processed sequentially with same structure | Same sparse matrix reused for all variables | Marginal but clean |

### Memory Budget

- Sparse adjacency matrix: ~1.37M non-zeros × 12 bytes ≈ 16 MB
- One variable reshaped to (344K × 28) dense matrix: ~77 MB
- Three output matrices (max, min, mean) per variable: ~231 MB
- Peak for one variable pass: ~325 MB
- Total with all 5 variables added to `cell_data`: the original `cell_data` with ~110 columns is ~5.7 GB at 8 bytes/element. Adding 15 columns (3 stats × 5 vars) adds ~775 MB. Fits in 16 GB.

### Numerical Equivalence

The sparse-matrix approach computes **identical** neighbor sets (same rook adjacency, same year matching). The `mean` via `sum/count` is IEEE-754 equivalent when summation order is consistent. For `max` and `min`, we use exact grouped operations. The Random Forest model is loaded and applied unchanged.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Sparse graph neighborhood aggregation for panel grid data
# =============================================================================

library(Matrix)    # sparse matrices
library(data.table) # fast reshaping and joining

# ---- Step 0: Convert cell_data to data.table if not already ----
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build spatial adjacency as a sparse matrix (once) ----
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: spdep nb object (list of integer index vectors)

build_sparse_adjacency <- function(id_order, neighbors) {
  n <- length(id_order)
  # Build COO (coordinate) representation
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) > 0 && !(length(nb_i) == 1 && nb_i[0] == 0L)) {
      # spdep nb objects use 0L to indicate no neighbors
      nb_i <- nb_i[nb_i != 0L]
      if (length(nb_i) > 0) {
        from_list[[i]] <- rep.int(i, length(nb_i))
        to_list[[i]]   <- nb_i
      }
    }
  }
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list, use.names = FALSE)

  # Sparse matrix: A[i,j] = 1 means j is a rook neighbor of i
  # So row i contains the neighbors of cell i
  A <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(n, n),
    repr = "C"   # CSC -> will convert to CSR-like via dgRMatrix or use dgCMatrix
  )
  return(A)
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency(id_order, rook_neighbors_unique)
n_cells <- length(id_order)
cat(sprintf("  Adjacency: %d cells, %d directed edges\n", n_cells, nnzero(A)))

# ---- Step 2: Create stable cell-index and year-index mappings ----
# Map cell IDs to row indices in the adjacency matrix
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Determine sorted unique years
years_unique <- sort(unique(cell_data$year))
n_years <- length(years_unique)
year_to_col <- setNames(seq_along(years_unique), as.character(years_unique))

cat(sprintf("  Panel: %d cells x %d years = %d expected rows\n",
            n_cells, n_years, n_cells * n_years))

# ---- Step 3: Assign spatial and temporal indices to cell_data ----
cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]
cell_data[, year_col    := year_to_col[as.character(year)]]

# Verify completeness
stopifnot(all(!is.na(cell_data$spatial_idx)))
stopifnot(all(!is.na(cell_data$year_col)))

# ---- Step 4: Precompute neighbor count matrix for mean calculation ----
# For mean: we need sum of neighbor values / count of non-NA neighbors
# For count of non-NA: A %*% (non-NA indicator matrix)
# For sum: A %*% (value matrix with NA replaced by 0, masked by non-NA)

# ---- Step 5: Function to reshape variable to (n_cells x n_years) matrix ----
reshape_to_matrix <- function(dt, var_name, n_cells, n_years) {
  # Create matrix filled with NA
  M <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  # Fill using spatial_idx and year_col
  M[cbind(dt$spatial_idx, dt$year_col)] <- dt[[var_name]]
  return(M)
}

# ---- Step 6: Compute neighbor stats using sparse matrix operations ----
compute_neighbor_stats_sparse <- function(A, V) {
  # A: n_cells x n_cells sparse adjacency (dgCMatrix)
  # V: n_cells x n_years dense matrix of variable values
  # Returns list with max_mat, min_mat, mean_mat (each n_cells x n_years)

  n_cells <- nrow(V)
  n_years <- ncol(V)

  # --- MEAN via sparse matrix multiplication ---
  # Replace NA with 0 for summation, track non-NA
  not_na <- !is.na(V)
  V_zero <- V
  V_zero[!not_na] <- 0

  # Indicator matrix: 1 where not NA, 0 where NA
  I_mat <- matrix(0, nrow = n_cells, ncol = n_years)
  I_mat[not_na] <- 1

  # Neighbor sum: A %*% V_zero (each row i gets sum of neighbor values)
  neighbor_sum   <- as.matrix(A %*% V_zero)
  # Neighbor count of non-NA: A %*% I_mat
  neighbor_count <- as.matrix(A %*% I_mat)

  # Mean
  mean_mat <- neighbor_sum / neighbor_count
  mean_mat[neighbor_count == 0] <- NA_real_

  # --- MAX and MIN via explicit grouped operations ---
  # Extract the sparse structure once
  # A is dgCMatrix: columns are stored. We need row-wise neighbors.
  # Convert to dgRMatrix or iterate over dgCMatrix columns smartly.
  # Most efficient: use the @i, @p, @x slots of dgCMatrix (CSC format)
  # For row-wise access, transpose to get A^T in CSC = A in CSR
  At <- t(A)  # Now At is dgCMatrix; column j of At = row j of A = neighbors of j

  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # At@p: column pointers (0-indexed), length n_cells+1
  # At@i: row indices (0-indexed) of non-zero entries
  p <- At@p
  row_idx <- At@i  # 0-indexed

  # Process year by year to keep memory bounded
  for (yr in seq_len(n_years)) {
    v <- V[, yr]  # values for this year, length n_cells

    # For each cell i, neighbors are: row_idx[(p[i]+1):p[i+1]] (converting to 1-indexed)
    # Vectorized approach: build neighbor value vector, then use grouping

    # Total number of non-zero entries
    nnz <- length(row_idx)
    if (nnz == 0) next

    # Neighbor values for all edges
    neighbor_vals <- v[row_idx + 1L]  # +1 for 0-indexed to 1-indexed

    # Group IDs: which cell does each edge belong to?
    # Cell i owns entries from index (p[i]+1) to p[i+1] (1-indexed)
    # Build group vector
    group_lengths <- diff(p)  # length n_cells, number of neighbors per cell
    group_id <- rep.int(seq_len(n_cells), times = group_lengths)

    # Remove NA neighbor values
    valid <- !is.na(neighbor_vals)
    if (sum(valid) == 0) next

    nv_valid <- neighbor_vals[valid]
    gid_valid <- group_id[valid]

    # Use data.table for fast grouped max/min
    dt_temp <- data.table(g = gid_valid, val = nv_valid)
    stats <- dt_temp[, .(mx = max(val), mn = min(val)), by = g]

    max_mat[stats$g, yr] <- stats$mx
    min_mat[stats$g, yr] <- stats$mn
  }

  return(list(max_mat = max_mat, min_mat = min_mat, mean_mat = mean_mat))
}

# ---- Step 7: Main loop over variables ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-sort cell_data by (spatial_idx, year_col) for consistent matrix fill
setkey(cell_data, spatial_idx, year_col)

cat("Computing neighbor statistics...\n")
t_start <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing variable: %s\n", var_name))

  # Reshape to matrix
  V <- reshape_to_matrix(cell_data, var_name, n_cells, n_years)

  # Compute stats
  stats <- compute_neighbor_stats_sparse(A, V)

  # Map back to cell_data rows
  # cell_data already has spatial_idx and year_col
  idx_mat <- cbind(cell_data$spatial_idx, cell_data$year_col)

  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_data[, (max_col)  := stats$max_mat[idx_mat]]
  cell_data[, (min_col)  := stats$min_mat[idx_mat]]
  cell_data[, (mean_col) := stats$mean_mat[idx_mat]]

  # Free memory
  rm(V, stats)
  gc()
}

t_elapsed <- proc.time() - t_start
cat(sprintf("Neighbor stats completed in %.1f seconds\n", t_elapsed[3]))

# ---- Step 8: Clean up helper columns ----
cell_data[, c("spatial_idx", "year_col") := NULL]

# ---- Step 9: Apply pre-trained Random Forest (unchanged) ----
# The model object (e.g., `rf_model`) is already in memory or loaded from disk.
# Predict using the enriched cell_data with all ~125 predictor columns.
# 
# Example:
# library(ranger)  # or randomForest
# predictions <- predict(rf_model, data = cell_data)$predictions
# cell_data[, prediction := predictions]

cat("Pipeline complete. Ready for Random Forest prediction.\n")
```

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~6.46M string-paste + match ops | One sparse matrix build (~1.37M entries) | ~100× |
| Mean computation (per variable) | 6.46M `lapply` calls with `mean()` | Single sparse matrix multiply `A %*% V` (344K × 28) | ~200× |
| Max/Min computation (per variable) | 6.46M `lapply` calls with `max()/min()` | 28 year-passes with `data.table` grouped aggregation | ~30–50× |
| Total (5 variables) | ~86+ hours estimated | **~2–10 minutes expected** | ~500–2500× |

**Why it's faster:**
- **Sparse matrix multiply** (`A %*% V_zero`) is a single BLAS-backed operation that computes all neighbor sums for all cells across all years simultaneously. This replaces ~32M R-level function calls.
- **The adjacency matrix is built once** (16 MB) and reused for all 5 variables and all 28 years.
- **Year-column iteration for max/min** processes only 28 passes (not 6.46M), and within each pass, `data.table` grouped aggregation is C-optimized.
- **Memory peak** stays well under 16 GB: the largest single object is a 344K × 28 dense matrix (~77 MB).

**Numerical equivalence** is preserved because:
- The same rook neighbor sets are encoded in the sparse matrix.
- `mean = sum / count` with NA exclusion matches the original `mean(neighbor_vals[!is.na()])`.
- `max` and `min` are computed on identical neighbor value sets per cell-year.
- The Random Forest model is never retrained — only `predict()` is called on the enriched data.