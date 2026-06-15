 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It creates a list of 6.46 million elements, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *topology* is identical across all 28 years — cell A's neighbors are the same in 1992 as in 2019. The only thing that changes is which *rows* hold those neighbors' data for a given year.

2. **`compute_neighbor_stats` iterates over 6.46M entries** per variable, indexing into a flat vector by row positions that were expensively computed in step 1.

3. **String key construction** (`paste(id, year, sep="_")`) and named-vector lookups (`idx_lookup[neighbor_keys]`) are O(n) hash operations repeated ~6.46M × avg_neighbors times inside `build_neighbor_lookup`. This is the dominant cost.

4. **The lookup is rebuilt once but is itself the bottleneck** — a single 6.46M-element list of integer vectors built via `lapply` with millions of `paste` and named-vector index operations.

### The Key Insight

- **Static:** The neighbor graph (which cell IDs are neighbors of which cell IDs) — 344,208 cells, ~1.37M directed edges. This never changes.
- **Dynamic:** The variable values attached to each cell, which change by year. There are 28 year-slices.

The redesign should: (a) build a **cell-level** neighbor lookup once (344K entries, not 6.46M), and (b) compute neighbor stats **year-by-year** using fast vectorized/matrix operations on year-slices, reusing the same cell-level topology each year.

---

## Optimization Strategy

### 1. Build a Cell-Level Neighbor Lookup (Once, Static)

Convert `rook_neighbors_unique` (an `nb` object) into a simple cell-ID-indexed list. Each element maps a cell to the integer positions of its neighbors within the ordered cell vector. This is 344,208 entries — trivial.

### 2. Organize Data by Year, Indexed by Cell Order

For each year, extract a numeric vector (or matrix column) of variable values aligned to the canonical cell order. This allows direct integer indexing — no string keys, no hash lookups.

### 3. Vectorized Neighbor Aggregation Per Year

For each year × variable combination (28 × 5 = 140 iterations), use the static cell-level neighbor list to gather neighbor values and compute max/min/mean. Use optimized C++-backed operations via `data.table` or a sparse-matrix approach.

### 4. Optionally Use a Sparse Adjacency Matrix

Convert the neighbor list to a sparse matrix `W` (344,208 × 344,208). Then:
- **Neighbor mean** = `(W %*% x) / (W %*% ones)` — a single sparse matrix-vector multiply per year×variable.
- **Neighbor max/min** — requires row-wise operations over sparse structure, but can be done efficiently with compiled code.

### Estimated Speedup

| Aspect | Old | New |
|---|---|---|
| Lookup construction | 6.46M string-key entries | 344K integer-index entries (once) |
| Stat computation calls | 6.46M × 5 vars | 344K × 28 years × 5 vars (same total cells, but vectorized) |
| Key mechanism | `paste` + named vector lookup | Direct integer indexing |
| Expected time | ~86 hours | **~2–10 minutes** |

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 0: Ensure cell_data is a data.table, ordered consistently
# ==============================================================================
cell_data <- as.data.table(cell_data)

# Canonical cell order — must match the order used in rook_neighbors_unique (nb object)
# id_order is the vector of cell IDs in the order corresponding to the nb object indices
stopifnot(length(id_order) == length(rook_neighbors_unique))

# ==============================================================================
# STEP 1: Build STATIC cell-level neighbor lookup (once)
#
# rook_neighbors_unique is an nb object: a list of length N_cells where element i
# contains integer indices (into id_order) of neighbors of cell id_order[i].
# We keep it as-is — it's already the perfect static structure.
# ==============================================================================

# For the sparse matrix approach, build a sparse adjacency matrix W (N x N)
# where W[i,j] = 1 if cell j is a neighbor of cell i.
build_sparse_adjacency <- function(nb_obj) {
  n <- length(nb_obj)
  # Build COO triplets
  i_idx <- integer(0)
  j_idx <- integer(0)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    i_idx <- c(i_idx, rep(i, length(nbrs)))
    j_idx <- c(j_idx, nbrs)
  }
  sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n, n))
}

# More memory-efficient construction using pre-allocation
build_sparse_adjacency_fast <- function(nb_obj) {
  n <- length(nb_obj)
  # Count total edges
  total_edges <- sum(vapply(nb_obj, function(x) {
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1)))
  
  i_idx <- integer(total_edges)
  j_idx <- integer(total_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    len <- length(nbrs)
    i_idx[pos:(pos + len - 1L)] <- i
    j_idx[pos:(pos + len - 1L)] <- nbrs
    pos <- pos + len
  }
  sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n, n))
}

message("Building sparse adjacency matrix (static, once)...")
W <- build_sparse_adjacency_fast(rook_neighbors_unique)
# W is N_cells x N_cells, ~1.37M non-zero entries — very small in memory

# Number of neighbors per cell (static)
ones_vec <- rep(1, ncol(W))
n_neighbors <- as.numeric(W %*% ones_vec)  # length N_cells

message("Adjacency matrix built: ", nrow(W), " cells, ", nnzero(W), " edges.")

# ==============================================================================
# STEP 2: Create a cell-position index for fast alignment
# ==============================================================================

# Map each cell ID to its position in id_order (1-based index into rows of W)
cell_pos <- setNames(seq_along(id_order), as.character(id_order))

# Add cell position to cell_data
cell_data[, cell_pos_idx := cell_pos[as.character(id)]]

# Verify all cells are mapped
stopifnot(!anyNA(cell_data$cell_pos_idx))

# Get sorted unique years
all_years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

# ==============================================================================
# STEP 3: Compute neighbor stats year-by-year using static adjacency
#
# For each variable and each year:
#   - Extract a vector of length N_cells aligned to id_order
#   - Use sparse matrix multiplication for mean
#   - Use row-wise operations on sparse structure for max and min
# ==============================================================================

# Pre-key cell_data for fast year+cell lookups
setkey(cell_data, year, cell_pos_idx)

# Function to compute neighbor max and min using the nb list directly
# (sparse matrix multiply handles mean, but max/min need explicit iteration)
compute_neighbor_max_min <- function(vals, nb_obj) {
  n <- length(nb_obj)
  nmax <- rep(NA_real_, n)
  nmin <- rep(NA_real_, n)
  
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) next
    nv <- vals[nbrs]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    nmax[i] <- max(nv)
    nmin[i] <- min(nv)
  }
  list(nmax = nmax, nmin = nmin)
}

# Optimized version using Rcpp if available, otherwise pure R with vapply
# For 344K cells with avg ~4 neighbors, this loop is fast (~0.5-2 sec per call)
compute_neighbor_max_min_fast <- function(vals, nb_obj) {
  n <- length(nb_obj)
  result <- vapply(seq_len(n), function(i) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1L] == 0L) return(c(NA_real_, NA_real_))
    nv <- vals[nbrs]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_))
    c(max(nv), min(nv))
  }, numeric(2))
  # result is 2 x n matrix
  list(nmax = result[1L, ], nmin = result[2L, ])
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns in cell_data
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

message("Computing neighbor statistics for ", length(neighbor_source_vars),
        " variables across ", length(all_years), " years...")

total_iterations <- length(all_years) * length(neighbor_source_vars)
iter_count <- 0L

for (yr in all_years) {
  
  # Extract the subset for this year, ordered by cell_pos_idx
  # This gives us a vector aligned to id_order
  yr_rows <- cell_data[.(yr)]  # keyed lookup by year
  
  # Build a full-length vector for each variable, aligned to cell position
  # Some cells may be missing in some years; handle with NA
  # Create a mapping from cell_pos_idx to row index in yr_rows
  yr_cell_positions <- yr_rows$cell_pos_idx
  
  for (var_name in neighbor_source_vars) {
    iter_count <- iter_count + 1L
    
    # Build aligned vector: length = n_cells, indexed by cell position
    vals_aligned <- rep(NA_real_, n_cells)
    vals_aligned[yr_cell_positions] <- yr_rows[[var_name]]
    
    # --- Neighbor MEAN via sparse matrix multiplication ---
    # Handle NAs: compute sum of non-NA neighbor values / count of non-NA neighbors
    not_na <- !is.na(vals_aligned)
    vals_zero <- vals_aligned
    vals_zero[is.na(vals_zero)] <- 0
    
    neighbor_sum   <- as.numeric(W %*% vals_zero)
    neighbor_count <- as.numeric(W %*% as.numeric(not_na))
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    # Cells with no neighbors at all (n_neighbors == 0) should be NA
    neighbor_mean[n_neighbors == 0] <- NA_real_
    
    # --- Neighbor MAX and MIN via direct nb list iteration ---
    maxmin <- compute_neighbor_max_min_fast(vals_aligned, rook_neighbors_unique)
    
    # --- Write results back into cell_data ---
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Map back: for each row in yr_rows, its cell_pos_idx tells us which
    # element of the result vectors to read
    cell_data[year == yr, (col_max)  := maxmin$nmax[yr_cell_positions]]
    cell_data[year == yr, (col_min)  := maxmin$nmin[yr_cell_positions]]
    cell_data[year == yr, (col_mean) := neighbor_mean[yr_cell_positions]]
    
    if (iter_count %% 10 == 0 || iter_count == total_iterations) {
      message(sprintf("  Progress: %d / %d (year=%d, var=%s)",
                      iter_count, total_iterations, yr, var_name))
    }
  }
}

# Clean up helper column
cell_data[, cell_pos_idx := NULL]

message("Neighbor feature computation complete.")

# ==============================================================================
# STEP 4: Predict using the pre-trained Random Forest (unchanged)
# ==============================================================================
# The trained RF model is used as-is — no retraining.
# The column names (neighbor_max_*, neighbor_min_*, neighbor_mean_*) must match
# what the model expects. Adjust naming convention if the original code used
# different column names.
#
# Example (adjust to match your actual model object and column naming):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor relationships** | We use the identical `rook_neighbors_unique` nb object — same edges, same directionality. |
| **Same aggregation functions** | `max`, `min`, `mean` computed on the same set of non-NA neighbor values. |
| **NA handling** | Identical: neighbors with NA values are excluded; cells with zero valid neighbors get NA. |
| **Neighbor mean precision** | The sparse matrix approach computes `sum(non-NA values) / count(non-NA values)` — algebraically identical to `mean(vals[!is.na(vals)])`. |
| **No model retraining** | The Random Forest object is never touched — only the feature-engineering pipeline is redesigned. |
| **Column names** | Output columns follow the same `neighbor_{max,min,mean}_{varname}` convention. Adjust the prefix/suffix pattern if your trained model expects a different naming scheme. |

## Performance Summary

| Metric | Original | Redesigned |
|---|---|---|
| Lookup entries built | 6.46M (string-keyed) | 344K (integer-indexed, static) |
| Stat computation granularity | Per row (6.46M) | Per cell per year (344K × 28), vectorized |
| Sparse mat-vec multiplies | — | 140 (28 years × 5 vars), each ~0.01s |
| Max/min loops | 6.46M × 5 | 344K × 140, via `vapply` |
| **Estimated total time** | **~86 hours** | **~2–10 minutes** |
| Peak memory | ~several GB for lookup list | ~50 MB for sparse matrix + aligned vectors |