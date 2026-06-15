 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` constructs a per-row (per cell-year) neighbor lookup by iterating over ~6.46 million rows with string-key hashing**, and **`compute_neighbor_stats` iterates over the same ~6.46 million entries per variable with R-level `lapply` loops**. This means:

1. **Redundant topology construction**: The rook-neighbor graph is a *spatial* relationship — it is identical across all 28 years. Yet `build_neighbor_lookup` embeds year into every key and re-resolves neighbors for every cell-year row. With ~6.46M rows, this creates ~6.46M list entries, each requiring string concatenation, lookup, and NA filtering. This is O(N_rows × avg_degree) string operations.

2. **R-level loop over millions of rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M elements. R's interpreter overhead per iteration is ~1–5 µs, so even trivially fast bodies sum to hours.

3. **String-keyed lookups**: Using `paste(id, year, sep="_")` as hash keys and named vector indexing is extremely slow at scale compared to integer indexing.

4. **Per-variable recomputation**: `compute_neighbor_stats` is called 5 times (once per variable), each time iterating over the full 6.46M-row lookup.

**Summary**: The 86+ hour runtime is caused by millions of R-level iterations with string operations, repeated identically for each year and each variable, when the underlying graph topology is year-invariant.

---

## Optimization Strategy

### Key Insight: Separate Topology from Attributes

The rook-neighbor graph is purely spatial (344,208 nodes, ~1.37M directed edges). The yearly panel just replicates node attributes across 28 time slices. Therefore:

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 × 344,208, ~1.37M nonzeros). This is the graph topology.

2. **Reshape each variable into a cell × year matrix** (344,208 × 28).

3. **Compute neighbor aggregates via sparse matrix operations** — sparse matrix–dense matrix multiplication gives neighbor *sums* and neighbor *counts* in one shot, yielding **mean** directly. For **max** and **min**, we use grouped operations over the CSC/CSR structure.

4. **Vectorize everything** — eliminate all R-level row iteration.

### Complexity Comparison

| Step | Original | Optimized |
|------|----------|-----------|
| Topology | O(6.46M) string ops | O(1.37M) integer insertions (once) |
| Mean per var | O(6.46M) R iterations | One sparse mat × dense mat multiply |
| Max/Min per var | O(6.46M) R iterations | Vectorized grouped operation over CSR |
| Total R-loop iterations | ~32.3M (5 vars × 6.46M) | **Zero** |

Expected speedup: **~200–500×**, bringing runtime to **minutes**.

---

## Optimized R Code

```r
# =============================================================================
# Optimized Neighbor Aggregation Pipeline
# Preserves numerical equivalence with original max/min/mean statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)   # sparse matrices
library(data.table)

# ---- 1. Build sparse adjacency matrix ONCE from the nb object ---------------
# rook_neighbors_unique: spdep nb object (list of integer vectors of neighbor indices)
# id_order: vector of cell IDs in the order matching the nb object

build_sparse_adjacency <- function(nb_obj) {
  # nb_obj[[i]] contains integer indices of neighbors of node i
  # We build a CSC sparse matrix (dgCMatrix) of dimension n x n
  n <- length(nb_obj)
  
  # Pre-compute total edges for pre-allocation
  n_edges <- sum(vapply(nb_obj, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  # Build COO triplets
  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    k <- length(nbrs)
    from_idx[pos:(pos + k - 1L)] <- i
    to_idx[pos:(pos + k - 1L)]   <- nbrs
    pos <- pos + k
  }
  
  # A[i,j] = 1 means j is a neighbor of i (row i aggregates over its neighbors)
  sparseMatrix(i = from_idx, j = to_idx, x = 1, dims = c(n, n))
}

cat("Building sparse adjacency matrix...\n")
A <- build_sparse_adjacency(rook_neighbors_unique)
n_cells <- nrow(A)
cat(sprintf("  Nodes: %d, Edges (nnz): %d\n", n_cells, nnz(A)))

# ---- 2. Prepare cell_data as data.table for fast reshaping ------------------
# cell_data must have columns: id, year, and the neighbor_source_vars
# id_order defines the mapping from cell id to matrix row index

setDT(cell_data)

# Create integer node index matching the adjacency matrix row order
id_to_node <- setNames(seq_along(id_order), as.character(id_order))
cell_data[, node_idx := id_to_node[as.character(id)]]

# Sorted unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))
cell_data[, year_col := year_to_col[as.character(year)]]

# ---- 3. Functions for neighbor stats via sparse matrix ops -------------------

# For MEAN: A %*% X gives sum of neighbor values for each node.
#           A %*% (ones where X is not NA) gives count of non-NA neighbors.
#           mean = sum / count

# For MAX and MIN: We need grouped operations over the sparse structure.
# We iterate over columns of A (CSC format) or use the explicit edge list.

# Pre-extract the CSR structure for max/min (row-oriented access)
# Convert A to dgRMatrix (CSR) for efficient row-wise access
A_csr <- as(A, "RsparseMatrix")

compute_neighbor_aggregates_matrix <- function(val_vec, node_idx, year_col,
                                                n_cells, n_years, A, A_csr) {
  # Build cell x year matrix (NA for missing)
  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(node_idx, year_col)] <- val_vec
  
  # --- MEAN via sparse matrix multiplication ---
  not_na <- !is.na(X)
  X_zero <- X
  X_zero[is.na(X_zero)] <- 0  # replace NA with 0 for multiplication
  
  neighbor_sum   <- A %*% X_zero        # n_cells x n_years (Matrix)
  neighbor_count <- A %*% (not_na * 1)  # n_cells x n_years (Matrix)
  
  # Convert to dense
  neighbor_sum   <- as.matrix(neighbor_sum)
  neighbor_count <- as.matrix(neighbor_count)
  
  neighbor_mean <- neighbor_sum / neighbor_count  # NaN where count==0
  neighbor_mean[neighbor_count == 0] <- NA_real_
  
  # --- MAX and MIN via CSR row iteration (vectorized per year) ---
  # Strategy: for each year, use the sparse row pointers to do grouped max/min
  # We use the @p (row pointer) and @j (column index) slots of CSR
  
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # CSR slots: A_csr@p (length n_cells+1), A_csr@j (0-based col indices)
  row_ptr <- A_csr@p   # length n_cells + 1, 0-based
  col_j   <- A_csr@j   # 0-based column indices of nonzeros
  
  for (yr in seq_len(n_years)) {
    x_yr <- X[, yr]  # length n_cells, node values this year
    
    # For each node i, neighbors are col_j[(row_ptr[i]+1):row_ptr[i+1]]  (R 1-based)
    # Vectorized approach: expand neighbor values and use grouped ops
    
    # All neighbor values in edge order
    nbr_vals <- x_yr[col_j + 1L]  # col_j is 0-based, so +1
    
    # Build a group vector: which row (node) does each edge belong to?
    # row_ptr is cumulative count per row
    # Number of neighbors per row:
    row_lengths <- diff(row_ptr)  # length n_cells
    
    # Group index for each edge
    grp <- rep.int(seq_len(n_cells), times = row_lengths)
    
    # Remove NAs
    valid <- !is.na(nbr_vals)
    nbr_vals_v <- nbr_vals[valid]
    grp_v      <- grp[valid]
    
    if (length(nbr_vals_v) > 0) {
      # Use data.table for fast grouped max/min
      dt_tmp <- data.table(g = grp_v, v = nbr_vals_v)
      agg <- dt_tmp[, .(mx = max(v), mn = min(v)), keyby = g]
      
      neighbor_max[agg$g, yr] <- agg$mx
      neighbor_min[agg$g, yr] <- agg$mn
    }
  }
  
  # --- Extract results back to the original row order of cell_data ---
  idx <- cbind(node_idx, year_col)
  list(
    nb_max  = neighbor_max[idx],
    nb_min  = neighbor_min[idx],
    nb_mean = neighbor_mean[idx]
  )
}

# ---- 4. Outer loop: compute and attach neighbor features --------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")
t0 <- proc.time()

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  
  agg <- compute_neighbor_aggregates_matrix(
    val_vec  = cell_data[[var_name]],
    node_idx = cell_data$node_idx,
    year_col = cell_data$year_col,
    n_cells  = n_cells,
    n_years  = n_years,
    A        = A,
    A_csr    = A_csr
  )
  
  # Assign columns matching the original naming convention
  # (adjust prefixes/suffixes to match what the trained RF expects)
  set(cell_data, j = paste0(var_name, "_nb_max"),  value = agg$nb_max)
  set(cell_data, j = paste0(var_name, "_nb_min"),  value = agg$nb_min)
  set(cell_data, j = paste0(var_name, "_nb_mean"), value = agg$nb_mean)
}

elapsed <- (proc.time() - t0)[3]
cat(sprintf("Done in %.1f seconds.\n", elapsed))

# Clean up helper columns
cell_data[, c("node_idx", "year_col") := NULL]

# ---- 5. Predict with the pre-trained Random Forest (unchanged) --------------
# The trained model object (e.g., rf_model) is loaded from disk.
# Column names of the 110 predictor variables must match exactly.
# No retraining occurs.

# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **mean** | `mean(vals[idx])` where idx are non-NA neighbor rows | `(A %*% X_zero) / (A %*% not_na)` — identical sum/count with NA exclusion | ✅ Exact (floating point associativity ≤ 1 ULP) |
| **max** | `max(vals[idx])` over non-NA | `data.table [, max(v), by=g]` over identical edge-resolved non-NA values | ✅ Exact |
| **min** | `min(vals[idx])` over non-NA | `data.table [, min(v), by=g]` over identical edge-resolved non-NA values | ✅ Exact |
| **NA handling** | Returns `c(NA,NA,NA)` when no non-NA neighbors | Count==0 → NA for mean; missing group → NA for max/min | ✅ Exact |

## Memory Profile (16 GB laptop)

| Object | Size (approx) |
|--------|---------------|
| Sparse A (dgCMatrix, 1.37M nnz) | ~22 MB |
| A_csr (RsparseMatrix) | ~22 MB |
| X matrix (344K × 28, double) | ~77 MB |
| neighbor_sum, neighbor_count (same) | ~77 MB each |
| neighbor_max, neighbor_min (same) | ~77 MB each |
| cell_data (~6.46M × 110 cols) | ~5.7 GB |
| **Peak total** | **~6.5 GB** ✅ |

## Expected Runtime

- Sparse adjacency build: ~2 seconds
- Per variable (sparse matmul + grouped max/min over 28 years): ~30–60 seconds
- **Total for 5 variables: ~3–5 minutes** (vs. 86+ hours original)