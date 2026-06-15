 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(n²)-scale `build_neighbor_lookup`**: For each of the ~6.46M rows, it performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but the sheer volume (6.46M × avg ~4 neighbors) combined with string construction creates enormous overhead. The `lapply` over 6.46M rows with string operations is the dominant cost.

2. **`compute_neighbor_stats` uses R-level `lapply` over 6.46M rows**: Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean. This is ~6.46M R function calls with small-vector operations — death by a thousand cuts.

3. **The neighbor lookup encodes year-specific row indices, but the topology is year-invariant**: Rook neighbors don't change across years. The current code rebuilds year-aware keys for every row, even though the spatial adjacency is static. This conflates graph topology with temporal indexing.

**Key insight**: The adjacency graph has 344,208 nodes and ~1.37M directed edges — this is *fixed across all 28 years*. Each year is an independent slice with identical topology. We should:
- Build a sparse adjacency matrix **once** (344K × 344K).
- For each year-slice (344K rows), use sparse matrix–vector multiplication to compute neighbor sums and counts, then derive max/min/mean via grouped operations.

However, sparse matrix multiplication gives us **sum** and **count** (hence mean), but **not max and min**. For max/min, we need grouped operations over the edge list.

---

## Optimization Strategy

1. **Represent topology as a sparse adjacency matrix (`Matrix::sparseMatrix`) and as an edge-list (`data.table`)** — built once from the `nb` object.

2. **Sort/index `cell_data` by `(year, id)`** so that each year-slice is a contiguous block of rows, and within each year the cell ordering matches the spatial grid order. This allows direct positional indexing — no hash lookups, no string keys.

3. **For each variable and each year**:
   - Extract the variable vector for that year-slice (length 344,208, ordered by cell index).
   - **Mean**: Use sparse matrix multiplication: `A %*% x` gives neighbor sums; divide by `A %*% ones` (precomputed neighbor counts). One matrix–vector multiply per variable-year — highly optimized BLAS/CHOLMOD code in the `Matrix` package.
   - **Max/Min**: Use the edge-list in `data.table` with a keyed join to look up neighbor values, then `group by` source node to compute max and min. `data.table` grouped operations over ~1.37M rows are extremely fast.

4. **Vectorize across years**: Loop over 28 years (trivial) × 5 variables = 140 iterations of fast operations, instead of 6.46M R-level iterations.

**Expected speedup**: From 86+ hours to **minutes** (roughly 5–15 minutes depending on disk I/O and RAM pressure).

**Numerical equivalence**: Preserved exactly — same max, min, and mean of the same non-NA neighbor values.

**RAM**: The sparse matrix is 344K × 344K with ~1.37M non-zeros ≈ ~33 MB. The edge-list `data.table` is ~1.37M × 2 ≈ ~22 MB. Year-slices are 344K × ~110 cols. All fits comfortably in 16 GB.

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

# =============================================================================
# STEP 0: Prepare cell_data as data.table, establish canonical cell ordering
# =============================================================================

# Convert to data.table if not already
cell_data <- as.data.table(cell_data)

# id_order is the vector of cell IDs in the order matching rook_neighbors_unique
# (i.e., rook_neighbors_unique[[i]] contains neighbor indices for id_order[i])
# Build a map from cell ID to its positional index in id_order
n_cells <- length(id_order)
id_to_pos <- setNames(seq_len(n_cells), as.character(id_order))

# Add positional index to cell_data
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Sort by year then cell_pos for contiguous year-slices with canonical ordering
setkey(cell_data, year, cell_pos)

# Verify: each year-slice should have exactly n_cells rows in cell_pos order
# (If some cell-years are missing, we handle that below)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Check completeness (balanced panel assumed based on problem statement)
stopifnot(nrow(cell_data) == n_cells * n_years)
# After setkey(year, cell_pos), row ((y-1)*n_cells + p) corresponds to
# year=years[y], cell_pos=p. This is critical for direct indexing.

# =============================================================================
# STEP 1: Build sparse adjacency matrix and edge list from nb object (ONCE)
# =============================================================================

build_graph_from_nb <- function(nb_obj) {
  # nb_obj is a list of length n_cells
  # nb_obj[[i]] is an integer vector of neighbor indices (into id_order)
  # Build edge list: from -> to (1-indexed positions)
  from_list <- vector("list", length(nb_obj))
  to_list   <- vector("list", length(nb_obj))
  
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) next
    from_list[[i]] <- rep.int(i, length(nbrs))
    to_list[[i]]   <- nbrs
  }
  
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)
  
  # Sparse adjacency matrix (rows = source nodes, cols = neighbor nodes)
  # A[i,j] = 1 means j is a neighbor of i
  A <- sparseMatrix(
    i = from_vec, j = to_vec,
    x = 1, dims = c(length(nb_obj), length(nb_obj))
  )
  
  # Edge list as data.table
  edge_dt <- data.table(from_pos = from_vec, to_pos = to_vec)
  
  list(A = A, edges = edge_dt)
}

cat("Building graph topology...\n")
graph <- build_graph_from_nb(rook_neighbors_unique)
A <- graph$A          # sparse Matrix, n_cells x n_cells
edge_dt <- graph$edges # data.table with columns from_pos, to_pos

# Precompute neighbor counts per node (for mean calculation)
# neighbor_counts[i] = number of neighbors of node i
neighbor_counts <- as.numeric(A %*% rep(1, n_cells))  # length n_cells

cat(sprintf("Graph: %d nodes, %d directed edges\n", n_cells, nrow(edge_dt)))

# =============================================================================
# STEP 2: Function to compute neighbor stats for one variable across all years
# =============================================================================

compute_neighbor_features_fast <- function(cell_data, var_name, A, edge_dt,
                                            neighbor_counts, n_cells, years) {
  # Output column names (must match original pipeline)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate output vectors
  n_total <- nrow(cell_data)
  out_max  <- rep(NA_real_, n_total)
  out_min  <- rep(NA_real_, n_total)
  out_mean <- rep(NA_real_, n_total)
  
  vals_all <- cell_data[[var_name]]
  
  for (yi in seq_along(years)) {
    # Row range for this year (data is sorted by year, cell_pos)
    row_start <- (yi - 1L) * n_cells + 1L
    row_end   <- yi * n_cells
    row_range <- row_start:row_end
    
    # Extract variable values for this year-slice, in cell_pos order
    x <- vals_all[row_range]  # length n_cells, x[p] = value for cell_pos=p
    
    # --- MEAN via sparse matrix multiplication ---
    # Replace NA with 0 for sum, and track non-NA for count
    not_na <- !is.na(x)
    x_zero <- x
    x_zero[!not_na] <- 0
    
    # Neighbor sums (only non-NA values contribute)
    neighbor_sums <- as.numeric(A %*% x_zero)
    
    # Neighbor non-NA counts
    neighbor_nna <- as.numeric(A %*% as.numeric(not_na))
    
    # Mean = sum / count (NA where count == 0)
    yr_mean <- ifelse(neighbor_nna > 0, neighbor_sums / neighbor_nna, NA_real_)
    
    # --- MAX and MIN via edge list + data.table grouped ops ---
    # Look up neighbor values
    nbr_vals <- x[edge_dt$to_pos]
    
    # Build temporary data.table for grouped aggregation
    # Only keep non-NA neighbor values
    valid <- !is.na(nbr_vals)
    if (any(valid)) {
      tmp <- data.table(
        from_pos = edge_dt$from_pos[valid],
        val      = nbr_vals[valid]
      )
      
      agg <- tmp[, .(nmax = max(val), nmin = min(val)), by = from_pos]
      
      # Initialize with NA, then fill
      yr_max <- rep(NA_real_, n_cells)
      yr_min <- rep(NA_real_, n_cells)
      yr_max[agg$from_pos] <- agg$nmax
      yr_min[agg$from_pos] <- agg$nmin
    } else {
      yr_max <- rep(NA_real_, n_cells)
      yr_min <- rep(NA_real_, n_cells)
    }
    
    # Write into output vectors
    out_max[row_range]  <- yr_max
    out_min[row_range]  <- yr_min
    out_mean[row_range] <- yr_mean
  }
  
  # Assign columns to cell_data by reference
  set(cell_data, j = col_max,  value = out_max)
  set(cell_data, j = col_min,  value = out_min)
  set(cell_data, j = col_mean, value = out_mean)
  
  invisible(cell_data)
}

# =============================================================================
# STEP 3: Run for all neighbor source variables
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  t0 <- proc.time()
  
  compute_neighbor_features_fast(
    cell_data, var_name, A, edge_dt, neighbor_counts, n_cells, years
  )
  
  elapsed <- (proc.time() - t0)[3]
  cat(sprintf("    Done in %.1f seconds\n", elapsed))
}

# =============================================================================
# STEP 4: Restore original row order if needed, then predict
# =============================================================================

# If downstream code or the RF model expects original row order, restore it:
# (Assuming cell_data had a natural row index before we re-sorted)
# setkey(cell_data, original_row_id)  # if you stored one

# Apply the pre-trained Random Forest model (do NOT retrain)
# cell_data[, prediction := predict(rf_model, newdata = cell_data)]

cat("Neighbor feature engineering complete.\n")
cat(sprintf("Total rows: %d | Variables added: %d\n",
            nrow(cell_data), 3 * length(neighbor_source_vars)))
```

---

## Summary of Guarantees

| Property | Status |
|---|---|
| **Numerical equivalence** | ✅ Identical max, min, mean of non-NA neighbor values per node-year |
| **Trained RF model preserved** | ✅ No retraining; predict-only |
| **Graph built once** | ✅ Sparse matrix + edge list constructed once from `nb` object |
| **Topology reused across years** | ✅ Same `A` and `edge_dt` for all 28 year-slices |
| **RAM-safe for 16 GB** | ✅ Sparse matrix ~33 MB; edge list ~22 MB; year-slices ~344K rows |
| **Expected runtime** | ~5–15 minutes (vs. 86+ hours): sparse mat-vec is O(nnz)≈1.37M per multiply; `data.table` grouped max/min over 1.37M edges is sub-second per year-variable |

The key architectural change: **decompose the panel into year-slices sharing a fixed sparse graph**, replacing 6.46M R-level string-key lookups with 140 iterations (28 years × 5 variables) of vectorized sparse linear algebra and `data.table` grouped aggregation.