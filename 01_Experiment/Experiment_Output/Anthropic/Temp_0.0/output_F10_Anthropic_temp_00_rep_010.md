 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The named-vector `idx_lookup` with 6.46M entries makes each lookup O(n) due to R's linear hashing on large named vectors.

2. **`compute_neighbor_stats` iterates over 6.46M list entries** per variable, extracting subsets of a numeric vector. With 5 variables, that's ~32.3 million R-level list iterations with per-element subsetting and `max/min/mean` calls. The overhead of millions of tiny R function calls dominates.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property — they don't change across years. But the lookup is built over the full cell-year panel, duplicating the same spatial graph 28 times and embedding year-matching logic in string keys. This inflates the problem from ~344K spatial nodes with ~1.37M edges to ~6.46M node-year rows with ~38.4M effective edge lookups.

**Root cause:** The implementation treats the panel as a flat table and rediscovers the spatial graph at every row, instead of separating the time-invariant topology from the time-varying attributes.

---

## Optimization Strategy

### Principle: Separate topology from attributes; vectorize aggregation via sparse matrix multiplication.

1. **Build the spatial adjacency once** as a sparse matrix `W` of dimension 344,208 × 344,208 with ~1.37M non-zero entries. This is the rook contiguity weight matrix (binary).

2. **For each year and each variable**, extract the attribute vector `x` (length 344,208), then compute:
   - **Neighbor sum** = `W %*% x`
   - **Neighbor count** = `W %*% (!is.na(x))` (to handle NAs correctly)
   - **Neighbor mean** = sum / count
   - **Neighbor max** and **min** via row-wise sparse operations

3. **Max and min** cannot be computed by matrix multiplication directly. Instead, use the sparse structure of `W` to do grouped max/min via `data.table` or direct C-level sparse row iteration. Alternatively, use the CSR representation of `W` to iterate rows in compiled code.

4. **Loop over 28 years × 5 variables = 140 iterations**, each operating on a vector of length ~344K with sparse matrix ops. This replaces 6.46M × 5 = 32.3M R-level list iterations.

5. **Memory:** The sparse matrix `W` with ~1.37M entries ≈ 33 MB. Each year-variable vector ≈ 2.6 MB. Total memory is well within 16 GB.

6. **Expected speedup:** From ~86 hours to **minutes**. The sparse matrix-vector multiply for mean is O(nnz) ≈ 1.37M operations per year-variable. Max/min via `data.table` grouped operations on the edge list is similarly O(nnz). Total: ~140 × O(1.37M) ≈ 192M operations — trivial for modern hardware.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE ENGINEERING
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, ... (~6.46M rows)
#   - id_order: integer vector of 344,208 cell IDs in canonical order
#   - rook_neighbors_unique: spdep nb object (list of length 344,208)
#   - rf_model: pre-trained Random Forest model (not retrained)
# =============================================================================

library(data.table)
library(Matrix)

# ---- Step 0: Convert cell_data to data.table if needed ----------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- Step 1: Build sparse adjacency matrix ONCE ----------------------------
# Convert spdep nb object to a sparse binary matrix W (344208 x 344208).
# This encodes the time-invariant rook topology.

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of integer vectors (neighbor indices), length n
  # Returns: sparse dgCMatrix of dimension n x n
  edges <- rbindlist(lapply(seq_len(n), function(i) {
    nbrs <- nb_obj[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(from = i, to = nbrs)
  }))
  
  sparseMatrix(
    i    = edges$from,
    j    = edges$to,
    x    = 1,
    dims = c(n, n),
    giveCsparse = TRUE
  )
}

n_cells <- length(id_order)
cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
W <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
cat("Adjacency matrix:", nrow(W), "x", ncol(W), "with", nnzero(W), "edges\n")

# ---- Step 2: Build edge list for max/min computation -----------------------
# Extract CSR-like edge list once for grouped max/min operations.

W_csr    <- as(W, "RsparseMatrix")  # Row-compressed form
edge_dt  <- data.table(
  from = rep(seq_len(n_cells), diff(W_csr@p)),
  to   = W_csr@j + 1L  # 0-based to 1-based
)
cat("Edge list:", nrow(edge_dt), "directed edges\n")

# ---- Step 3: Create cell-index mapping --------------------------------------
# Map each (id, year) to its row in cell_data, and each id to its position
# in id_order (spatial index).

# Spatial index: id -> position in id_order (1..n_cells)
id_to_spatial <- setNames(seq_len(n_cells), as.character(id_order))

# Add spatial index to cell_data
cell_data[, spatial_idx := id_to_spatial[as.character(id)]]

# Ensure data is keyed for fast year-based subsetting
setkey(cell_data, year, spatial_idx)

# Get sorted unique years
years <- sort(unique(cell_data$year))
cat("Years:", min(years), "-", max(years), "(", length(years), "years)\n")

# ---- Step 4: Compute neighbor stats via sparse operations -------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

cat("Computing neighbor statistics...\n")
t_start <- Sys.time()

for (yr in years) {
  # Extract the year-slice: a data.table keyed by spatial_idx
  # Because we keyed on (year, spatial_idx), binary search is fast
  yr_slice <- cell_data[.(yr)]
  
  # Build a full-length vector for each variable (indexed by spatial_idx)
  # Some cells may be missing in a given year; those remain NA.
  # yr_slice$spatial_idx gives the positions that are present.
  present_idx <- yr_slice$spatial_idx
  
  # Row indices in cell_data for this year (for writing back results)
  # We need the actual row positions in the original cell_data
  row_positions <- which(cell_data$year == yr)
  # These correspond 1:1 with yr_slice rows, in the same order (both sorted

  # by spatial_idx due to the key).
  
  for (var_name in neighbor_source_vars) {
    # Build full spatial vector (NA for missing cells)
    x_full <- rep(NA_real_, n_cells)
    x_full[present_idx] <- yr_slice[[var_name]]
    
    # --- Neighbor MEAN via sparse matrix-vector multiply ---
    # Replace NA with 0 for sum computation; track valid counts separately
    x_zero     <- x_full
    x_valid    <- rep(0, n_cells)
    valid_mask <- !is.na(x_full)
    x_zero[!valid_mask]  <- 0
    x_valid[valid_mask]  <- 1
    
    neighbor_sum   <- as.numeric(W %*% x_zero)
    neighbor_count <- as.numeric(W %*% x_valid)
    
    neighbor_mean <- ifelse(neighbor_count > 0, 
                            neighbor_sum / neighbor_count, 
                            NA_real_)
    
    # --- Neighbor MAX and MIN via edge list grouping ---
    # Look up neighbor values for all edges
    edge_dt[, val := x_full[to]]
    
    # Remove edges where neighbor value is NA
    valid_edges <- edge_dt[!is.na(val)]
    
    # Grouped max and min by 'from' node
    if (nrow(valid_edges) > 0) {
      agg <- valid_edges[, .(nmax = max(val), nmin = min(val)), by = from]
      
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
      neighbor_max[agg$from] <- agg$nmax
      neighbor_min[agg$from] <- agg$nmin
    } else {
      neighbor_max <- rep(NA_real_, n_cells)
      neighbor_min <- rep(NA_real_, n_cells)
    }
    
    # --- Write results back to cell_data for present cells ---
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    set(cell_data, i = row_positions, j = col_max,  value = neighbor_max[present_idx])
    set(cell_data, i = row_positions, j = col_min,  value = neighbor_min[present_idx])
    set(cell_data, i = row_positions, j = col_mean, value = neighbor_mean[present_idx])
  }
  
  cat(sprintf("  Year %d done (%.1f sec elapsed)\n", yr, 
              as.numeric(difftime(Sys.time(), t_start, units = "secs"))))
}

cat("Neighbor feature engineering complete.\n")
cat(sprintf("Total time: %.1f minutes\n", 
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))

# ---- Step 5: Clean up temporary column -------------------------------------
cell_data[, spatial_idx := NULL]

# ---- Step 6: Predict with pre-trained Random Forest (unchanged) -------------
# The RF model expects the same column names as the training data.
# The neighbor feature columns are named identically to the original pipeline:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, etc.
#
# Predict:
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|-----------|----------|-----------|-------------|
| **max** | `max(vals[neighbor_row_indices])` after NA removal | `data.table` grouped `max(val)` over the same edge set, same year | Identical: same set of non-NA neighbor values, same `max()` function |
| **min** | `min(vals[neighbor_row_indices])` after NA removal | `data.table` grouped `min(val)` over the same edge set, same year | Identical |
| **mean** | `mean(vals[neighbor_row_indices])` after NA removal | `(W %*% x_zero) / (W %*% x_valid)` = sum of non-NA values / count of non-NA values | Identical: algebraically equivalent to `mean()` of non-NA neighbors |
| **NA handling** | Returns `c(NA, NA, NA)` when no valid neighbors | `neighbor_count == 0` → NA; empty `valid_edges` group → NA | Identical |

The sparse matrix `W` encodes exactly the same directed edges as `rook_neighbors_unique`. The year-slicing ensures each cell's neighbors are looked up within the same year, matching the original `paste(id, year)` key logic.

---

## Performance Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup construction | ~6.46M string paste + named vector lookups | One-time sparse matrix build (~1.37M entries) |
| Per-variable computation | 6.46M R list iterations | 28 sparse mat-vec multiplies + 28 `data.table` group-bys |
| Total R-level iterations | ~32.3M list element calls | 140 vectorized operations |
| Estimated wall time | 86+ hours | **~5–15 minutes** |
| Peak memory | Large list of 6.46M integer vectors | Sparse matrix (~33 MB) + edge `data.table` (~33 MB) |
| RF model | Unchanged | Unchanged |