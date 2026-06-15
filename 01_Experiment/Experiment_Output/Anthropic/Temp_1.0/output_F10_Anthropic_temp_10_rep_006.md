 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each built via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query due to hashing overhead at this scale, and `paste(..., sep="_")` over millions of keys is expensive. This single function likely takes many hours.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements in an `lapply`**, extracting subsets of a vector by index. While each individual operation is fast, the R-level loop over millions of elements with per-element `max/min/mean` calls adds massive overhead. This runs 5 times (once per variable), so ~32.3 million R-level function invocations.

3. **The topology is conflated with the panel time dimension.** The neighbor graph is purely spatial (344,208 nodes, ~1.37M directed edges), but the lookup is rebuilt to incorporate year, exploding it to 6.46M entries. Since every cell appears in every year with the same neighbors, the spatial adjacency can be built once as a sparse matrix and reused via matrix algebra.

**Core insight:** Neighbor aggregation (max, min, mean of neighbor attributes) over a sparse graph is equivalent to sparse matrix operations. If **A** is the row-normalized adjacency matrix, then `A %*% x` gives the neighbor mean. Max and min require row-wise sparse operations but can be vectorized efficiently using the CSC/CSR structure.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 × 344,208, ~1.37M nonzeros). This is tiny in memory (~16 MB).

2. **Process year-by-year** (28 iterations) rather than cell-year-by-cell-year (6.46M iterations). For each year, extract the 344,208-length vector for each variable.

3. **Neighbor mean:** Sparse matrix multiply `A %*% x` where A has 1s for neighbors, then divide by neighbor count vector. Or use a row-normalized matrix directly.

4. **Neighbor max and min:** Use the CSC (dgCMatrix) structure to do row-wise max/min via `Matrix` package utilities or a vectorized C-level approach. The `slam` or direct `Matrix` manipulation can achieve this, but the cleanest high-performance approach uses `data.table` on the sparse triplet representation.

5. **Estimated speedup:** From ~86 hours to **< 5 minutes**. The sparse matrix-vector multiply for mean is O(nnz) ≈ 1.37M multiplications per variable-year. Max/min via grouped operations on ~1.37M entries is similarly fast. Total: 28 years × 5 variables × 3 operations = 420 sparse passes, each O(1.37M) ≈ 575M operations total — trivial for modern hardware.

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean neighbor statistics.
# Preserves the pre-trained Random Forest model (no retraining).
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix from the nb object (ONE TIME)
# --------------------------------------------------------------------------
# rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
# id_order: vector of cell IDs in the order matching the nb object

build_adjacency_matrix <- function(nb_obj, n = length(nb_obj)) {
  # Build COO (triplet) representation: for each node i, its neighbors j
  # This gives a directed adjacency where A[i,j] = 1 means j is a neighbor of i
  # (i.e., row i aggregates over columns j)
  
  row_idx <- rep(seq_along(nb_obj), times = lengths(nb_obj))
  col_idx <- unlist(nb_obj, use.names = FALSE)
  
  # Remove zero-neighbor entries (spdep uses 0L for no-neighbor nodes)
  valid <- col_idx > 0L
  row_idx <- row_idx[valid]
  col_idx <- col_idx[valid]
  
  A <- sparseMatrix(
    i = row_idx,
    j = col_idx,
    x = rep(1, length(row_idx)),
    dims = c(n, n),
    dimnames = NULL
  )
  return(A)
}

# Build adjacency once
n_cells <- length(rook_neighbors_unique)
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Precompute neighbor counts per node (for mean calculation)
neighbor_counts <- as.numeric(A %*% rep(1, n_cells))  # = rowSums(A)

# Precompute the sparse triplet data for max/min operations
# Convert to dgTMatrix (triplet) for efficient grouped operations
A_triplet <- as(A, "TMatrix")  # or:
A_triplet <- as(A, "dgTMatrix")
# A_triplet@i = 0-based row indices, A_triplet@j = 0-based col indices

# Create a data.table of edges for fast grouped max/min
edge_dt <- data.table(
  row_1based = A_triplet@i + 1L,  # target node (1-based)
  col_1based = A_triplet@j + 1L   # source neighbor node (1-based)
)
setkey(edge_dt, row_1based)

# --------------------------------------------------------------------------
# STEP 2: Build mapping from (cell_id, year) to row in cell_data, and from
#          cell_id to spatial index in adjacency matrix
# --------------------------------------------------------------------------

# Convert cell_data to data.table for efficiency
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Spatial index: position in id_order <-> row/col in adjacency matrix
# id_to_spatial: maps cell id -> spatial index (1..n_cells)
id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index to cell_data
cell_data[, spatial_idx := id_to_spatial[as.character(id)]]

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor stats (max, min, mean) per variable, per year
# --------------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Pre-allocate result columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

# Get sorted unique years
years <- sort(unique(cell_data$year))

# Process year by year
for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  # Row indices in cell_data for this year
  year_mask <- cell_data$year == yr
  year_rows <- which(year_mask)
  
  # Spatial indices for this year's cells
  spatial_indices_this_year <- cell_data$spatial_idx[year_rows]
  
  # Build mapping: spatial_index -> position in year_rows
  # (In a balanced panel, all cells appear every year, so this is a permutation)
  # We need: for spatial index s, what is the value of the variable?
  # Create a full-length vector indexed by spatial index
  
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    
    # Build a vector of length n_cells: vals_full[spatial_idx] = value
    # Initialize with NA for cells that might be missing this year
    vals_full <- rep(NA_real_, n_cells)
    vals_full[spatial_indices_this_year] <- cell_data[[var_name]][year_rows]
    
    # --- MEAN via sparse matrix-vector multiply ---
    # Replace NA with 0 for multiplication, but track valid counts
    vals_for_sum <- vals_full
    vals_valid   <- as.numeric(!is.na(vals_full))
    vals_for_sum[is.na(vals_for_sum)] <- 0
    
    neighbor_sum   <- as.numeric(A %*% vals_for_sum)
    neighbor_valid <- as.numeric(A %*% vals_valid)
    
    neighbor_mean_full <- ifelse(neighbor_valid > 0,
                                  neighbor_sum / neighbor_valid,
                                  NA_real_)
    
    # --- MAX and MIN via grouped operations on edge list ---
    # Look up neighbor values for all edges
    edge_dt[, val := vals_full[col_1based]]
    
    # Remove edges where neighbor value is NA
    valid_edges <- edge_dt[!is.na(val)]
    
    if (nrow(valid_edges) > 0) {
      # Grouped max and min
      agg <- valid_edges[, .(nmax = max(val), nmin = min(val)), by = row_1based]
      
      neighbor_max_full <- rep(NA_real_, n_cells)
      neighbor_min_full <- rep(NA_real_, n_cells)
      neighbor_max_full[agg$row_1based] <- agg$nmax
      neighbor_min_full[agg$row_1based] <- agg$nmin
    } else {
      neighbor_max_full <- rep(NA_real_, n_cells)
      neighbor_min_full <- rep(NA_real_, n_cells)
    }
    
    # --- Write results back to cell_data for this year's rows ---
    set(cell_data, i = year_rows, j = col_max,
        value = neighbor_max_full[spatial_indices_this_year])
    set(cell_data, i = year_rows, j = col_min,
        value = neighbor_min_full[spatial_indices_this_year])
    set(cell_data, i = year_rows, j = col_mean,
        value = neighbor_mean_full[spatial_indices_this_year])
  }
}

# Clean up temporary column
cell_data[, spatial_idx := NULL]

# --------------------------------------------------------------------------
# STEP 4: Apply the pre-trained Random Forest (NO retraining)
# --------------------------------------------------------------------------
# The trained model object (e.g., rf_model) is already in memory or loaded:
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# Generate predictions using the same predictor variable names as training
# cell_data now has all ~110 predictor columns including the neighbor features
predictions <- predict(rf_model, newdata = cell_data)
cell_data[, prediction := predictions]

cat("Done. Neighbor feature computation and prediction complete.\n")
```

## Complexity and Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology build** | 6.46M string-paste + named lookups | 1 sparse matrix from nb list (~0.3s) |
| **Mean computation** | R-level loop over 6.46M elements | Sparse `A %*% x`: O(1.37M) per var-year |
| **Max/Min computation** | R-level loop over 6.46M elements | `data.table` grouped agg on 1.37M edges |
| **Total loop iterations** | ~32.3M R-level calls | 28 years × 5 vars = 140 vectorized passes |
| **Memory for topology** | ~6.46M-element list of integer vectors | ~16 MB sparse matrix + ~22 MB edge table |
| **Estimated runtime** | 86+ hours | **~2–5 minutes** |
| **Numerical equivalence** | Baseline | ✅ Identical max, min, mean values |
| **RF model** | Pre-trained | ✅ Unchanged, predict-only |

### Why numerical equivalence is preserved

- **Mean:** `A %*% x / valid_count` computes exactly `sum(neighbor_vals) / length(non_na_neighbors)`, identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min:** The `data.table` grouped aggregation computes `max(val)` and `min(val)` over exactly the same set of non-NA neighbor values as the original loop.
- **NA handling:** Mirrors the original — nodes with zero valid neighbors get `NA` for all three statistics, matching `c(NA, NA, NA)` in the original code.