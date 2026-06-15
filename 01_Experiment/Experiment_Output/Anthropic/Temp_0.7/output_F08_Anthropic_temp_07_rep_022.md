 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates the static spatial topology with the time-varying data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows)** instead of recognizing that the neighbor graph is identical across all 28 years. It constructs 6.46 million lists of neighbor row-indices, each involving string concatenation (`paste`), named-vector lookups, and `NA` filtering. This is O(rows × avg_neighbors) with high constant factors due to R's string hashing.

2. **The lookup is year-entangled**: Each entry maps a *row* (cell-year) to other *rows* (neighbor cell-years). Since the neighbor topology is the same every year, this means the same spatial structure is redundantly encoded 28 times — once per year.

3. **`compute_neighbor_stats` iterates over 6.46M entries** with `lapply`, performing subsetting, `NA` removal, and summary statistics in pure R loops. This is repeated for each of the 5 variables, totaling ~32.3 million list operations.

4. **Memory pressure**: The `neighbor_lookup` object stores ~6.46M integer vectors. With an average of ~4 rook neighbors per cell, this is ~25.8M integers plus R list overhead — feasible but wasteful given 28× redundancy.

### Quantified Waste

| Component | Current Work | Optimized Work | Reduction |
|---|---|---|---|
| Neighbor lookup construction | 6.46M string-key lookups | 344K integer-index lookups (once) | **~19×** |
| Neighbor stat computation | 6.46M × 5 = 32.3M R-level iterations | 344K × 5 × 28 via vectorized matrix ops | **Orders of magnitude** |
| Total string operations | ~25.8M `paste()` calls | 0 | **Eliminated** |

---

## Optimization Strategy

**Core Insight**: Separate the *static spatial graph* from the *dynamic year-varying values*.

### Step 1: Build the neighbor lookup ONCE over cells, not cell-years

The `rook_neighbors_unique` (`spdep::nb` object) already encodes the spatial graph over the 344,208 cells. We simply need a mapping from each cell to its neighbor cells by *cell index* (position in `id_order`). This is a one-time operation over 344K cells.

### Step 2: Compute neighbor stats per year using vectorized matrix operations

For each year:
1. Extract the column vector of values for a given variable (e.g., `ntl`) for all 344K cells.
2. For each cell, gather neighbor values using the precomputed cell-level neighbor index.
3. Compute max, min, mean in a vectorized fashion.

**Key technique**: Convert the neighbor list into a sparse adjacency matrix (from the `Matrix` package). Then neighbor-mean is simply a sparse matrix–vector product divided by the neighbor count. Neighbor-max and neighbor-min can be computed via row-wise sparse operations.

### Step 3: Use `data.table` for fast group-by-year slicing and column assignment

This avoids repeated data.frame copying.

### Complexity Comparison

| | Current | Optimized |
|---|---|---|
| Neighbor lookup | O(6.46M × string ops) | O(344K × integer ops), once |
| Per-variable stats | O(6.46M × R list ops) | O(28 × sparse mat-vec on 344K) |
| Total time estimate | ~86+ hours | **~2–10 minutes** |

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ==============================================================================
# STEP 0: Prepare inputs
# ==============================================================================
# Assumptions about inputs:
#   - cell_data: data.frame or data.table with columns: id, year, ntl, ec, 
#                pop_density, def, usd_est_n2, and ~110 predictor columns.
#   - id_order: integer/character vector of length 344,208 giving the cell IDs
#               in the order corresponding to rook_neighbors_unique.
#   - rook_neighbors_unique: an spdep::nb object of length 344,208.
#   - rf_model: the pre-trained Random Forest model (untouched).

# Convert to data.table for performance (no copy if already data.table)
cell_data <- as.data.table(cell_data)

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ==============================================================================
# STEP 1: Build STATIC sparse adjacency matrix from the nb object (ONCE)
# ==============================================================================
# This encodes the rook-neighbor graph as a sparse matrix.
# Entry A[i,j] = 1 means cell j is a neighbor of cell i.

build_sparse_adjacency <- function(nb_obj) {
  n <- length(nb_obj)
  # Build COO (coordinate) format
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    neighs <- nb_obj[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    if (length(neighs) == 1L && neighs[1L] == 0L) next
    from <- c(from, rep(i, length(neighs)))
    to   <- c(to, neighs)
  }
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

cat("Building sparse adjacency matrix (one-time, ~344K cells)...\n")
t0 <- proc.time()
A <- build_sparse_adjacency(rook_neighbors_unique)
cat("  Done in", (proc.time() - t0)[3], "seconds.\n")

# Precompute the number of neighbors per cell (static)
neighbor_counts <- rowSums(A)  # numeric vector of length n_cells

# ==============================================================================
# STEP 2: Build a STATIC cell-ID-to-cell-index mapping
# ==============================================================================
# This lets us go from cell_data$id to the row index in the adjacency matrix.
id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))

# ==============================================================================
# STEP 3: Ensure cell_data is sorted by (year, id) and aligned to id_order
# ==============================================================================
# We need a fast way to extract a vector of values for all cells in a given year,
# aligned to the adjacency matrix row order (i.e., id_order).

# Create cell index column
cell_data[, cell_idx := id_to_cellidx[as.character(id)]]

# Sort by year and cell_idx for fast aligned extraction
setkey(cell_data, year, cell_idx)

# Verify alignment: for each year, we should have exactly n_cells rows
# and cell_idx should be 1:n_cells (complete panel assumed).
# If the panel is unbalanced, we handle NAs below.

# ==============================================================================
# STEP 4: Compute neighbor stats using sparse matrix operations
# ==============================================================================
# For neighbor MEAN:  (A %*% vals) / neighbor_counts
# For neighbor MAX and MIN: we need row-wise max/min over neighbor values.
#
# Sparse mat-vec gives us the SUM (and thus MEAN). For MAX and MIN, we use a
# direct approach with the neighbor list to avoid dense expansion, but we do it
# per-year on 344K cells (not 6.46M cell-years), which is fast.

# Pre-extract the neighbor list as a simple integer list (from the nb object)
# for max/min computation. This is done ONCE.
cat("Pre-extracting neighbor list as integer vectors (one-time)...\n")
t0 <- proc.time()
neighbor_list <- lapply(seq_len(n_cells), function(i) {
  neighs <- rook_neighbors_unique[[i]]
  if (length(neighs) == 1L && neighs[1L] == 0L) return(integer(0))
  neighs
})
cat("  Done in", (proc.time() - t0)[3], "seconds.\n")

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  cell_data[, (col_max)  := NA_real_]
  cell_data[, (col_min)  := NA_real_]
  cell_data[, (col_mean) := NA_real_]
}

# Main computation: loop over years (28 iterations) × variables (5 iterations)
# = 140 iterations, each operating on a 344K-length vector.

cat("Computing neighbor statistics...\n")
t_total <- proc.time()

for (yr in years) {
  cat("  Year:", yr, "\n")
  
  # Extract the subset for this year (already keyed by year, cell_idx)
  yr_rows <- cell_data[.(yr)]  # fast keyed subset
  
  # Get the row indices in cell_data for this year
  # Since we're keyed on (year, cell_idx), the rows for this year are contiguous
  row_indices <- cell_data[, which(year == yr)]
  
  # For a complete balanced panel, yr_rows should have n_cells rows
  # aligned to cell_idx 1:n_cells. Build a value vector aligned to id_order.
  # Handle potential missing cells gracefully.
  
  # Build aligned value vectors for each variable
  # yr_rows$cell_idx gives the cell index for each row in yr_rows
  cell_indices_this_year <- yr_rows$cell_idx
  
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Create a full-length vector aligned to id_order (NA for missing cells)
    vals <- rep(NA_real_, n_cells)
    vals[cell_indices_this_year] <- yr_rows[[var_name]]
    
    # --- Neighbor MEAN via sparse matrix-vector product ---
    # A %*% vals gives the sum of neighbor values for each cell.
    # Cells with all-NA neighbors will get 0 from the product, so we need
    # to handle NAs carefully.
    
    # Replace NAs with 0 for the matrix product, and track non-NA counts
    vals_nona <- vals
    vals_nona[is.na(vals_nona)] <- 0
    
    # Indicator of non-NA values
    valid <- as.numeric(!is.na(vals))
    
    neighbor_sum   <- as.numeric(A %*% vals_nona)    # sum of non-NA neighbor vals
    neighbor_valid <- as.numeric(A %*% valid)         # count of non-NA neighbors
    
    neighbor_mean_vec <- ifelse(neighbor_valid > 0,
                                neighbor_sum / neighbor_valid,
                                NA_real_)
    
    # --- Neighbor MAX and MIN via vectorized list operations ---
    # This iterates over 344K cells (not 6.46M), which is fast in R.
    neighbor_max_vec <- rep(NA_real_, n_cells)
    neighbor_min_vec <- rep(NA_real_, n_cells)
    
    for (i in seq_len(n_cells)) {
      nb_idx <- neighbor_list[[i]]
      if (length(nb_idx) == 0L) next
      nb_vals <- vals[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      neighbor_max_vec[i] <- max(nb_vals)
      neighbor_min_vec[i] <- min(nb_vals)
    }
    
    # Assign back to cell_data using row indices
    set(cell_data, i = row_indices, j = col_max,  value = neighbor_max_vec[cell_indices_this_year])
    set(cell_data, i = row_indices, j = col_min,  value = neighbor_min_vec[cell_indices_this_year])
    set(cell_data, i = row_indices, j = col_mean, value = neighbor_mean_vec[cell_indices_this_year])
  }
}

cat("Total neighbor stats time:", (proc.time() - t_total)[3], "seconds.\n")

# Clean up helper column
cell_data[, cell_idx := NULL]
```

However, the inner `for (i in seq_len(n_cells))` loop for max/min over 344K cells is still ~344K R-level iterations per year per variable. While far better than 6.46M, we can further optimize using `vapply` and, better yet, using **Rcpp** or a vectorized trick. Here is a fully vectorized version that eliminates even that loop:

```r
# ==============================================================================
# FULLY VECTORIZED VERSION (replaces the per-cell loop for max/min)
# ==============================================================================
# Strategy: Use the COO representation of the adjacency matrix to compute
# max and min via data.table grouping operations.

library(data.table)
library(Matrix)

# ---- STEP 1: Build static structures (ONCE) ----

cat("Building static spatial structures (one-time)...\n")
t0 <- proc.time()

n_cells <- length(id_order)
id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))

# Extract COO (coordinate) edges from the nb object
edges_from <- integer(0)
edges_to   <- integer(0)
for (i in seq_len(n_cells)) {
  neighs <- rook_neighbors_unique[[i]]
  if (length(neighs) == 1L && neighs[1L] == 0L) next
  edges_from <- c(edges_from, rep(i, length(neighs)))
  edges_to   <- c(edges_to, neighs)
}
# edges_dt: each row is a directed edge (from_cell -> to_cell means
# to_cell is a neighbor of from_cell)
edges_dt <- data.table(from_cell = edges_from, to_cell = edges_to)

# Also build sparse adjacency for mean computation
A <- sparseMatrix(i = edges_from, j = edges_to, x = 1,
                  dims = c(n_cells, n_cells))

cat("  Done in", (proc.time() - t0)[3], "seconds.\n")
cat("  Edge count:", nrow(edges_dt), "\n")

# ---- STEP 2: Prepare cell_data ----

cell_data <- as.data.table(cell_data)
cell_data[, cell_idx := id_to_cellidx[as.character(id)]]
setkey(cell_data, year, cell_idx)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  for (stat in c("neighbor_max_", "neighbor_min_", "neighbor_mean_")) {
    col_name <- paste0(stat, var_name)
    cell_data[, (col_name) := NA_real_]
  }
}

# ---- STEP 3: Compute neighbor stats per year, fully vectorized ----

cat("Computing neighbor statistics (vectorized)...\n")
t_total <- proc.time()

for (yr in years) {
  cat("  Year:", yr, "")
  t_yr <- proc.time()
  
  # Get row indices in cell_data for this year
  row_indices <- cell_data[, which(year == yr)]
  cell_indices_this_year <- cell_data$cell_idx[row_indices]
  
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    
    # Build value vector aligned to cell index (1:n_cells)
    vals <- rep(NA_real_, n_cells)
    vals[cell_indices_this_year] <- cell_data[[var_name]][row_indices]
    
    # --- MEAN via sparse matrix-vector product ---
    vals_nona <- vals
    vals_nona[is.na(vals_nona)] <- 0
    valid <- as.numeric(!is.na(vals))
    
    neighbor_sum   <- as.numeric(A %*% vals_nona)
    neighbor_valid <- as.numeric(A %*% valid)
    neighbor_mean_vec <- ifelse(neighbor_valid > 0,
                                neighbor_sum / neighbor_valid,
                                NA_real_)
    
    # --- MAX and MIN via data.table grouping on edges ---
    # For each edge (from_cell, to_cell), look up the value at to_cell.
    # Then group by from_cell and take max/min.
    edge_vals <- vals[edges_dt$to_cell]
    
    # Use data.table for fast grouped max/min
    edge_work <- data.table(
      from_cell = edges_dt$from_cell,
      val       = edge_vals
    )
    # Remove edges where neighbor value is NA
    edge_work <- edge_work[!is.na(val)]
    
    if (nrow(edge_work) > 0) {
      agg <- edge_work[, .(nmax = max(val), nmin = min(val)), by = from_cell]
      
      neighbor_max_vec <- rep(NA_real_, n_cells)
      neighbor_min_vec <- rep(NA_real_, n_cells)
      neighbor_max_vec[agg$from_cell] <- agg$nmax
      neighbor_min_vec[agg$from_cell] <- agg$nmin
    } else {
      neighbor_max_vec <- rep(NA_real_, n_cells)
      neighbor_min_vec <- rep(NA_real_, n_cells)
    }
    
    # Write results back to cell_data
    set(cell_data, i = row_indices, j = col_max,
        value = neighbor_max_vec[cell_indices_this_year])
    set(cell_data, i = row_indices, j = col_min,
        value = neighbor_min_vec[cell_indices_this_year])
    set(cell_data, i = row_indices, j = col_mean,
        value = neighbor_mean_vec[cell_indices_this_year])
  }
  
  cat("(", round((proc.time() - t_yr)[3], 1), "s)\n")
}

# Clean up
cell_data[, cell_idx := NULL]

cat("Total neighbor stats time:", round((proc.time() - t_total)[3], 1), "seconds.\n")

# ---- STEP 4: Predict with the pre-trained Random Forest (UNCHANGED) ----
# The trained model is used as-is. The computed neighbor features have
# identical numerical values to the original implementation.
# 
# Example (adjust to your actual prediction code):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-keyed list entries | 1 sparse matrix + 1 edge table (344K × 344K sparse) |
| **Neighbor mean** | R-level loop over 6.46M rows | Sparse matrix–vector product (28 × 5 = 140 calls) |
| **Neighbor max/min** | R-level loop over 6.46M rows | `data.table` grouped aggregation on ~1.37M edges (140 calls) |
| **Estimated runtime** | ~86+ hours | **~2–10 minutes** |
| **Memory** | ~6.46M R list entries + overhead | ~1.37M-entry sparse matrix + 1.37M-row edge table (~50 MB) |
| **Numerical equivalence** | Baseline | ✅ Identical (same max, min, mean over same neighbor sets) |
| **RF model** | Pre-trained | ✅ Untouched — no retraining |

The key architectural insight is: **factor the computation into (static topology) × (dynamic values per year)**, eliminating 28× redundant graph traversal and replacing R-level loops with sparse linear algebra and `data.table` grouped aggregations.