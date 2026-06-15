 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine — O(1) hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for every neighbor (expensive string allocation × ~4 neighbors × 6.46M rows ≈ 25.8M string constructions).
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`) — named vector lookup in R is **O(n)** per query in the worst case because R's named vectors use linear hashing with potential collisions, and the vector has 6.46M entries.

**Net cost:** ~6.46M iterations × ~4 neighbors × expensive string ops + hash lookups = extremely slow. This alone could take hours.

### Bottleneck B: `compute_neighbor_stats` — repeated `lapply` over 6.46M rows, called 5 times

Each call iterates over all 6.46M rows, subsetting a numeric vector by index and computing `max`, `min`, `mean`. The R-level `lapply` loop with per-element function calls is slow due to interpreter overhead. Called 5 times (once per variable), this is ~32.3M R function invocations.

### Why raster focal/kernel operations are a useful analogy but not directly applicable

Raster focal operations (e.g., `terra::focal`) compute neighborhood statistics in C and are extremely fast — but they assume a **regular rectangular grid** where every cell has the same neighborhood structure. Here, the panel has an `spdep::nb` object (irregular neighbor structure, potentially with boundary cells having fewer neighbors, and the neighbor list is precomputed). However, the **key insight** from the focal analogy is: **we should vectorize the neighbor aggregation using sparse matrix multiplication or data.table joins rather than row-by-row R loops.**

### Memory estimate

6.46M rows × 110 columns × 8 bytes ≈ 5.7 GB. With neighbor features (5 vars × 3 stats = 15 new columns), we add ~0.77 GB. Total ~6.5 GB fits in 16 GB RAM, but we must avoid unnecessary copies.

---

## 2. Optimization Strategy

### Strategy: Sparse adjacency matrix + vectorized matrix operations

1. **Replace `build_neighbor_lookup`** with a sparse **row-adjacency matrix** `W` of dimension `(n_rows × n_rows)` where `n_rows = 6.46M`. Entry `W[i,j] = 1` if row `j` is a rook neighbor of row `i` **in the same year**. This matrix is constructed once using the spatial neighbor list and year matching.

2. **Replace `compute_neighbor_stats`** for `mean` with a single sparse matrix-vector multiplication: `W %*% x / row_counts`. For `max` and `min`, use a grouped operation via `data.table`.

3. **Key realization:** Since the spatial neighbor structure is **identical across all 28 years**, we can:
   - Build a small spatial adjacency matrix `W_spatial` (344,208 × 344,208) once.
   - For each variable, reshape to a wide matrix (344,208 rows × 28 columns), then compute neighbor stats using sparse matrix ops on each year-column simultaneously.

This avoids the 6.46M-row loop entirely.

### Expected speedup

- `build_neighbor_lookup`: eliminated entirely (replaced by sparse matrix construction, ~seconds).
- `compute_neighbor_stats`: replaced by sparse matrix multiplication (~seconds per variable).
- **Total estimated time: 2–10 minutes** instead of 86+ hours.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# 
# Prerequisites:
#   - cell_data: data.frame/data.table with columns 'id', 'year', and the
#     neighbor_source_vars. Rows are ordered consistently.
#   - id_order: vector of unique spatial cell IDs (same order as rook_neighbors_unique)
#   - rook_neighbors_unique: spdep::nb object (list of integer index vectors)
#   - The trained Random Forest model object (untouched)
#
# This code preserves the exact same numerical results as the original
# implementation: for each cell-year row, it computes max, min, and mean
# of each source variable across that cell's rook neighbors in the same year.
# =============================================================================

library(data.table)
library(Matrix)

# --------------------------------------------------------------------------
# STEP 1: Build the spatial sparse adjacency matrix (344,208 x 344,208)
# --------------------------------------------------------------------------
build_spatial_adjacency <- function(id_order, neighbors_nb) {
  n <- length(id_order)
  # Build COO (coordinate) format triplets
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  
  for (i in seq_len(n)) {
    nb_i <- neighbors_nb[[i]]
    # spdep::nb encodes "no neighbors" as a single 0; skip those
    if (length(nb_i) == 1L && nb_i[1L] == 0L) next
    from_list[[i]] <- rep.int(i, length(nb_i))
    to_list[[i]]   <- nb_i
  }
  
  row_idx <- unlist(from_list, use.names = FALSE)
  col_idx <- unlist(to_list,   use.names = FALSE)
  
  W <- sparseMatrix(
    i = row_idx,
    j = col_idx,
    x = rep.int(1, length(row_idx)),
    dims = c(n, n)
  )
  return(W)
}

cat("Building spatial adjacency matrix...\n")
W_spatial <- build_spatial_adjacency(id_order, rook_neighbors_unique)
cat("  Dimensions:", dim(W_spatial), "\n")
cat("  Non-zeros: ", nnzero(W_spatial), "\n")

# --------------------------------------------------------------------------
# STEP 2: Convert cell_data to data.table and create mapping
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure we know the mapping from cell id -> spatial index
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Get sorted unique years
years <- sort(unique(cell_data$year))
n_years <- length(years)
n_cells <- length(id_order)

cat("Panel: ", n_cells, "cells x", n_years, "years =", 
    n_cells * n_years, "potential rows\n")

# --------------------------------------------------------------------------
# STEP 3: For each variable, reshape to matrix, compute stats, reshape back
# --------------------------------------------------------------------------
# We reshape each variable into a (n_cells x n_years) matrix where
# row i corresponds to id_order[i] and column j corresponds to years[j].
# Then:
#   neighbor_mean = (W_spatial %*% X) / (W_spatial %*% valid_mask)
#   neighbor_max and neighbor_min require a different approach since
#   sparse matrix algebra doesn't directly support max/min.
#
# For max/min, we use an efficient grouped approach:
#   - Expand the neighbor pairs, join values, and aggregate.
#
# However, since we have the matrix form, we can iterate over cells
# in a VECTORIZED way per year using the sparse structure.
# --------------------------------------------------------------------------

# Pre-compute the neighbor list from the sparse matrix (CSC format) for max/min
# This is just the nb object re-indexed — we already have it.
# We'll use it for max/min via data.table.

# Create a long-form neighbor edge table (spatial only, ~1.37M rows)
cat("Building spatial edge table for max/min...\n")
edge_from <- vector("list", n_cells)
edge_to   <- vector("list", n_cells)
for (i in seq_len(n_cells)) {
  nb_i <- rook_neighbors_unique[[i]]
  if (length(nb_i) == 1L && nb_i[1L] == 0L) next
  edge_from[[i]] <- rep.int(i, length(nb_i))
  edge_to[[i]]   <- nb_i
}
edges_dt <- data.table(
  from_spatial = unlist(edge_from, use.names = FALSE),
  to_spatial   = unlist(edge_to,   use.names = FALSE)
)
cat("  Edge table rows:", nrow(edges_dt), "\n")

# Add spatial index to cell_data
cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Create a key for fast joining: (spatial_idx, year)
setkey(cell_data, spatial_idx, year)

# --------------------------------------------------------------------------
# STEP 4: Compute neighbor features for each variable
# --------------------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(cell_data, edges_dt, W_spatial,
                                          id_order, years, var_name) {
  cat("Processing variable:", var_name, "\n")
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # --- MEAN via sparse matrix multiplication (fastest) ---
  cat("  Computing neighbor means via sparse matmul...\n")
  
  # Build (n_cells x n_years) matrix
  # Map each row of cell_data to (spatial_idx, year_idx)
  year_to_idx <- setNames(seq_along(years), as.character(years))
  
  sp_idx  <- cell_data$spatial_idx
  yr_idx  <- year_to_idx[as.character(cell_data$year)]
  vals    <- cell_data[[var_name]]
  
  # Sparse matrix of values: rows = spatial cells, cols = years
  # For cells with NA, we need to handle them carefully
  valid_mask <- as.numeric(!is.na(vals))
  vals_clean <- ifelse(is.na(vals), 0, vals)
  
  X_val <- sparseMatrix(
    i = sp_idx, j = yr_idx, x = vals_clean,
    dims = c(n_cells, n_years)
  )
  X_mask <- sparseMatrix(
    i = sp_idx, j = yr_idx, x = valid_mask,
    dims = c(n_cells, n_years)
  )
  
  # Neighbor sums and counts
  neighbor_sum   <- W_spatial %*% X_val    # (n_cells x n_years)
  neighbor_count <- W_spatial %*% X_mask   # (n_cells x n_years)
  
  # Mean = sum / count (NA where count == 0)
  # Extract back to long form
  mean_vals <- numeric(nrow(cell_data))
  count_vals <- numeric(nrow(cell_data))
  
  # Extract efficiently
  for (j in seq_len(n_years)) {
    rows_j <- which(yr_idx == j)
    sp_j   <- sp_idx[rows_j]
    s_vec  <- neighbor_sum[, j]
    c_vec  <- neighbor_count[, j]
    mean_vals[rows_j]  <- s_vec[sp_j]
    count_vals[rows_j] <- c_vec[sp_j]
  }
  
  mean_result <- ifelse(count_vals == 0, NA_real_, mean_vals / count_vals)
  
  # --- MAX and MIN via data.table grouped join ---
  cat("  Computing neighbor max/min via data.table join...\n")
  
  # We need: for each (from_spatial, year), get all neighbor values and take max/min
  # Strategy: cross join edges with years, then join to get values, then aggregate
  
  # Create a lookup: (spatial_idx, year) -> value
  val_lookup <- cell_data[, .(spatial_idx, year, val = get(var_name))]
  setkey(val_lookup, spatial_idx, year)
  
  # Expand edges across all years efficiently
  # Instead of full cross join (1.37M * 28 = 38.4M rows), 
  # we do it year by year to control memory
  
  max_result <- rep(NA_real_, nrow(cell_data))
  min_result <- rep(NA_real_, nrow(cell_data))
  
  # Also build a lookup from (spatial_idx, year) -> row index in cell_data
  row_lookup <- cell_data[, .(spatial_idx, year, row_pos = .I)]
  setkey(row_lookup, spatial_idx, year)
  
  for (y in years) {
    # Get values for this year
    yr_vals <- val_lookup[year == y]
    setkey(yr_vals, spatial_idx)
    
    # Join edges to get neighbor values
    # edges_dt: from_spatial, to_spatial
    # We want: for each from_spatial, the val of each to_spatial in year y
    edge_vals <- edges_dt[yr_vals, on = .(to_spatial = spatial_idx), 
                          nomatch = 0L,
                          .(from_spatial, neighbor_val = i.val)]
    
    # Remove NA neighbor values
    edge_vals <- edge_vals[!is.na(neighbor_val)]
    
    if (nrow(edge_vals) == 0) next
    
    # Aggregate
    agg <- edge_vals[, .(nb_max = max(neighbor_val), 
                         nb_min = min(neighbor_val)), 
                     by = from_spatial]
    
    # Map back to cell_data rows
    agg_rows <- row_lookup[year == y]
    setkey(agg_rows, spatial_idx)
    setkey(agg, from_spatial)
    
    merged <- agg_rows[agg, on = .(spatial_idx = from_spatial), nomatch = 0L]
    
    max_result[merged$row_pos] <- merged$nb_max
    min_result[merged$row_pos] <- merged$nb_min
  }
  
  # --- Assign to cell_data ---
  max_col  <- paste0("max_neighbor_",  var_name)
  min_col  <- paste0("min_neighbor_",  var_name)
  mean_col <- paste0("mean_neighbor_", var_name)
  
  cell_data[, (max_col)  := max_result]
  cell_data[, (min_col)  := min_result]
  cell_data[, (mean_col) := mean_result]
  
  cat("  Done with", var_name, "\n")
  return(cell_data)
}

# --------------------------------------------------------------------------
# STEP 5: Run for all variables
# --------------------------------------------------------------------------
for (var_name in neighbor_source_vars) {
  cell_data <- compute_all_neighbor_features(
    cell_data, edges_dt, W_spatial, id_order, years, var_name
  )
}

# Clean up helper column
cell_data[, spatial_idx := NULL]

cat("All neighbor features computed.\n")

# --------------------------------------------------------------------------
# STEP 6: Apply the pre-trained Random Forest (unchanged)
# --------------------------------------------------------------------------
# The trained model object (e.g., `rf_model`) is used as-is.
# Ensure column names match what the model expects.
# Example:
#   predictions <- predict(rf_model, newdata = cell_data)
# --------------------------------------------------------------------------
```

---

## 4. Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-iteration `lapply` with string `paste` + named vector lookup | Sparse matrix `W_spatial` built once from `nb` object (~seconds) |
| **Mean computation** | Per-row R loop × 5 vars = 32.3M function calls | Sparse matrix multiplication `W %*% X` — fully vectorized in C (~seconds per variable) |
| **Max/Min computation** | Per-row R loop with subsetting | `data.table` grouped join + aggregation, looped over 28 years (~seconds per variable) |
| **Total estimated time** | 86+ hours | **~2–10 minutes** |
| **Memory** | Multiple large intermediate lists | Sparse matrices + in-place `data.table` columns; fits in 16 GB |
| **Numerical results** | Original estimand | **Exactly preserved** — same max, min, mean over non-NA rook-neighbor values per cell-year |
| **RF model** | Pre-trained | **Untouched** — applied after feature computation |

### Why this preserves the original numerical estimand

- **Mean:** `(W %*% x) / (W %*% valid_mask)` computes exactly `sum(neighbor_vals) / count(non_NA_neighbor_vals)` — identical to `mean(neighbor_vals[!is.na(neighbor_vals)])`.
- **Max/Min:** The `data.table` join replicates the exact same subsetting logic: for each cell-year, gather all rook-neighbor values in the same year, drop NAs, and take `max`/`min`.
- **NA handling:** Cells with zero valid neighbors get `NA` for all three statistics, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`.