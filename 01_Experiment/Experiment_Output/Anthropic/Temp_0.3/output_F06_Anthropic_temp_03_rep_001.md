 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` — O(n²) row-level `lapply`

This function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's reference index in `id_to_ref`.
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character keys by pasting `id` and `year`.
4. Performs named-vector lookups (`idx_lookup[neighbor_keys]`).

Named-vector lookup in R is hash-based but still carries overhead per call. Doing this 6.46 million times with string construction (`paste`) and subsetting is extremely slow. The resulting `neighbor_lookup` list itself consumes substantial memory (a list of ~6.46M integer vectors).

### Bottleneck 2: `compute_neighbor_stats` — Row-level `lapply` over 6.46M rows × 5 variables

For each of the 5 variables, another `lapply` iterates over 6.46M entries, extracting neighbor values, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times, totaling ~32.3 million R-level function invocations. The `do.call(rbind, result)` on a 6.46M-element list is also very slow.

### Why raster focal/kernel operations don't directly apply

Raster focal operations (e.g., `terra::focal`) assume a **complete regular grid** with uniform kernel windows. This panel dataset has:
- Potentially irregular boundaries (not all cells have 4 rook neighbors).
- A temporal dimension (year) that must be matched exactly.
- Missing data patterns.

However, the **analogy is instructive**: focal operations are fast because they use vectorized matrix/array operations rather than row-by-row iteration. We should adopt the same principle using **sparse matrix multiplication and vectorized column operations**.

### Root cause summary

| Component | Calls | Per-call cost | Total |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M | String paste + named lookup | ~hours |
| `compute_neighbor_stats` | 6.46M × 5 | Subset + `max/min/mean` | ~hours |
| `do.call(rbind, ...)` | 5 times on 6.46M-element lists | Memory reallocation | ~minutes–hours |

---

## Optimization Strategy

### Strategy: Sparse Adjacency Matrix + Vectorized Group Operations

**Key insight**: Since the neighbor structure is *identical across all years* (rook contiguity is spatial, not temporal), we can:

1. **Expand the spatial neighbor graph to a cell-year adjacency graph** using a sparse matrix (one-time cost).
2. **Compute neighbor stats using sparse matrix operations** for `mean` and vectorized grouped operations for `max`/`min`.

Specifically:

- **Mean**: For a row-standardized sparse adjacency matrix `W`, `W %*% x` gives the neighbor mean directly. This is a single sparse matrix-vector multiply — extremely fast.
- **Max and Min**: These are not linear operations, so we can't use matrix multiplication. Instead, we use the sparse matrix structure to extract neighbor indices in bulk via `dgCMatrix` column pointers, then compute grouped max/min using `data.table` or vectorized C-level operations.

### Memory estimate

A sparse matrix for ~6.46M rows with ~4 neighbors each ≈ ~25.8M non-zero entries. At 12 bytes each (row index + column pointer + value in `dgCMatrix`), that's ~310 MB — fits in 16 GB RAM.

### Preserving the estimand

The optimized code computes **exactly the same** `max`, `min`, and `mean` of rook-neighbor values per cell-year as the original. The trained Random Forest model is not retrained — we only produce the same predictor columns faster.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# 
# Requirements: Matrix, data.table
# Preserves: exact numerical results, trained RF model (no retraining)
# =============================================================================

library(Matrix)
library(data.table)

# -------------------------------------------------------------------------
# Step 1: Build the cell-year sparse adjacency matrix (ONE TIME)
# -------------------------------------------------------------------------
build_cellyear_adjacency <- function(cell_data, id_order, rook_neighbors) {
  # Convert to data.table for fast keyed joins
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  
  n_rows <- nrow(dt)
  
  # Create a fast lookup: (id, year) -> row_idx
  # Using data.table keyed join
  setkey(dt, id, year)
  
  # Get unique years
  years <- sort(unique(dt$year))
  
  # Build mapping from cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Pre-allocate lists for sparse matrix triplets
  # Expected size: n_rows * ~4 neighbors (rook) = ~25.8M entries
  cat("Building adjacency triplets...\n")
  
  # For each cell in id_order, get its neighbor cell IDs
  # This is done once at the cell level (344,208 cells), not cell-year level
  n_cells <- length(id_order)
  
  # Build cell-level neighbor edge list: (from_cell_id, to_cell_id)
  from_cell <- integer(0)
  to_cell   <- integer(0)
  
  for (k in seq_along(id_order)) {
    nb_indices <- rook_neighbors[[k]]
    # Remove 0s (spdep convention for no neighbors)
    nb_indices <- nb_indices[nb_indices > 0]
    if (length(nb_indices) > 0) {
      from_cell <- c(from_cell, rep(id_order[k], length(nb_indices)))
      to_cell   <- c(to_cell, id_order[nb_indices])
    }
  }
  
  cat(sprintf("  Cell-level edges: %d\n", length(from_cell)))
  
  # Now expand to cell-year level using data.table join
  # For each edge (from_cell, to_cell), and for each year,
  # we need (row_idx_of_from_cell_year, row_idx_of_to_cell_year)
  
  edges_dt <- data.table(from_id = from_cell, to_id = to_cell)
  
  # Cross join edges with years
  years_dt <- data.table(year = years)
  edges_expanded <- edges_dt[, .(year = years), by = .(from_id, to_id)]
  
  cat(sprintf("  Expanded edges (before join): %d\n", nrow(edges_expanded)))
  
  # Join to get row indices for 'from' side
  lookup <- dt[, .(id, year, row_idx)]
  setkey(lookup, id, year)
  
  setnames(edges_expanded, c("from_id", "to_id", "year"))
  
  # Join from side
  edges_expanded[lookup, from_row := i.row_idx, on = .(from_id = id, year = year)]
  
  # Join to side
  edges_expanded[lookup, to_row := i.row_idx, on = .(to_id = id, year = year)]
  
  # Remove edges where either side is missing (cell-year doesn't exist)
  edges_expanded <- edges_expanded[!is.na(from_row) & !is.na(to_row)]
  
  cat(sprintf("  Valid cell-year edges: %d\n", nrow(edges_expanded)))
  
  # Build sparse adjacency matrix (from_row, to_row) = 1
  # Rows = "focal" cell-years, Cols = "neighbor" cell-years
  # So row i has 1s in columns corresponding to neighbors of cell-year i
  W <- sparseMatrix(
    i = edges_expanded$from_row,
    j = edges_expanded$to_row,
    x = 1,
    dims = c(n_rows, n_rows)
  )
  
  cat("Adjacency matrix built.\n")
  return(W)
}

# -------------------------------------------------------------------------
# Step 2: Compute neighbor stats using sparse matrix (PER VARIABLE)
# -------------------------------------------------------------------------
compute_neighbor_stats_fast <- function(cell_data, W, var_name) {
  cat(sprintf("Computing neighbor stats for: %s\n", var_name))
  
  x <- cell_data[[var_name]]
  n <- length(x)
  
  # --- Handle NAs ---
  # Replace NA with 0 for matrix multiplication, but track them
  is_na <- is.na(x)
  x_clean <- x
  x_clean[is_na] <- 0
  
  # Binary vector: 1 if not NA, 0 if NA
  x_valid <- as.numeric(!is_na)
  
  # --- Neighbor count (of non-NA values) ---
  # W %*% x_valid gives count of non-NA neighbors for each row
  neighbor_count <- as.numeric(W %*% x_valid)
  
  # --- Neighbor MEAN ---
  # W %*% x_clean gives sum of non-NA neighbor values
  neighbor_sum <- as.numeric(W %*% x_clean)
  neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # --- Neighbor MAX and MIN ---
  # These require non-linear operations; use the sparse matrix structure
  # Extract the adjacency as a dgCMatrix and iterate over rows efficiently
  
  # Convert to dgCMatrix (compressed sparse column) — but we need row access
  # So we transpose: W^T in CSC format gives us column access = row access of W
  Wt <- t(W)  # Now Wt is dgCMatrix; column j of Wt = row j of W = neighbors of j
  
  # Pre-allocate results
  neighbor_max <- rep(NA_real_, n)
  neighbor_min <- rep(NA_real_, n)
  
  # Access the internal structure of dgCMatrix
  # Wt@p: column pointers (length n+1)
  # Wt@i: row indices (0-based)
  p <- Wt@p
  row_i <- Wt@i  # 0-based row indices
  
  # Process in chunks to be cache-friendly
  # For each focal cell-year j, its neighbors are at row_i[(p[j]+1):p[j+1]] (converting to 1-based)
  
  # Vectorized approach using data.table for grouped max/min
  # Build a table of (focal_row, neighbor_value)
  
  # Total number of non-zero entries
  nnz <- length(row_i)
  cat(sprintf("  Processing %d neighbor links for max/min...\n", nnz))
  
  # Determine focal index for each non-zero entry
  # p is length n+1, p[j+1] - p[j] = number of entries in column j (0-indexed)
  # We need to map each entry index to its column index
  col_counts <- diff(p)  # length n
  focal_idx <- rep(seq_len(n), times = col_counts)  # focal cell-year index
  neighbor_idx <- row_i + 1L  # convert to 1-based
  
  # Get neighbor values (with NAs)
  nb_vals <- x[neighbor_idx]
  
  # Use data.table for grouped max/min, handling NAs
  edges_dt <- data.table(focal = focal_idx, val = nb_vals)
  edges_dt <- edges_dt[!is.na(val)]  # remove NA neighbor values
  
  if (nrow(edges_dt) > 0) {
    stats <- edges_dt[, .(nb_max = max(val), nb_min = min(val)), by = focal]
    neighbor_max[stats$focal] <- stats$nb_max
    neighbor_min[stats$focal] <- stats$nb_min
  }
  
  cat(sprintf("  Done: %s\n", var_name))
  
  # Return as a matrix matching original format: [max, min, mean]
  cbind(neighbor_max, neighbor_min, neighbor_mean)
}

# -------------------------------------------------------------------------
# Step 3: Add features to cell_data (PER VARIABLE)
# -------------------------------------------------------------------------
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, W) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, W, var_name)
  cell_data[[paste0(var_name, "_nb_max")]]  <- stats_mat[, 1]
  cell_data[[paste0(var_name, "_nb_min")]]  <- stats_mat[, 2]
  cell_data[[paste0(var_name, "_nb_mean")]] <- stats_mat[, 3]
  cell_data
}

# -------------------------------------------------------------------------
# Step 4: Main execution
# -------------------------------------------------------------------------

# Build adjacency matrix ONCE
cat("=== Building cell-year adjacency matrix ===\n")
W <- build_cellyear_adjacency(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, W)
}

cat("=== All neighbor features computed ===\n")

# The trained Random Forest model is NOT retrained.
# cell_data now has the same _nb_max, _nb_min, _nb_mean columns as before.
# Proceed to prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| **Neighbor lookup** | ~6.46M `paste` + named lookups (~hours) | Sparse matrix build via `data.table` join (~2–5 min) | ~60–100× |
| **Mean** (per var) | 6.46M `lapply` iterations (~hours) | Single sparse matrix–vector multiply (~seconds) | ~1000× |
| **Max/Min** (per var) | 6.46M `lapply` with subsetting (~hours) | `data.table` grouped aggregation (~30–60 sec) | ~100× |
| **Total (5 vars)** | **~86+ hours** | **~10–20 minutes** | **~250–500×** |

### Memory footprint

| Object | Size |
|---|---|
| Sparse matrix `W` (6.46M × 6.46M, ~25.8M non-zeros) | ~310 MB |
| Temporary `edges_dt` for max/min | ~200 MB peak |
| `cell_data` (~6.46M × 110 cols) | ~5.7 GB |
| **Total peak** | **~8–10 GB** (fits in 16 GB) |

### Why this preserves the estimand exactly

1. **Mean**: `(W %*% x_clean) / (W %*% x_valid)` is algebraically identical to computing `mean(neighbor_vals[!is.na(neighbor_vals)])` per row — the sparse matrix sums non-NA values and counts them separately.
2. **Max/Min**: The `data.table` grouped aggregation computes `max(val)` and `min(val)` over exactly the same sets of non-NA neighbor values as the original `lapply`.
3. **No model retraining**: The Random Forest model is untouched; only the feature engineering pipeline is accelerated.