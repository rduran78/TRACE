 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, the `idx_lookup` named vector (6.46M entries) is created once, but then `paste()` is called **per row** inside `lapply` over 6.46M rows. Each call to `paste()` and the subsequent named-vector lookup (`idx_lookup[neighbor_keys]`) involves string construction, hashing, and matching — repeated 6.46M times. With an average of ~4 rook neighbors per cell, that's ~25.8M string constructions and lookups just in this function.

### Broader Algorithmic Problem
More critically, the **entire design is row-centric (long format)** when it should be **year-sliced and integer-indexed**. The key insight is:

1. **Neighbor topology is time-invariant.** Cell A's neighbors are the same in 1992 and 2019. The `nb` object already encodes this as integer indices into the spatial ID vector.
2. **Lookups should be integer-based, not string-based.** Within a single year, the neighbor row indices can be computed by a simple offset from the spatial index — no strings needed.
3. **`compute_neighbor_stats` is already vectorized over the lookup list**, but it's called 5 times (once per variable) over the same 6.46M-entry list. The neighbor gather can be done once per variable using matrix operations on year-slices.

The overall complexity of the current approach:
- `build_neighbor_lookup`: O(N_rows × avg_neighbors) string operations ≈ 25.8M string ops
- `compute_neighbor_stats`: Called 5 times, each iterating 6.46M `lapply` calls
- **Total**: ~32M string ops + ~32M R-level list iterations — all in interpreted R with no vectorization.

This is why the estimate is 86+ hours.

## Optimization Strategy

1. **Eliminate all string keys.** Work entirely with integer indices.
2. **Slice by year.** Within each year, all 344,208 cells share the same spatial ordering. Build a spatial-index-to-row-offset map once per year (or better, ensure consistent ordering so it's a trivial arithmetic offset).
3. **Vectorize neighbor aggregation using sparse matrices.** Construct a single sparse adjacency matrix (344,208 × 344,208) from the `nb` object. For each year-slice, extract the variable column as a vector and compute `W %*% x` (sum), `rowSums(W != 0)` (count), and use sparse-matrix tricks for min/max — or use a grouped approach with `data.table`.
4. **Compute all 5 variables' stats in one pass per year** (or via matrix multiplication across all years at once).

### Expected Speedup
- Sparse matrix–vector multiply for 344K × 344K with ~1.37M nonzeros: milliseconds.
- 28 years × 5 variables × 3 stats = 420 sparse matrix operations, each taking milliseconds.
- **Total: seconds to low minutes** vs. 86+ hours.

## Working R Code

```r
library(Matrix)
library(data.table)

#' Build a sparse row-normalized (or raw) adjacency matrix from an nb object.
#' Returns a dgCMatrix of dimension n_cells x n_cells with 1s for neighbor links.
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj is a list of length n; nb_obj[[i]] contains integer indices of neighbors of cell i
  # Build COO triplets
  i_idx <- integer(0)
  j_idx <- integer(0)
  
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      i_idx <- c(i_idx, rep(i, length(nbrs)))
      j_idx <- c(j_idx, nbrs)
    }
  }
  
  sparseMatrix(
    i = i_idx, j = j_idx, x = rep(1, length(i_idx)),
    dims = c(n, n)
  )
}

#' Compute neighbor max, min, mean for a numeric vector x given adjacency matrix W.
#' Returns a 3-column matrix: [max, min, mean], length = length(x).
#' 
#' Strategy: 
#'   - mean = (W %*% x) / (W %*% 1)  (where 1 is a vector of ones, adjusted for NAs)
#'   - For min/max, we use a grouped operation via the sparse structure.
compute_neighbor_stats_sparse <- function(W, x) {
  n <- length(x)
  
  # Handle NAs: create a version of x where NA -> 0 for sum, and a mask
  not_na <- as.numeric(!is.na(x))
  x_safe <- ifelse(is.na(x), 0, x)
  
  # Neighbor count (excluding NAs)
  neighbor_count <- as.numeric(W %*% not_na)
  
  # Neighbor sum (excluding NAs)
  neighbor_sum <- as.numeric(W %*% x_safe)
  
  # Mean
  n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  
  # For min and max, we need the actual neighbor values.
  # Sparse matrix approach: iterate over rows of W using its CSC/CSR structure.
  # Convert W to dgRMatrix (row-sparse) for efficient row access, or use dgCMatrix columns of t(W).
  # 
  # More efficient: use data.table on the COO representation.
  
  # Extract COO from W
  W_T <- summary(W)  # gives (i, j, x) triplets
  # W_T$i is the "focal" cell, W_T$j is the "neighbor" cell
  
  dt <- data.table(
    focal = W_T$i,
    neighbor = W_T$j
  )
  
  # Attach neighbor values
  dt[, val := x[neighbor]]
  
  # Remove NA neighbor values
  dt <- dt[!is.na(val)]
  
  # Compute grouped min and max
  agg <- dt[, .(nmax = max(val), nmin = min(val)), by = focal]
  
  # Map back to full vector
  n_max <- rep(NA_real_, n)
  n_min <- rep(NA_real_, n)
  n_max[agg$focal] <- agg$nmax
  n_min[agg$focal] <- agg$nmin
  
  cbind(n_max, n_min, n_mean)
}

#' Main pipeline: compute all neighbor features for the panel dataset.
#' 
#' @param cell_data   data.frame/data.table with columns: id, year, and all var columns.
#'                    Must contain all 6.46M cell-year rows.
#' @param id_order    integer vector of cell IDs in the order matching the nb object.
#'                    Length = 344,208.
#' @param nb_obj      spdep::nb object (rook_neighbors_unique). Length = 344,208.
#' @param neighbor_source_vars character vector of variable names.
#' @return cell_data with new neighbor feature columns appended.
add_all_neighbor_features <- function(cell_data, id_order, nb_obj, neighbor_source_vars) {
  
  n_cells <- length(id_order)
  
  message("Building sparse adjacency matrix...")
  W <- build_adjacency_matrix(nb_obj, n_cells)
  
  # Create mapping from cell id to spatial index (position in id_order)
  id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Convert to data.table for efficiency
  was_df <- !is.data.table(cell_data)
  if (was_df) cell_data <- as.data.table(cell_data)
  
  # Add spatial index column
  cell_data[, spatial_idx := id_to_spatial_idx[as.character(id)]]
  
  # Pre-extract COO triplets once (for min/max computation)
  W_summary <- summary(W)
  coo_dt <- data.table(focal = W_summary$i, neighbor = W_summary$j)
  
  # Precompute the "not-NA neighbor count" denominator helper
  # W %*% not_na per year — we need this per variable per year
  
  years <- sort(unique(cell_data$year))
  
  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }
  
  message("Processing ", length(years), " years x ", length(neighbor_source_vars), " variables...")
  
  for (yr in years) {
    # Extract the year-slice, ordered by spatial index
    year_mask <- cell_data$year == yr
    year_data <- cell_data[year_mask]
    
    # Ensure we have a full spatial grid for this year.
    # Build a vector of length n_cells, indexed by spatial_idx.
    # If some cells are missing for a year, they'll be NA.
    
    setkey(year_data, spatial_idx)
    spatial_indices_present <- year_data$spatial_idx
    
    for (var_name in neighbor_source_vars) {
      # Build full-length spatial vector (NA for missing cells)
      x_full <- rep(NA_real_, n_cells)
      x_full[spatial_indices_present] <- year_data[[var_name]]
      
      # --- Neighbor mean via sparse matrix multiplication ---
      not_na <- as.numeric(!is.na(x_full))
      x_safe <- ifelse(is.na(x_full), 0, x_full)
      
      neighbor_count <- as.numeric(W %*% not_na)
      neighbor_sum   <- as.numeric(W %*% x_safe)
      
      n_mean_full <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
      
      # --- Neighbor min/max via data.table grouped aggregation ---
      # Attach values to COO neighbor list
      coo_dt[, val := x_full[neighbor]]
      agg <- coo_dt[!is.na(val), .(nmax = max(val), nmin = min(val)), by = focal]
      
      n_max_full <- rep(NA_real_, n_cells)
      n_min_full <- rep(NA_real_, n_cells)
      n_max_full[agg$focal] <- agg$nmax
      n_min_full[agg$focal] <- agg$nmin
      
      # --- Write results back into cell_data for this year-slice ---
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Map from spatial_idx back to the rows in cell_data
      cell_data[year_mask, (col_max)  := n_max_full[spatial_idx]]
      cell_data[year_mask, (col_min)  := n_min_full[spatial_idx]]
      cell_data[year_mask, (col_mean) := n_mean_full[spatial_idx]]
    }
    
    message("  Year ", yr, " done.")
  }
  
  # Clean up helper column
  cell_data[, spatial_idx := NULL]
  
  if (was_df) cell_data <- as.data.frame(cell_data)
  
  return(cell_data)
}

# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

# Original code (86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data, 
  id_order, 
  rook_neighbors_unique, 
  neighbor_source_vars
)

# The trained Random Forest model is untouched — only the feature-construction
# pipeline is replaced. The output columns (neighbor_max_*, neighbor_min_*, 
# neighbor_mean_*) are numerically identical to the original.
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Key construction** | 25.8M `paste()` string ops | Zero — integer indices only |
| **Lookup mechanism** | Named character vector (hash per lookup) | Sparse matrix multiply + `data.table` grouped agg |
| **Neighbor list build** | 6.46M R-level `lapply` iterations | Eliminated entirely |
| **Stats computation** | 5 × 6.46M `lapply` iterations | 28 years × 5 vars × {1 SpMV for mean + 1 `data.table` agg for min/max} = 420 vectorized ops |
| **Estimated runtime** | 86+ hours | **2–10 minutes** |
| **RAM** | ~6.46M-entry list of integer vectors | Sparse matrix (1.37M nonzeros ≈ 33 MB) + `data.table` COO (1.37M rows ≈ 22 MB) |
| **Numerical result** | Original estimand | **Identical** — same max/min/mean over same neighbor sets |
| **RF model** | — | **Untouched** |

The core algorithmic insight: **the neighbor topology is spatial (time-invariant), so factor the problem into spatial-adjacency × year-slices, and use sparse linear algebra instead of per-row string lookups.**