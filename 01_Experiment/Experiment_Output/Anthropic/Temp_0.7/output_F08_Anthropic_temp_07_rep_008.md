 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) indices.** It creates a list of 6.46 million entries, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. This is O(rows × avg_neighbors) string operations — roughly 6.46M × 4 ≈ 25.8 million string-match lookups.

2. **The neighbor topology is static.** Rook contiguity among 344,208 cells never changes across the 28 years. Yet the current code re-discovers neighbors for every cell-year row, duplicating work 28× unnecessarily.

3. **`compute_neighbor_stats` operates on the full 6.46M-row data frame.** Because the lookup indices point into the stacked cell×year data frame, every variable's neighbor stats require 6.46M list-element iterations with subsetting, NA checks, and summary computation — all in pure R loops.

4. **The outer loop repeats this for 5 variables**, compounding the cost: 5 × 6.46M = 32.3 million R-level `lapply` iterations.

### The Key Insight

> **Neighbor relationships are a property of cells (static). Variable values are a property of cell-years (dynamic).**

If we separate these two concerns, we can:
- Build the neighbor lookup **once over 344,208 cells** (not 6.46M rows).
- Compute neighbor stats **per year** using fast vectorized/matrix operations on 344,208-length vectors, not list-element-wise R loops over millions of rows.

---

## Optimization Strategy

### 1. Build a Cell-Level Neighbor Index (Once)

Construct a sparse adjacency structure over the 344,208 cells. This is simply a cleaned version of `rook_neighbors_unique` mapped to integer cell indices. Cost: negligible, done once.

### 2. Build a Sparse Adjacency Matrix (Once)

Convert the neighbor list into a sparse matrix `W` (344,208 × 344,208) using the `Matrix` package. Each row `i` has 1s in columns corresponding to cell `i`'s rook neighbors. This is the static topology encoded as a reusable linear-algebra object.

### 3. Compute Neighbor Stats via Sparse Matrix–Vector Products (Per Year, Per Variable)

For each year and each variable:
- Extract the 344,208-length variable vector `v` for that year.
- **Neighbor sum** = `W %*% v` (sparse matrix–vector multiply, highly optimized in C).
- **Neighbor count** = `W %*% (!is.na(v))` (to handle NAs correctly).
- **Neighbor mean** = sum / count.
- **Neighbor max/min**: Use a custom but vectorized approach with the neighbor list, or use row-wise sparse operations.

For **max and min**, sparse matrix multiplication doesn't directly apply, but we can use a fast vectorized approach over the cell-level neighbor list (344K iterations instead of 6.46M), or use `data.table` grouped operations.

### 4. Merge Back into the Panel

The results for each year are 344,208-length vectors. Map them back into the full data frame by (cell, year) alignment.

### Expected Speedup

| Component | Current | Optimized |
|---|---|---|
| Neighbor lookup construction | 6.46M string ops | 344K integer list (once) |
| Stat computation iterations | 5 vars × 6.46M = 32.3M | 5 vars × 28 years × 344K = 48.2M but vectorized |
| Per-iteration cost | R-level list subset + summary | C-level sparse matmul (mean) + vectorized (max/min) |
| **Estimated total time** | **86+ hours** | **~2–10 minutes** |

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build the static cell-level sparse adjacency matrix (ONCE)
# ==============================================================================

build_sparse_adjacency <- function(id_order, neighbors_nb) {
  # id_order: vector of 344,208 cell IDs in the order matching the nb object
  # neighbors_nb: spdep nb object (list of integer index vectors)
  #
  # Returns: a sparse logical/numeric adjacency matrix W (n x n)
  #          AND the id_order for alignment
  
  n <- length(id_order)
  stopifnot(length(neighbors_nb) == n)
  
  # Build COO (coordinate) triplets
  # For each cell i, neighbors_nb[[i]] gives integer indices of its neighbors
  from <- rep(seq_len(n), times = lengths(neighbors_nb))
  to   <- unlist(neighbors_nb)
  
  # Remove any 0-length entries (islands with no neighbors are handled naturally)
  valid <- !is.na(to) & to >= 1L & to <= n
  from  <- from[valid]
  to    <- to[valid]
  
  W <- sparseMatrix(
    i    = from,
    j    = to,
    x    = rep(1, length(from)),
    dims = c(n, n)
  )
  
  list(
    W        = W,
    id_order = id_order,
    id_to_idx = setNames(seq_len(n), as.character(id_order))
  )
}

# ==============================================================================
# STEP 2: Compute neighbor max, min, mean for one variable across all years
#          using the static adjacency
# ==============================================================================

compute_neighbor_features_fast <- function(dt, var_name, adj, neighbors_nb) {
  # dt:            data.table with columns: id, year, <var_name>
  # var_name:      character, the variable to compute neighbor stats for
  # adj:           output of build_sparse_adjacency()
  # neighbors_nb:  the raw nb list (for max/min computation)
  #
  # Returns: dt with three new columns added (modifies by reference)
  
  W        <- adj$W
  id_order <- adj$id_order
  id_to_idx <- adj$id_to_idx
  n        <- length(id_order)
  
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Pre-allocate result columns
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  
  # Ensure dt is keyed for fast subsetting
  setkey(dt, year, id)
  
  for (yr in years) {
    # Extract the rows for this year
    # We need a vector of length n aligned to id_order
    yr_rows <- dt[.(yr)]  # subset by year via key
    
    # Map yr_rows to the cell index order
    # yr_rows$id needs to be mapped to positions in id_order
    yr_idx <- id_to_idx[as.character(yr_rows$id)]
    
    # Build the aligned vector (length n, NA for missing cells)
    v <- rep(NA_real_, n)
    v[yr_idx] <- yr_rows[[var_name]]
    
    # --- Neighbor MEAN via sparse matrix multiplication ---
    # Handle NAs: replace NA with 0 for sum, track non-NA counts
    v_nona     <- v
    v_nona[is.na(v_nona)] <- 0
    not_na     <- as.numeric(!is.na(v))
    
    neighbor_sum   <- as.numeric(W %*% v_nona)       # length n
    neighbor_count <- as.numeric(W %*% not_na)        # length n
    
    neighbor_mean <- ifelse(neighbor_count > 0,
                            neighbor_sum / neighbor_count,
                            NA_real_)
    
    # --- Neighbor MAX and MIN via vectorized cell-level loop ---
    # This iterates over 344K cells (not 6.46M rows), which is fast
    neighbor_max <- rep(NA_real_, n)
    neighbor_min <- rep(NA_real_, n)
    
    for (i in seq_len(n)) {
      nb_idx <- neighbors_nb[[i]]
      if (length(nb_idx) == 0L) next
      nb_vals <- v[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      neighbor_max[i] <- max(nb_vals)
      neighbor_min[i] <- min(nb_vals)
    }
    
    # Map results back: for each row in yr_rows, get the result from its cell index
    result_max  <- neighbor_max[yr_idx]
    result_min  <- neighbor_min[yr_idx]
    result_mean <- neighbor_mean[yr_idx]
    
    # Write back into dt
    # We need the row indices in the original dt for this year
    row_indices <- which(dt$year == yr)
    # But since we used setkey, the order within dt[.(yr)] matches row_indices
    set(dt, i = row_indices, j = max_col,  value = result_max)
    set(dt, i = row_indices, j = min_col,  value = result_min)
    set(dt, i = row_indices, j = mean_col, value = result_mean)
  }
  
  invisible(dt)
}

# ==============================================================================
# STEP 2b: Even faster max/min using Rcpp (optional, drop-in replacement)
#           If Rcpp is available, this replaces the R-level for loop for max/min
# ==============================================================================

# If you want to avoid the 344K R-loop for max/min, use this Rcpp version:
#
# Rcpp::cppFunction('
# #include <Rcpp.h>
# using namespace Rcpp;
#
# // [[Rcpp::export]]
# List neighbor_max_min_cpp(NumericVector v, List neighbors_nb) {
#   int n = neighbors_nb.size();
#   NumericVector nmax(n, NA_REAL);
#   NumericVector nmin(n, NA_REAL);
#   for (int i = 0; i < n; i++) {
#     IntegerVector nb = neighbors_nb[i];
#     if (nb.size() == 0) continue;
#     double cmax = R_NegInf;
#     double cmin = R_PosInf;
#     bool found = false;
#     for (int j = 0; j < nb.size(); j++) {
#       int idx = nb[j] - 1;  // R is 1-indexed
#       if (idx < 0 || idx >= n) continue;
#       double val = v[idx];
#       if (NumericVector::is_na(val)) continue;
#       found = true;
#       if (val > cmax) cmax = val;
#       if (val < cmin) cmin = val;
#     }
#     if (found) {
#       nmax[i] = cmax;
#       nmin[i] = cmin;
#     }
#   }
#   return List::create(Named("max") = nmax, Named("min") = nmin);
# }
# ')

# ==============================================================================
# STEP 3: Main pipeline (replaces the original outer loop)
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # --- STATIC: Build adjacency once ---
  message("Building static sparse adjacency matrix (344,208 cells)...")
  adj <- build_sparse_adjacency(id_order, rook_neighbors_unique)
  message("  Done. Adjacency matrix: ", nrow(adj$W), " x ", ncol(adj$W),
          ", ", length(adj$W@x), " non-zero entries.")
  
  # --- DYNAMIC: Compute neighbor features per variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name, " ...")
    t0 <- proc.time()
    compute_neighbor_features_fast(
      dt           = cell_data,
      var_name     = var_name,
      adj          = adj,
      neighbors_nb = rook_neighbors_unique
    )
    elapsed <- (proc.time() - t0)[3]
    message("  Done in ", round(elapsed, 1), " seconds.")
  }
  
  message("All neighbor features computed.")
  return(cell_data)
}

# ==============================================================================
# USAGE (drop-in replacement for the original outer loop)
# ==============================================================================

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Now proceed directly to prediction with the pre-trained Random Forest:
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Changes and Guarantees

| Concern | Original | Redesigned |
|---|---|---|
| **Topology encoding** | Rebuilt per cell-year row (6.46M string lookups) | Sparse matrix built once over 344K cells |
| **Mean computation** | R `lapply` over 6.46M entries | Sparse matrix–vector product (`W %*% v`), C-optimized |
| **Max/Min computation** | R `lapply` over 6.46M entries | R loop over 344K cells × 28 years (or Rcpp) |
| **Total iterations** | ~32.3M R-level list operations | 5 vars × 28 yrs = 140 sparse matmuls + 140 × 344K vectorized loops |
| **Estimated runtime** | 86+ hours | **~2–10 minutes** |
| **Numerical equivalence** | — | Identical: same max, min, mean over same neighbor sets |
| **RF model** | — | Untouched; used only for prediction after features are computed |
| **Memory** | ~6.46M-element list of integer vectors | Sparse matrix (~1.4M entries ≈ 22 MB) + year-slice vectors |

The redesign cleanly separates the **static spatial topology** (encoded once as a sparse matrix and neighbor list) from the **dynamic yearly variable values** (processed per-year in vectorized operations), eliminating the redundant 28× recomputation of neighbor relationships and replacing millions of interpreted R loop iterations with optimized C-level sparse linear algebra.