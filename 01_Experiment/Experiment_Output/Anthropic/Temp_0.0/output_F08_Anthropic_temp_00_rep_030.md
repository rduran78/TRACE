 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It produces ~6.46 million list entries, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *topology* is identical across all 28 years — cell A's neighbors are the same in 1992 as in 2019. The only thing that changes is which *rows* hold those neighbors' data for a given year.

2. **`compute_neighbor_stats` iterates over ~6.46M list entries** for each of the 5 variables, performing subsetting and aggregation per row. That's ~32.3 million R-level list iterations total.

3. **String key construction (`paste(id, year)`) and named-vector lookups (`idx_lookup[neighbor_keys]`)** are O(n) in the number of rows and are repeated inside a per-row `lapply`. This is the single most expensive operation — approximately 6.46M × (avg ~4 neighbors) = ~25.8 million `paste` + hash-lookup operations just in `build_neighbor_lookup`.

### The Key Insight

The neighbor graph is **static** (cell-to-cell). The variables are **dynamic** (change by year). Therefore:

- Build the neighbor topology **once** at the cell level (344K entries, not 6.46M).
- For each variable, compute neighbor stats **per year** using vectorized matrix operations on the static topology, avoiding any per-row R loops.

---

## Optimization Strategy

### Step 1: Build a Static Cell-Level Neighbor Lookup (Once)

Convert `rook_neighbors_unique` (an `nb` object) into a cell-level adjacency structure indexed by integer position. This is just a direct reformat — 344,208 list entries. This is done **once** and reused forever.

### Step 2: Organize Data by Year for Vectorized Access

Split or index the data by year. For each year, create a fast mapping from cell ID to row index. Since cells are the same each year, if we sort by `(year, id)`, we can use direct integer indexing.

### Step 3: Vectorized Neighbor Aggregation via Sparse Matrix Multiplication

The most powerful optimization: represent the neighbor adjacency as a **sparse matrix** `W` (344,208 × 344,208). Then for each year and each variable:

- `neighbor_sum = W %*% x` (sum of neighbor values)
- `neighbor_count = W %*% (!is.na(x))` (count of non-NA neighbors)
- `neighbor_mean = neighbor_sum / neighbor_count`

For min and max, we use a sparse-matrix trick or a fast C++-backed grouped operation.

This reduces ~6.46M × 5 R-level loops to **28 × 5 = 140 sparse matrix-vector multiplications** (each taking milliseconds on 344K cells), plus 140 grouped min/max operations.

**Expected speedup: from ~86 hours to ~2–5 minutes.**

### Step 4: Preserve the Estimand

The numerical results (neighbor max, min, mean) are identical — we're just computing the same aggregation more efficiently. The trained Random Forest model is loaded and used as-is; no retraining occurs.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits: static topology + dynamic variables
# =============================================================================

library(Matrix)   # for sparse matrix operations
library(data.table)  # for fast grouped operations

#' Step 1: Build a sparse adjacency matrix from the nb object (ONCE)
#'
#' @param id_order   Integer vector of cell IDs in the order matching the nb object
#' @param neighbors  An nb object (list of integer index vectors) from spdep
#' @return A sparse logical/numeric matrix W of dimension (n_cells x n_cells)
build_adjacency_matrix <- function(id_order, neighbors) {
  n <- length(id_order)
  stopifnot(length(neighbors) == n)
  
  # Build COO (coordinate) representation
  # For each cell i, neighbors[[i]] gives the indices j of its neighbors
  from <- rep(seq_len(n), times = lengths(neighbors))
  to   <- unlist(neighbors)
  
  # Remove any 0-length entries (islands with no neighbors)
  valid <- !is.na(to)
  from  <- from[valid]
  to    <- to[valid]
  
  # Create sparse matrix (rows = focal cell, cols = neighbor cell)
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

#' Step 2: Compute neighbor max, min, mean for one variable across all years
#'
#' @param dt          data.table with columns: id, year, and the target variable
#' @param var_name    Character name of the variable
#' @param W           Sparse adjacency matrix (n_cells x n_cells)
#' @param id_order    Integer vector of cell IDs matching W's row/col order
#' @return data.table with columns: id, year, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(dt, var_name, W, id_order) {
  
  n_cells <- length(id_order)
  
  # Create a mapping from cell ID to matrix index (position in id_order)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  
  # Pre-index: for each cell, which matrix rows are its neighbors?
  # This is encoded in W already. We also need neighbor lists for min/max.
  # Extract neighbor lists from W (CSC format) — do this once.
  # We use W in dgCMatrix (column-compressed) form for matrix-vector multiply,
  # and extract row-wise neighbor lists for min/max.
  
  Wt <- t(W)  # transpose for efficient row-wise access via columns of Wt
  # For row i of W, the neighbors are the nonzero entries in column i of Wt
  neighbor_indices <- lapply(seq_len(n_cells), function(i) {
    Wt@i[seq.int(Wt@p[i] + 1L, Wt@p[i + 1L])] + 1L
  })
  
  # Pre-allocate output columns
  n_rows <- nrow(dt)
  nb_max  <- rep(NA_real_, n_rows)
  nb_min  <- rep(NA_real_, n_rows)
  nb_mean <- rep(NA_real_, n_rows)
  
  # Key the data.table for fast subsetting
  setkey(dt, year)
  
  # Add matrix index column (once)
  dt[, .mat_idx := id_to_idx[as.character(id)]]
  
  for (yr in years) {
    # Extract rows for this year
    yr_rows <- dt[.(yr), which = TRUE]
    
    if (length(yr_rows) == 0L) next
    
    # Build a values vector aligned to matrix indices
    # (some cells may be missing in a given year)
    vals_vec <- rep(NA_real_, n_cells)
    mat_indices <- dt$.mat_idx[yr_rows]
    vals_vec[mat_indices] <- dt[[var_name]][yr_rows]
    
    # --- MEAN via sparse matrix-vector multiplication ---
    # Replace NA with 0 for sum, track non-NA for count
    not_na <- !is.na(vals_vec)
    vals_zero <- vals_vec
    vals_zero[!not_na] <- 0
    
    neighbor_sum   <- as.numeric(W %*% vals_zero)       # sum of neighbor values
    neighbor_count <- as.numeric(W %*% as.numeric(not_na))  # count of non-NA neighbors
    
    mean_vec <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- MAX and MIN via vectorized grouped operation ---
    # Use the precomputed neighbor_indices list
    # This is the one part that still uses a loop, but it's over 344K cells
    # (not 6.46M rows) and each iteration is a simple numeric vector operation.
    max_vec <- rep(NA_real_, n_cells)
    min_vec <- rep(NA_real_, n_cells)
    
    # Vectorized approach using vapply over cells that exist this year
    # Only compute for cells that are present in this year's data
    active_cells <- mat_indices  # cells present this year
    
    max_min <- vapply(active_cells, function(ci) {
      nb_idx <- neighbor_indices[[ci]]
      if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_))
      nv <- vals_vec[nb_idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_))
      c(max(nv), min(nv))
    }, numeric(2))
    # max_min is a 2 x length(active_cells) matrix
    
    max_vec[active_cells] <- max_min[1L, ]
    min_vec[active_cells] <- max_min[2L, ]
    
    # Write results back to the output vectors at the correct row positions
    nb_max[yr_rows]  <- max_vec[mat_indices]
    nb_min[yr_rows]  <- min_vec[mat_indices]
    nb_mean[yr_rows] <- mean_vec[mat_indices]
  }
  
  # Clean up temp column
  dt[, .mat_idx := NULL]
  
  data.table(
    nb_max  = nb_max,
    nb_min  = nb_min,
    nb_mean = nb_mean
  )
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Convert to data.table if not already
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Step 1: Build sparse adjacency matrix ONCE (static topology)
message("Building sparse adjacency matrix (static topology)...")
W <- build_adjacency_matrix(id_order, rook_neighbors_unique)
message(sprintf("  Adjacency matrix: %d x %d, %d nonzero entries",
                nrow(W), ncol(W), nnzero(W)))

# Step 2: Compute neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for '%s'...", var_name))
  
  stats <- compute_neighbor_stats_fast(cell_data, var_name, W, id_order)
  
  # Add columns with original naming convention
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  
  cell_data[, (max_col)  := stats$nb_max]
  cell_data[, (min_col)  := stats$nb_min]
  cell_data[, (mean_col) := stats$nb_mean]
  
  message(sprintf("  Done: added %s, %s, %s", max_col, min_col, mean_col))
}

message("All neighbor features computed. Ready for Random Forest prediction.")

# Step 3: Predict with the pre-trained Random Forest (unchanged)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization: Eliminate the `vapply` Loop for Min/Max

The `vapply` over 344K cells per year (for min/max) takes roughly 1–3 seconds per year × 28 years × 5 variables ≈ 2–7 minutes total. If even that is too slow, here is an Rcpp-accelerated version:

```r
# Optional: Rcpp-accelerated min/max for neighbor aggregation
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_min_max_cpp(NumericVector vals, List neighbor_list) {
  int n = neighbor_list.size();
  NumericMatrix out(n, 2);  // col 0 = max, col 1 = min
  
  for (int i = 0; i < n; i++) {
    IntegerVector nb = neighbor_list[i];
    int m = nb.size();
    if (m == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      continue;
    }
    double vmax = R_NegInf;
    double vmin = R_PosInf;
    int valid = 0;
    for (int j = 0; j < m; j++) {
      double v = vals[nb[j] - 1];  // R is 1-indexed
      if (!NumericVector::is_na(v)) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        valid++;
      }
    }
    if (valid == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
    }
  }
  return out;
}
')
```

Replace the `vapply` block with:

```r
    # In the year loop, replace the vapply block with:
    mm <- neighbor_min_max_cpp(vals_vec, neighbor_indices)
    max_vec <- mm[, 1]
    min_vec <- mm[, 2]
```

This brings the min/max computation down to ~0.1 seconds per year-variable combination, making the entire pipeline complete in **under 1 minute**.

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup granularity | 6.46M cell-year rows | 344K cells (static) |
| Key construction | `paste(id, year)` × 6.46M | None (integer indexing) |
| Mean computation | R-level loop over 6.46M | Sparse matrix multiply (28 × 5 = 140 ops) |
| Min/Max computation | R-level loop over 6.46M | Loop over 344K cells per year (or Rcpp) |
| Estimated runtime | ~86+ hours | ~1–5 minutes |
| Numerical results | Identical | Identical |
| RF model | Unchanged | Unchanged |