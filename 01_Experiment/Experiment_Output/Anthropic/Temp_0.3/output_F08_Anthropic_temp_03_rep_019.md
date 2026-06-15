 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance bottleneck is in `build_neighbor_lookup`. This function constructs a lookup table that maps each **cell-year row** to the row indices of its neighbors **in that same year**. Because the rook-neighbor topology is purely spatial and never changes across years, the function is redundantly recomputing the same spatial relationships 28 times (once per year) for each of the 344,208 cells. The result is a list of ~6.46 million entries, each built via string-pasting and hash-table lookups. This is then repeated for every call pattern, and the `compute_neighbor_stats` function iterates over that same 6.46-million-entry list for each of the 5 variables.

**Specific costs:**

1. **`build_neighbor_lookup`**: Creates ~6.46M string keys (`paste(id, year)`), builds a named lookup vector of 6.46M entries, then for each of the 6.46M rows, does string-paste and named-vector lookups to find neighbor rows. This is O(n_cells × n_years) with large constant factors from string operations. Estimated: tens of hours alone.

2. **`compute_neighbor_stats`**: Iterates over the 6.46M-entry neighbor lookup list 5 times (once per variable), extracting and summarizing neighbor values. Each `lapply` call over 6.46M entries with R-level anonymous functions is slow.

3. **The static-vs-changing distinction is not exploited at all.** The neighbor *topology* (which cells are neighbors of which) is static. Only the *variable values* change by year. The current code entangles both.

## Optimization Strategy

**Principle:** Separate the static neighbor topology from the year-varying data, then vectorize the computation year-by-year using matrix operations.

1. **Build the neighbor topology once** as a sparse structure — specifically, a sparse adjacency matrix (or a simple integer-index list mapping each cell to its neighbor cells). This is done once for 344,208 cells, not 6.46M cell-years.

2. **For each year**, subset the data to that year's rows (344,208 rows), arrange them in cell-ID order, and compute neighbor max/min/mean using the static topology via **sparse matrix multiplication** (for mean/sum) and vectorized operations (for max/min).

3. **Use a sparse adjacency matrix** from the `Matrix` package. For mean: `A %*% x / A %*% (non-NA indicator)`. For max and min: iterate over the cell-level neighbor list (344K entries, not 6.46M) or use a grouped operation.

4. This reduces the inner-loop work from 6.46M list iterations to 344K, and replaces string-key lookups with integer indexing. Expected speedup: **~50-200×**, bringing runtime from 86+ hours to under 1 hour.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build static neighbor topology ONCE (from the spdep nb object)
# ==============================================================================

build_static_neighbor_structures <- function(id_order, neighbors) {
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors)
  #
  # Returns:
  #   adj_matrix : sparse binary adjacency matrix (n_cells x n_cells)
  #   neighbor_list : list of integer vectors (neighbor indices per cell)
  
  n <- length(id_order)
  
  # Build sparse adjacency matrix
  # Each neighbors[[i]] contains the indices (into id_order) of cell i's neighbors
  from <- rep(seq_len(n), times = lengths(neighbors))
  to   <- unlist(neighbors)
  
  # Remove any 0-length entries (cells with no neighbors produce integer(0))
  valid <- !is.na(to) & to > 0
  from  <- from[valid]
  to    <- to[valid]
  
  adj_matrix <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n, n)
  )
  
  # Also keep the list form for max/min (which can't be done via matrix multiply)
  neighbor_list <- neighbors  # already integer-index vectors into id_order
  
  list(
    adj_matrix    = adj_matrix,
    neighbor_list = neighbor_list,
    id_order      = id_order,
    n_cells       = n
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats for all variables, one year at a time
# ==============================================================================

compute_neighbor_stats_for_year <- function(year_dt, static, var_names) {
  # year_dt    : data.table for one year, with column 'id' and all var_names
  #              MUST be keyed/ordered to match id_order
  # static     : output of build_static_neighbor_structures
  # var_names  : character vector of variable names
  #
  # Returns year_dt with new columns: {var}_neighbor_max, _min, _mean
  
  A    <- static$adj_matrix
  nlist <- static$neighbor_list
  n    <- static$n_cells
  
  # Precompute the number of valid neighbors per cell for each variable
  # (to handle NAs properly)
  
  for (var in var_names) {
    x <- year_dt[[var]]  # length n, aligned to id_order
    
    # --- Neighbor MEAN via sparse matrix multiply ---
    # Replace NA with 0 for summation, track non-NA counts
    not_na <- as.numeric(!is.na(x))
    x_safe <- ifelse(is.na(x), 0, x)
    
    neighbor_sum   <- as.numeric(A %*% x_safe)
    neighbor_count <- as.numeric(A %*% not_na)
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- Neighbor MAX and MIN via vectorized list operation ---
    # This iterates over 344K cells (not 6.46M cell-years)
    neighbor_max <- rep(NA_real_, n)
    neighbor_min <- rep(NA_real_, n)
    
    for (i in seq_len(n)) {
      nb_idx <- nlist[[i]]
      if (length(nb_idx) == 0L) next
      nb_vals <- x[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      neighbor_max[i] <- max(nb_vals)
      neighbor_min[i] <- min(nb_vals)
    }
    
    # Assign to data.table
    set(year_dt, j = paste0(var, "_neighbor_max"),  value = neighbor_max)
    set(year_dt, j = paste0(var, "_neighbor_min"),  value = neighbor_min)
    set(year_dt, j = paste0(var, "_neighbor_mean"), value = neighbor_mean)
  }
  
  year_dt
}

# ==============================================================================
# STEP 2b: Even faster max/min using Rcpp (optional but recommended)
# ==============================================================================
# If the for-loop over 344K cells for max/min is still slow (5 vars × 28 years
# × 344K = ~48M iterations), we can use Rcpp. Here is a pure-R fallback that
# uses vapply for modest speedup, plus an Rcpp version.

compute_max_min_vectorized <- function(x, neighbor_list) {
  # Pure R, but using vapply instead of for-loop
  n <- length(neighbor_list)
  result <- vapply(seq_len(n), function(i) {
    nb_idx <- neighbor_list[[i]]
    if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_))
    nb_vals <- x[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0L) return(c(NA_real_, NA_real_))
    c(max(nb_vals), min(nb_vals))
  }, numeric(2))
  # result is 2 x n matrix
  list(max = result[1, ], min = result[2, ])
}

# Rcpp version (much faster — recommended for production):
# Uncomment and use if Rcpp is available.
#
# Rcpp::sourceCpp(code = '
# #include <Rcpp.h>
# using namespace Rcpp;
#
# // [[Rcpp::export]]
# List neighbor_max_min_cpp(NumericVector x, List neighbor_list) {
#   int n = neighbor_list.size();
#   NumericVector out_max(n, NA_REAL);
#   NumericVector out_min(n, NA_REAL);
#
#   for (int i = 0; i < n; i++) {
#     IntegerVector nb = neighbor_list[i];
#     if (nb.size() == 0) continue;
#     double cur_max = R_NegInf;
#     double cur_min = R_PosInf;
#     int valid = 0;
#     for (int j = 0; j < nb.size(); j++) {
#       double val = x[nb[j] - 1];  // R is 1-indexed
#       if (NumericVector::is_na(val)) continue;
#       if (val > cur_max) cur_max = val;
#       if (val < cur_min) cur_min = val;
#       valid++;
#     }
#     if (valid > 0) {
#       out_max[i] = cur_max;
#       out_min[i] = cur_min;
#     }
#   }
#   return List::create(Named("max") = out_max, Named("min") = out_min);
# }
# ')

# ==============================================================================
# STEP 3: Main pipeline — replaces the original outer loop
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for performance
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  # ---- STATIC: build once ----
  message("Building static neighbor topology...")
  static <- build_static_neighbor_structures(id_order, rook_neighbors_unique)
  
  # Create mapping from cell id to position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add a position column to cell_data for alignment
  cell_data[, .cell_pos := id_to_pos[as.character(id)]]
  
  # Pre-allocate output columns
  for (var in neighbor_source_vars) {
    cell_data[, paste0(var, "_neighbor_max")  := NA_real_]
    cell_data[, paste0(var, "_neighbor_min")  := NA_real_]
    cell_data[, paste0(var, "_neighbor_mean") := NA_real_]
  }
  
  # ---- CHANGING: process year by year ----
  years <- sort(unique(cell_data$year))
  message(sprintf("Processing %d years x %d variables...", length(years), length(neighbor_source_vars)))
  
  A     <- static$adj_matrix
  nlist <- static$neighbor_list
  n     <- static$n_cells
  
  for (yr in years) {
    message(sprintf("  Year %d ...", yr))
    
    # Get row indices for this year
    yr_rows <- which(cell_data$year == yr)
    
    # Build a vector aligned to id_order for each variable
    # cell_data$.cell_pos[yr_rows] gives the position in id_order for each row
    pos <- cell_data$.cell_pos[yr_rows]
    
    for (var in neighbor_source_vars) {
      # Create id_order-aligned vector
      x <- rep(NA_real_, n)
      x[pos] <- cell_data[[var]][yr_rows]
      
      # --- MEAN via sparse matrix multiply ---
      not_na <- as.numeric(!is.na(x))
      x_safe <- ifelse(is.na(x), 0, x)
      
      neighbor_sum   <- as.numeric(A %*% x_safe)
      neighbor_count <- as.numeric(A %*% not_na)
      neighbor_mean  <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
      
      # --- MAX and MIN ---
      # Use the pure-R vapply version (or swap in Rcpp version for speed)
      mm <- compute_max_min_vectorized(x, nlist)
      
      # Write results back to the correct rows
      set(cell_data, i = yr_rows, j = paste0(var, "_neighbor_max"),  value = mm$max[pos])
      set(cell_data, i = yr_rows, j = paste0(var, "_neighbor_min"),  value = mm$min[pos])
      set(cell_data, i = yr_rows, j = paste0(var, "_neighbor_mean"), value = neighbor_mean[pos])
    }
  }
  
  # Clean up helper column
  cell_data[, .cell_pos := NULL]
  
  message("Neighbor feature computation complete.")
  cell_data
}

# ==============================================================================
# STEP 4: Usage — drop-in replacement for the original code
# ==============================================================================

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with prediction using the pre-trained Random Forest:
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Redesigned |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key entries rebuilt per run | 344K-cell sparse matrix + integer list, built once |
| **Mean computation** | R-level `lapply` over 6.46M entries per variable | Sparse matrix multiply (`A %*% x`), vectorized, per year |
| **Max/Min computation** | R-level `lapply` over 6.46M entries per variable | `vapply` over 344K cells per year (or Rcpp for further speedup) |
| **String operations** | ~13M `paste()` calls + named vector lookups | Zero string operations |
| **Total iterations** | ~6.46M × 5 vars × 3 stats = ~97M R-level function calls | Mean: 28 sparse matmuls × 5 vars = 140 matmuls; Max/Min: 28 × 5 × 344K = ~48M integer-indexed lookups |
| **Estimated runtime** | 86+ hours | ~20–60 minutes (pure R); ~5–15 minutes (with Rcpp max/min) |
| **Numerical results** | Original estimand | **Identical** — same max, min, mean over same neighbor sets |
| **Trained RF model** | Not touched | **Preserved** — no retraining; same feature columns produced |