 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` — O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's reference index.
2. Retrieves its rook neighbors (spatial, ~4 per cell).
3. Constructs character keys by pasting `id_year` strings.
4. Matches those keys against a named lookup vector.

This means **~6.46 million iterations**, each doing character concatenation and named-vector lookups. Named vector lookups in R are hash-based but still carry significant per-call overhead at this scale. The resulting `neighbor_lookup` is a list of 6.46M integer vectors — large in memory and slow to build.

**Key insight:** The neighbor structure is purely spatial (rook neighbors don't change across years). The same spatial neighbor graph applies identically to every year. There is no need to build a 6.46M-element lookup. You only need a **344,208-element spatial lookup** and then index into year-slices of the data.

### Bottleneck B: `compute_neighbor_stats` — O(n) `lapply` over 6.46M rows

For each of the 5 variables, another `lapply` iterates over all 6.46M rows, subsetting a numeric vector by the neighbor indices, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times = ~32.3 million R-level function calls with per-element subsetting.

**Key insight:** Since the neighbor structure is spatial and identical across years, the computation can be **vectorized by year** using matrix operations. For each year, arrange the variable values into a spatial vector indexed by cell, build a sparse neighbor matrix once, and compute the stats via sparse matrix–vector multiplication (for mean) and analogous operations (for max/min).

### Why raster focal/kernel operations are a poor fit here

The cells are on an irregular grid (spdep::nb object, not necessarily a regular raster). Even if they were regular, the requirement is to preserve the exact numerical estimand from the pre-trained Random Forest, so we must use the exact same rook-neighbor relationships. Raster focal operations assume a regular grid and a fixed kernel, which may not match. **We use sparse matrix operations instead**, which are the correct generalization.

### Memory estimate

- Sparse neighbor matrix: 344,208 × 344,208 with ~1.37M nonzero entries ≈ trivial (~30 MB).
- Data at 6.46M × 110 columns ≈ ~5.7 GB as numeric. Tight on 16 GB but feasible if we avoid unnecessary copies.

---

## 2. Optimization Strategy

| Step | Current | Optimized |
|------|---------|-----------|
| Neighbor lookup | 6.46M-element list of integer vectors built via character key matching | 344K-element spatial-only list + sparse matrix `W` built once |
| Stat computation | `lapply` over 6.46M rows per variable | Year-grouped vectorized sparse-matrix operations |
| Total iterations | 6.46M × 5 = 32.3M R function calls | 28 years × 5 vars = 140 vectorized operations |
| Expected time | 86+ hours | **Minutes** (dominated by sparse matrix ops) |

**Approach:**
1. Build a sparse binary adjacency matrix `W` (344,208 × 344,208) from `rook_neighbors_unique` — done once.
2. Sort/index `cell_data` so that for each year we can extract a numeric vector aligned to the spatial cell order.
3. For each variable and each year, use the sparse matrix to compute neighbor max, min, mean in vectorized fashion.
4. Write results back into `cell_data`.

For **mean**: `W_row_normalized %*% x` gives the neighbor mean directly.
For **max** and **min**: We iterate over the (spatial-only) neighbor list — but only 344K cells, not 6.46M rows. This is ~19× fewer iterations and each is done once per year.

---

## 3. Working R Code

```r
library(Matrix)
library(data.table)

# ============================================================
# STEP 0: Convert cell_data to data.table for performance
# ============================================================
# Assumes cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assumes rook_neighbors_unique is an spdep::nb object (list of integer vectors)
# Assumes id_order is the vector of cell IDs corresponding to indices in rook_neighbors_unique

cell_dt <- as.data.table(cell_data)

# ============================================================
# STEP 1: Build sparse adjacency matrix W (once)
# ============================================================
build_sparse_neighbor_matrix <- function(neighbors, n) {
  # neighbors: spdep::nb object — list of length n, each element is integer
  #            vector of neighbor indices (or 0L for no neighbors)
  from <- rep(seq_len(n), times = vapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  to <- unlist(lapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else x
  }), use.names = FALSE)
  
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(id_order)
W <- build_sparse_neighbor_matrix(rook_neighbors_unique, n_cells)

# Row-normalized version for computing means
row_sums <- rowSums(W)
row_sums[row_sums == 0] <- NA  # will produce NA for isolated cells
W_norm <- W / row_sums  # each row sums to 1 (or NA row for isolated cells)

# ============================================================
# STEP 2: Create a mapping from cell id to spatial index
# ============================================================
id_to_spatial_idx <- setNames(seq_along(id_order), as.character(id_order))

# Add spatial index to data.table
cell_dt[, spatial_idx := id_to_spatial_idx[as.character(id)]]

# Sort for efficient year-grouped operations
setkey(cell_dt, year, spatial_idx)

# ============================================================
# STEP 3: Precompute neighbor list in simple form for max/min
#          (only 344K elements, not 6.46M)
# ============================================================
nb_list <- lapply(seq_len(n_cells), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 1L && nb[1] == 0L) integer(0) else nb
})

# ============================================================
# STEP 4: Vectorized neighbor stat computation
# ============================================================
compute_neighbor_features_fast <- function(dt, var_name, W, W_norm, nb_list,
                                           n_cells, id_order) {
  max_col <- paste0("n_max_", var_name)
  min_col <- paste0("n_min_", var_name)
  mean_col <- paste0("n_mean_", var_name)
  
  # Pre-allocate output columns
  dt[, (max_col) := NA_real_]
  dt[, (min_col) := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Extract rows for this year (already keyed by year, spatial_idx)
    yr_rows <- which(dt$year == yr)
    yr_sub <- dt[yr_rows]
    
    # Build a full-length spatial vector (NA for missing cells)
    x_full <- rep(NA_real_, n_cells)
    x_full[yr_sub$spatial_idx] <- yr_sub[[var_name]]
    
    # --- MEAN via sparse matrix multiplication ---
    # W_norm %*% x_full: for each cell, average of neighbor values
    # Cells with all-NA neighbors will get NaN or NA naturally
    n_mean_vec <- as.numeric(W_norm %*% x_full)
    # Fix: if a cell has no neighbors (row_sums==0), result is already NA
    # If all neighbor values are NA, the dot product gives NA — correct
    
    # --- MAX and MIN via vectorized neighbor list ---
    # Only 344K iterations, each very fast
    n_max_vec <- rep(NA_real_, n_cells)
    n_min_vec <- rep(NA_real_, n_cells)
    
    for (i in seq_len(n_cells)) {
      nb <- nb_list[[i]]
      if (length(nb) == 0L) next
      vals <- x_full[nb]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) next
      n_max_vec[i] <- max(vals)
      n_min_vec[i] <- min(vals)
    }
    
    # Handle mean: sparse matmul with NAs needs correction.
    # W_norm %*% x_full doesn't correctly ignore NAs (it propagates them).
    # We need: for each cell, mean of non-NA neighbor values.
    # Correct approach: sum of non-NA values / count of non-NA values
    
    x_notna <- as.numeric(!is.na(x_full))
    x_zero <- x_full
    x_zero[is.na(x_zero)] <- 0
    
    neighbor_sum   <- as.numeric(W %*% x_zero)
    neighbor_count <- as.numeric(W %*% x_notna)
    
    n_mean_vec <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # Write back to data.table using spatial_idx alignment
    set(dt, i = yr_rows, j = max_col,  value = n_max_vec[yr_sub$spatial_idx])
    set(dt, i = yr_rows, j = min_col,  value = n_min_vec[yr_sub$spatial_idx])
    set(dt, i = yr_rows, j = mean_col, value = n_mean_vec[yr_sub$spatial_idx])
  }
  
  dt
}

# ============================================================
# STEP 5: Eliminate the R-level loop for max/min using Rcpp
#          (optional but recommended — drops from ~minutes to seconds)
# ============================================================
# If Rcpp is available, replace the inner max/min loop:

if (requireNamespace("Rcpp", quietly = TRUE)) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_max_min_cpp(NumericVector x, List nb_list) {
  int n = nb_list.size();
  NumericVector out_max(n, NA_REAL);
  NumericVector out_min(n, NA_REAL);
  
  for (int i = 0; i < n; i++) {
    IntegerVector nb = nb_list[i];
    if (nb.size() == 0) continue;
    
    double cur_max = R_NegInf;
    double cur_min = R_PosInf;
    int valid = 0;
    
    for (int j = 0; j < nb.size(); j++) {
      double val = x[nb[j] - 1];  // R is 1-indexed
      if (!NumericVector::is_na(val)) {
        if (val > cur_max) cur_max = val;
        if (val < cur_min) cur_min = val;
        valid++;
      }
    }
    
    if (valid > 0) {
      out_max[i] = cur_max;
      out_min[i] = cur_min;
    }
  }
  
  return List::create(Named("max") = out_max, Named("min") = out_min);
}
')
  USE_RCPP <- TRUE
} else {
  USE_RCPP <- FALSE
}

# ============================================================
# STEP 6: Final optimized function (with optional Rcpp)
# ============================================================
compute_neighbor_features_optimized <- function(dt, var_name, W, nb_list,
                                                 n_cells, use_rcpp = FALSE) {
  max_col  <- paste0("n_max_", var_name)
  min_col  <- paste0("n_min_", var_name)
  mean_col <- paste0("n_mean_", var_name)
  
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]
  
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    yr_rows <- which(dt$year == yr)
    yr_spatial <- dt$spatial_idx[yr_rows]
    yr_vals   <- dt[[var_name]][yr_rows]
    
    # Build spatial vector
    x_full <- rep(NA_real_, n_cells)
    x_full[yr_spatial] <- yr_vals
    
    # MEAN (NA-safe via sparse ops)
    x_zero  <- x_full;  x_zero[is.na(x_zero)] <- 0
    x_notna <- as.numeric(!is.na(x_full))
    
    neighbor_sum   <- as.numeric(W %*% x_zero)
    neighbor_count <- as.numeric(W %*% x_notna)
    n_mean_vec     <- ifelse(neighbor_count > 0,
                             neighbor_sum / neighbor_count, NA_real_)
    
    # MAX / MIN
    if (use_rcpp) {
      mm <- neighbor_max_min_cpp(x_full, nb_list)
      n_max_vec <- mm$max
      n_min_vec <- mm$min
    } else {
      n_max_vec <- rep(NA_real_, n_cells)
      n_min_vec <- rep(NA_real_, n_cells)
      for (i in seq_len(n_cells)) {
        nb <- nb_list[[i]]
        if (length(nb) == 0L) next
        vals <- x_full[nb]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) next
        n_max_vec[i] <- max(vals)
        n_min_vec[i] <- min(vals)
      }
    }
    
    # Write back
    set(dt, i = yr_rows, j = max_col,  value = n_max_vec[yr_spatial])
    set(dt, i = yr_rows, j = min_col,  value = n_min_vec[yr_spatial])
    set(dt, i = yr_rows, j = mean_col, value = n_mean_vec[yr_spatial])
  }
  
  dt
}

# ============================================================
# STEP 7: Run the outer loop
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare nb_list (handle spdep 0-neighbor convention)
nb_list <- lapply(seq_len(n_cells), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 1L && nb[1] == 0L) integer(0) else as.integer(nb)
})

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_dt <- compute_neighbor_features_optimized(
    cell_dt, var_name, W, nb_list, n_cells, use_rcpp = USE_RCPP
  )
}

# ============================================================
# STEP 8: Remove helper column and convert back if needed
# ============================================================
cell_dt[, spatial_idx := NULL]

# Convert back to data.frame for predict() compatibility
cell_data <- as.data.frame(cell_dt)

# ============================================================
# STEP 9: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary of Improvements

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Neighbor lookup construction | 6.46M character-key lookups | Sparse matrix built once from 344K-cell nb object |
| Mean computation | `lapply` over 6.46M rows × 5 vars | Sparse matrix multiply: 28 years × 5 vars = 140 ops |
| Max/Min computation | `lapply` over 6.46M rows × 5 vars | Loop over 344K cells × 28 years × 5 vars (or Rcpp) |
| Estimated runtime | **86+ hours** | **~5–15 minutes** (pure R) / **~2–5 minutes** (with Rcpp) |
| Numerical results | Baseline | **Identical** — same rook neighbors, same max/min/mean |
| RAM usage | Huge list of 6.46M int vectors | Sparse matrix (~30 MB) + data.table in-place ops |
| RF model | Unchanged | **Unchanged — no retraining** |

The key insight is that the neighbor graph is **spatial, not spatiotemporal** — it is invariant across years. By factoring the computation into (spatial structure) × (temporal slices), we reduce the problem from 6.46M per-row operations to 344K per-cell operations done 28 times, and the mean computation becomes a single sparse matrix–vector product per year. This preserves the exact numerical output required by the pre-trained Random Forest model.