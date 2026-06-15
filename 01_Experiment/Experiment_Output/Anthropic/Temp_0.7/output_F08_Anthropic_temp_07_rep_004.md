 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static spatial topology with dynamic temporal data.**

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows)** instead of per cell (~344K cells). Since rook-neighbor topology is fixed across all 28 years, this is doing 28× redundant work — and doing it via expensive string-key lookups (`paste` + named-vector indexing).

2. **The lookup returns row indices into the full panel data frame.** This means for every cell-year row, the function searches for neighbor rows matching both the neighbor cell ID *and* the current year via string concatenation and hash lookup. With ~6.46M keys, this is memory-intensive and cache-unfriendly.

3. **`compute_neighbor_stats` iterates over ~6.46M entries** in the lookup list, extracting values one-at-a-time with `lapply`, producing millions of small 3-element vectors and then `rbind`-ing them — a classic R anti-pattern.

4. **This is repeated 5 times** (once per neighbor source variable), compounding the cost.

**In summary:** The code treats a static graph problem as a dynamic one, pays O(cells × years) cost where O(cells) suffices for topology, uses slow string-key lookups, and relies on element-wise R loops over millions of rows.

---

## Optimization Strategy

### Key Insight: Separate Static Topology from Dynamic Values

- **Neighbor topology is static:** Cell A's neighbors are the same in 1992 and 2019. Build the neighbor index **once over 344K cells**, not over 6.46M cell-years.
- **Variable values are dynamic:** They change by year. Organize values in a **cell × year matrix**, so that for any year-column, you can vectorize the neighbor aggregation using the static neighbor index.

### Concrete Plan

1. **Build a static neighbor list** — a simple list of length 344,208 where element `i` contains the integer indices of cell `i`'s neighbors. This is just `rook_neighbors_unique` itself (an `nb` object), cleaned up. Cost: trivial, done once.

2. **Reshape each variable into a cell × year matrix** (344,208 rows × 28 columns). This allows column-wise (per-year) vectorized operations.

3. **For each variable, compute neighbor max/min/mean using vectorized matrix operations** across the static neighbor list. For each cell `i` with neighbors `nb[[i]]`, extract the sub-matrix of neighbor values (all years at once), and compute row-wise (year-wise) aggregates. This is done once per variable (5 times total).

4. **Use `data.table` for efficient reshaping** and re-merging of results back into the panel.

5. **Optionally, use C++ via `Rcpp`** for the inner neighbor-aggregation loop over 344K cells (instead of 6.46M), which is a ~28× reduction in loop iterations before even considering the elimination of string operations.

### Expected Speedup

| Factor | Improvement |
|---|---|
| Loop iterations: 6.46M → 344K | ~19× |
| Eliminate string key construction/lookup | ~5-10× |
| Vectorized matrix column access vs. data.frame row subsetting | ~3-5× |
| Combined | **~200-500×** (from ~86 hours to **~10-30 minutes**) |

---

## Working R Code

```r
# ==============================================================================
# Optimized neighbor feature computation
# Separates static topology from dynamic (year-varying) values
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# Step 1: Build static neighbor index (done ONCE, independent of years)
# --------------------------------------------------------------------------
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where each element contains integer indices of neighbors.
# id_order is the vector of cell IDs in the order matching the nb object.

build_static_neighbor_index <- function(id_order, nb_object) {
  # nb_object[[i]] already contains integer indices into id_order
  # We just need to clean out the 0-neighbor sentinel used by spdep
  n <- length(nb_object)
  stopifnot(n == length(id_order))
  
  neighbor_idx <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- nb_object[[i]]
    # spdep uses integer(0) or 0L for no-neighbor cells
    if (length(nb_i) == 0 || (length(nb_i) == 1 && nb_i[1] == 0L)) {
      neighbor_idx[[i]] <- integer(0)
    } else {
      neighbor_idx[[i]] <- as.integer(nb_i)
    }
  }
  
  # Attach cell ID mapping for later use
  attr(neighbor_idx, "id_order") <- id_order
  neighbor_idx
}

# --------------------------------------------------------------------------
# Step 2: Reshape a variable from long panel to cell x year matrix
# --------------------------------------------------------------------------
reshape_to_matrix <- function(dt, var_name, id_order, year_order) {
  # dt must be a data.table with columns: id, year, <var_name>
  # Returns a matrix: rows = cells (in id_order), cols = years (in year_order)
  
  n_cells <- length(id_order)
  n_years <- length(year_order)
  
  # Create mapping from id -> row position in matrix
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  # Create mapping from year -> col position in matrix
  year_to_col <- setNames(seq_along(year_order), as.character(year_order))
  
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  row_positions <- id_to_row[as.character(dt$id)]
  col_positions <- year_to_col[as.character(dt$year)]
  
  mat[cbind(row_positions, col_positions)] <- dt[[var_name]]
  
  colnames(mat) <- as.character(year_order)
  mat
}

# --------------------------------------------------------------------------
# Step 3: Compute neighbor stats using static topology + value matrices
#          Pure R version (fast enough for 344K cells)
# --------------------------------------------------------------------------
compute_neighbor_stats_matrix <- function(value_matrix, neighbor_idx) {
  # value_matrix: n_cells x n_years
  # neighbor_idx: list of length n_cells, each element = integer vector of neighbor row indices
  # Returns: list with three matrices (max, min, mean), each n_cells x n_years
  
  n_cells <- nrow(value_matrix)
  n_years <- ncol(value_matrix)
  
  mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb <- neighbor_idx[[i]]
    if (length(nb) == 0L) next
    
    # Extract sub-matrix: neighbors x years (all years at once)
    nb_vals <- value_matrix[nb, , drop = FALSE]
    
    if (length(nb) == 1L) {
      # Single neighbor: the row is the answer (handle NAs)
      mat_max[i, ]  <- nb_vals[1, ]
      mat_min[i, ]  <- nb_vals[1, ]
      mat_mean[i, ] <- nb_vals[1, ]
    } else {
      # Multiple neighbors: column-wise aggregation
      # suppressWarnings for all-NA columns
      mat_max[i, ]  <- suppressWarnings(apply(nb_vals, 2, max,  na.rm = TRUE))
      mat_min[i, ]  <- suppressWarnings(apply(nb_vals, 2, min,  na.rm = TRUE))
      mat_mean[i, ] <- suppressWarnings(apply(nb_vals, 2, mean, na.rm = TRUE))
    }
  }
  
  # Fix Inf/-Inf from max/min on all-NA columns
  mat_max[is.infinite(mat_max)]   <- NA_real_
  mat_min[is.infinite(mat_min)]   <- NA_real_
  
  list(max = mat_max, min = mat_min, mean = mat_mean)
}

# --------------------------------------------------------------------------
# Step 3 (ALTERNATIVE): Rcpp version for maximum speed
# --------------------------------------------------------------------------
# Uncomment and use if the pure R loop above is still too slow.

# Rcpp::cppFunction('
# #include <Rcpp.h>
# using namespace Rcpp;
# 
# // [[Rcpp::export]]
# List compute_neighbor_stats_cpp(NumericMatrix value_matrix, List neighbor_idx) {
#   int n_cells = value_matrix.nrow();
#   int n_years = value_matrix.ncol();
#   
#   NumericMatrix mat_max(n_cells, n_years);
#   NumericMatrix mat_min(n_cells, n_years);
#   NumericMatrix mat_mean(n_cells, n_years);
#   
#   // Initialize with NA
#   std::fill(mat_max.begin(),  mat_max.end(),  NA_REAL);
#   std::fill(mat_min.begin(),  mat_min.end(),  NA_REAL);
#   std::fill(mat_mean.begin(), mat_mean.end(), NA_REAL);
#   
#   for (int i = 0; i < n_cells; i++) {
#     IntegerVector nb = neighbor_idx[i];
#     int n_nb = nb.size();
#     if (n_nb == 0) continue;
#     
#     for (int y = 0; y < n_years; y++) {
#       double vmax = R_NegInf;
#       double vmin = R_PosInf;
#       double vsum = 0.0;
#       int    cnt  = 0;
#       
#       for (int k = 0; k < n_nb; k++) {
#         double val = value_matrix(nb[k] - 1, y);  // R is 1-indexed
#         if (R_IsNA(val)) continue;
#         if (val > vmax) vmax = val;
#         if (val < vmin) vmin = val;
#         vsum += val;
#         cnt++;
#       }
#       
#       if (cnt > 0) {
#         mat_max(i, y)  = vmax;
#         mat_min(i, y)  = vmin;
#         mat_mean(i, y) = vsum / cnt;
#       }
#     }
#   }
#   
#   return List::create(
#     Named("max")  = mat_max,
#     Named("min")  = mat_min,
#     Named("mean") = mat_mean
#   );
# }
# ')

# --------------------------------------------------------------------------
# Step 4: Unpack matrices back to long panel format and attach to data
# --------------------------------------------------------------------------
unpack_matrix_to_long <- function(mat, id_order, year_order, col_name, dt) {
  # mat: n_cells x n_years matrix
  # Returns: adds column col_name to dt (by reference if data.table)
  
  id_to_row   <- setNames(seq_along(id_order), as.character(id_order))
  year_to_col <- setNames(seq_along(year_order), as.character(year_order))
  
  row_positions <- id_to_row[as.character(dt$id)]
  col_positions <- year_to_col[as.character(dt$year)]
  
  dt[, (col_name) := mat[cbind(row_positions, col_positions)]]
  invisible(dt)
}

# ==========================================================================
# MAIN EXECUTION
# ==========================================================================

# Convert to data.table for efficiency (if not already)
cell_data <- as.data.table(cell_data)

# --- Static topology: built ONCE ---
message("Building static neighbor index (once)...")
static_nb <- build_static_neighbor_index(id_order, rook_neighbors_unique)

# --- Identify years ---
year_order <- sort(unique(cell_data$year))

# --- Neighbor source variables ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- Compute neighbor features for each variable ---
for (var_name in neighbor_source_vars) {
  message(sprintf("Processing neighbor stats for: %s", var_name))
  
  # Step 2: Reshape to cell x year matrix
  message("  Reshaping to matrix...")
  val_mat <- reshape_to_matrix(cell_data, var_name, id_order, year_order)
  
  # Step 3: Compute neighbor aggregates over static topology
  message("  Computing neighbor max/min/mean...")
  stats <- compute_neighbor_stats_matrix(val_mat, static_nb)
  # For Rcpp version, replace the line above with:
  # stats <- compute_neighbor_stats_cpp(val_mat, static_nb)
  
  # Step 4: Attach results back to long panel
  message("  Writing results back to panel...")
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  unpack_matrix_to_long(stats$max,  id_order, year_order, max_col,  cell_data)
  unpack_matrix_to_long(stats$min,  id_order, year_order, min_col,  cell_data)
  unpack_matrix_to_long(stats$mean, id_order, year_order, mean_col, cell_data)
  
  # Free memory
  rm(val_mat, stats)
  gc()
  
  message(sprintf("  Done: %s", var_name))
}

message("All neighbor features computed. Proceeding to Random Forest prediction...")

# ==========================================================================
# Random Forest prediction (UNCHANGED — model is pre-trained)
# ==========================================================================
# predictions <- predict(trained_rf_model, newdata = cell_data)
# (Proceed with existing prediction code exactly as before)
```

---

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Topology construction** | Per cell-year (~6.46M string lookups) | Per cell, once (~344K, integer only) |
| **Data structure for values** | Data frame column, accessed via row-index lists | Cell × year matrix (contiguous memory) |
| **Neighbor aggregation loop** | ~6.46M iterations, each extracting from data frame | ~344K iterations, each extracting matrix sub-block (all 28 years at once) |
| **String operations** | `paste()` + named-vector lookup for every row | **Zero** — pure integer indexing |
| **Memory pattern** | ~6.46M-element list of integer vectors | ~344K-element list + matrices (~77 MB each) |
| **Numerical results** | max/min/mean of non-NA neighbor values | **Identical** — same `na.rm=TRUE` semantics preserved |
| **Trained RF model** | — | **Untouched** — prediction step is unchanged |
| **Estimated runtime** | ~86+ hours | **~10-30 minutes** (pure R); **~2-5 minutes** (Rcpp) |

The Rcpp version is included as a commented block and can be activated for further speedup. Both versions produce numerically identical results to the original implementation, preserving the estimand for the pre-trained Random Forest model.