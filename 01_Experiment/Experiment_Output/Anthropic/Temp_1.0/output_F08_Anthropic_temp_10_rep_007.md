 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core bottleneck is `build_neighbor_lookup`, which constructs a **per-row** (cell × year) lookup — producing ~6.46 million list entries — even though the neighbor *topology* is identical across all 28 years. The function pastes string keys, searches named vectors, and iterates row-by-row in pure R, all at the cell-year level.

**Key observations:**

1. **Static topology, changing values.** The rook-neighbor graph is purely spatial. Cell *i*'s neighbors are always the same cells regardless of year. Yet `build_neighbor_lookup` re-discovers and re-indexes these neighbors for every (cell, year) row.

2. **O(n × T) string hashing.** The function creates ~6.46M string keys (`paste(id, year)`), stores them in a named vector, then for every row looks up neighbor keys — totaling billions of character comparisons.

3. **`compute_neighbor_stats` is fine in principle**, but it operates on a 6.46M-element lookup list, which is itself the product of the bloated build step.

4. **The numerical estimand is a simple max/min/mean of neighbor values within the same year.** This can be computed with a single matrix operation per variable if we reshape the data into a cell × year matrix and apply the static neighbor list to columns of that matrix.

## Optimization Strategy

| Aspect | Current | Proposed |
|---|---|---|
| Neighbor lookup granularity | Per cell-year row (~6.46M entries) | Per cell (~344K entries, built once) |
| Value access | Named-vector string lookup | Direct integer-indexed matrix columns |
| Stats computation | R `lapply` over 6.46M rows | Vectorized sparse-matrix multiplication / integer-indexed matrix ops over 344K cells × 28 years |
| Estimated time | 86+ hours | Minutes |

**Approach:**

1. **Build a cell-level neighbor index once** from `rook_neighbors_unique` (a standard `nb` object, already integer-indexed by position in `id_order`). This is just the `nb` object itself — no rebuild needed.

2. **Reshape each variable into a 344,208 × 28 matrix** (rows = cells in `id_order` order, columns = years).

3. **For each variable, compute neighbor max/min/mean** by iterating over the 344K cells (not 6.46M rows), pulling neighbor rows from the matrix, and computing column-wise stats. This is done with compiled R internals (`vapply`, vectorized subsetting).

4. **Melt the result matrices back** and join to the original `cell_data` data frame.

5. **Feed into the pre-trained Random Forest** exactly as before — column names and numerical values are identical.

## Working R Code

```r
library(data.table)

#' Optimized neighbor feature computation.
#' Exploits the fact that the neighbor graph is static across years,
#' while variable values change by year.
#'
#' @param cell_data    data.frame/data.table with columns: id, year, and all source vars
#' @param id_order     integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param neighbors    spdep::nb object (list of integer index vectors into id_order)
#' @param source_vars  character vector of variable names to compute neighbor stats for
#' @return cell_data with new columns: <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
compute_all_neighbor_features <- function(cell_data, id_order, neighbors, source_vars) {

  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)

  # --- 1. Build a mapping from cell id to position in id_order (once) ---
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # --- 2. Determine the year vector (sorted) ---
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))

  # --- 3. Map each row of dt to (cell_position, year_column) ---
  dt[, cell_pos := id_to_pos[as.character(id)]]
  dt[, year_col := year_to_col[as.character(year)]]

  # --- 4. For each source variable, build matrix, compute stats, merge back ---
  for (var_name in source_vars) {

    message("Processing neighbor stats for: ", var_name)

    # 4a. Build cell × year matrix (NA where data is missing)
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(dt$cell_pos, dt$year_col)] <- dt[[var_name]]

    # 4b. Compute neighbor stats: result matrices (n_cells × n_years)
    max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (i in seq_len(n_cells)) {
      nb_idx <- neighbors[[i]]
      if (length(nb_idx) == 0L) next
      # nb_idx are integer positions into id_order — which are row indices of val_mat
      # Extract neighbor sub-matrix: length(nb_idx) rows × n_years cols
      nb_vals <- val_mat[nb_idx, , drop = FALSE]
      # Column-wise stats (each column = one year)
      # Using colMeans / apply is vectorized across years
      max_mat[i, ]  <- apply(nb_vals, 2L, max, na.rm = TRUE)
      min_mat[i, ]  <- apply(nb_vals, 2L, min, na.rm = TRUE)
      mean_mat[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }

    # Replace -Inf/Inf from max/min of all-NA slices with NA
    max_mat[is.infinite(max_mat)] <- NA_real_
    min_mat[is.infinite(min_mat)] <- NA_real_
    mean_mat[is.nan(mean_mat)]    <- NA_real_

    # 4c. Look up results for each row of dt using (cell_pos, year_col) index
    idx_mat <- cbind(dt$cell_pos, dt$year_col)

    dt[, (paste0(var_name, "_neighbor_max"))  := max_mat[idx_mat]]
    dt[, (paste0(var_name, "_neighbor_min"))  := min_mat[idx_mat]]
    dt[, (paste0(var_name, "_neighbor_mean")) := mean_mat[idx_mat]]
  }

  # --- 5. Clean up helper columns ---
  dt[, c("cell_pos", "year_col") := NULL]

  return(as.data.frame(dt))
}
```

### Further acceleration: replace the inner `for` loop with chunked C-level vectorization

The inner loop (344K iterations of `apply` on small sub-matrices) is the remaining hotspot. We can eliminate `apply` overhead:

```r
# Drop-in replacement for the inner loop in step 4b, using vapply + direct indexing.
# This version pre-allocates neighbor row indices as an integer list (already done
# via the nb object) and uses vectorized column operations.

compute_neighbor_stat_matrices <- function(val_mat, neighbors, n_cells, n_years) {

  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb == 0L) next

    if (n_nb == 1L) {
      # Single neighbor: no aggregation needed — just copy the row
      row_vals <- val_mat[nb_idx, ]
      max_mat[i, ]  <- row_vals
      min_mat[i, ]  <- row_vals
      mean_mat[i, ] <- row_vals
    } else {
      nb_vals <- val_mat[nb_idx, , drop = FALSE]
      # .colSums, .colMeans are internal C-level functions — faster than apply()
      # For max/min we still need apply, but the sub-matrices are tiny (2-4 rows)
      max_mat[i, ]  <- apply(nb_vals, 2L, max, na.rm = TRUE)
      min_mat[i, ]  <- apply(nb_vals, 2L, min, na.rm = TRUE)
      mean_mat[i, ] <- .colMeans(nb_vals, m = n_nb, n = n_years, na.rm = TRUE)
    }
  }

  max_mat[is.infinite(max_mat)] <- NA_real_
  min_mat[is.infinite(min_mat)] <- NA_real_
  mean_mat[is.nan(mean_mat)]    <- NA_real_

  list(max = max_mat, min = min_mat, mean = mean_mat)
}
```

### Optional: Rcpp version for maximum speed

If the pure-R inner loop is still too slow (~10–20 min), the following Rcpp function reduces it to seconds:

```r
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_stats_cpp(NumericMatrix val_mat, List neighbors) {
  int n_cells = val_mat.nrow();
  int n_years = val_mat.ncol();

  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  NumericMatrix mean_mat(n_cells, n_years);

  // Initialize to NA
  std::fill(max_mat.begin(),  max_mat.end(),  NA_REAL);
  std::fill(min_mat.begin(),  min_mat.end(),  NA_REAL);
  std::fill(mean_mat.begin(), mean_mat.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    IntegerVector nb = as<IntegerVector>(neighbors[i]);
    int n_nb = nb.size();
    if (n_nb == 0) continue;

    for (int t = 0; t < n_years; t++) {
      double mx = R_NegInf, mn = R_PosInf, sm = 0.0;
      int cnt = 0;
      for (int k = 0; k < n_nb; k++) {
        double v = val_mat(nb[k] - 1, t);   // R is 1-indexed
        if (NumericMatrix::is_na(v)) continue;
        if (v > mx) mx = v;
        if (v < mn) mn = v;
        sm += v;
        cnt++;
      }
      if (cnt > 0) {
        max_mat(i, t)  = mx;
        min_mat(i, t)  = mn;
        mean_mat(i, t) = sm / cnt;
      }
    }
  }

  return List::create(Named("max") = max_mat,
                      Named("min") = min_mat,
                      Named("mean") = mean_mat);
}
')
```

### Full orchestration (putting it all together)

```r
# ---- Load pre-trained model and data ----
# rf_model       <- readRDS("rf_model.rds")
# cell_data      <- readRDS("cell_data.rds")           # ~6.46M rows
# id_order       <- readRDS("id_order.rds")             # 344,208 cell IDs
# rook_neighbors_unique <- readRDS("rook_neighbors.rds") # spdep::nb object

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# ---- Compute neighbor features (optimized) ----
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ---- Predict with the pre-trained Random Forest (unchanged) ----
# Ensure predictor columns match the model's expected features
pred_vars <- names(rf_model$forest$xlevels)  # or however your model stores feature names
# If using ranger:
# pred_vars <- rf_model$forest$independent.variable.names

predictions <- predict(rf_model, newdata = cell_data[, pred_vars])
cell_data$predicted <- predictions$predictions   # ranger
# cell_data$predicted <- predictions              # randomForest
```

## Summary

| Item | Detail |
|---|---|
| **Root cause** | Neighbor lookup built at cell×year granularity (6.46M entries) despite static topology |
| **Fix** | Separate static topology (344K cells) from dynamic values (cell × year matrix); compute stats via matrix indexing |
| **Speedup source** | 18.75× fewer iterations (344K vs 6.46M); eliminates all string key construction and lookup; matrix subsetting uses compiled C internals |
| **Expected runtime** | Pure R: ~10–20 minutes; Rcpp: ~1–3 minutes (vs. 86+ hours) |
| **Numerical equivalence** | Identical — same max/min/mean of same neighbor values per cell-year |
| **RF model** | Untouched — used only for prediction after features are computed |