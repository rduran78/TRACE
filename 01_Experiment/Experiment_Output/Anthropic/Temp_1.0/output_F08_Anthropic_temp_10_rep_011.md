 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The root cause of the 86+ hour runtime is that `build_neighbor_lookup` constructs a **row-level** lookup across the entire 6.46-million-row panel dataset. For every one of those ~6.46M rows, it:

1. Looks up the cell's neighbors (a spatial/topological operation — **static across years**).
2. Then maps each neighbor cell ID × the current row's year to a **row index** in the panel (a year-specific operation).

This produces a list of ~6.46M elements, each containing integer row indices. The construction itself is O(N_rows × avg_neighbors), dominated by millions of string-paste-and-match operations. Then `compute_neighbor_stats` iterates over this massive list again for each of the 5 variables.

**Key insight:** The neighbor graph (which cell is adjacent to which) is **purely spatial and static** — it never changes across the 28 years. Only the **variable values** change by year. Therefore:

- The spatial topology should be encoded **once** as a cell-to-cell lookup (344K entries), not a row-to-row lookup (6.46M entries).
- The variable values should be sliced **by year**, then the static cell-level neighbor indices applied within each year slice.

This reduces the lookup construction from ~6.46M entries to ~344K entries, and replaces millions of string-key lookups with fast integer indexing within year-specific matrices/vectors.

---

## Optimization Strategy

1. **Build a static cell-level neighbor index once** — a list of length 344,208 where element `i` contains the integer positions of cell `i`'s rook neighbors within the canonical `id_order` vector. This is year-independent and built once.

2. **Reshape each variable into a cell × year matrix** — rows = cells (in `id_order` order), columns = years. This allows direct integer indexing.

3. **For each year-column, vectorize the neighbor aggregation** using the static cell-level neighbor list — compute max, min, mean of the neighbor values. This is done per variable, per year, but the neighbor list is reused across all variables and years.

4. **Write results back** to the original `cell_data` data.frame in the correct row order.

**Complexity reduction:**
- Lookup construction: 6.46M → 344K (18.8× fewer entries, no string operations).
- Stat computation: the inner loop is 344K cells × 28 years × 5 vars, all using integer-indexed numeric vectors — trivially fast.

**Estimated speedup:** From 86+ hours to **minutes**.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from year-varying values
# =============================================================================

# --- Step 0: Ensure consistent ordering ------------------------------------
# id_order:              character or numeric vector of all 344,208 cell IDs
#                        (same order used when building rook_neighbors_unique)
# rook_neighbors_unique: spdep nb object (list of length 344,208)
# cell_data:             data.frame with columns: id, year, ntl, ec,
#                        pop_density, def, usd_est_n2, ... (~6.46M rows)

# --- Step 1: Build STATIC cell-level neighbor index (done ONCE) -------------
# Each element i contains the integer positions (within id_order) of cell i's
# rook neighbors.  This is the spatial topology — year-independent.

build_cell_neighbor_index <- function(id_order, nb_object) {
  # nb_object[[i]] already contains integer indices into id_order

  # (spdep convention), but may contain 0L for cells with no neighbors.
  n <- length(id_order)
  stopifnot(length(nb_object) == n)
  
  lapply(seq_len(n), function(i) {
    nbrs <- nb_object[[i]]
    # spdep uses 0L to denote "no neighbors"
    nbrs <- nbrs[nbrs > 0L]
    as.integer(nbrs)
  })
}

cell_neighbor_idx <- build_cell_neighbor_index(id_order, rook_neighbors_unique)


# --- Step 2: Map cell IDs to canonical integer positions --------------------
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))


# --- Step 3: Build a row-order mapping so we can read/write cell_data -------
# For each cell (by canonical position) and each year, record the row number
# in cell_data.  We store this as a cell × year matrix of row indices.

years       <- sort(unique(cell_data$year))
n_cells     <- length(id_order)
n_years     <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))

# Pre-allocate matrix: rows = cells (canonical order), cols = years
row_index_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)

# Fill in: for every row in cell_data, place its row number
cell_positions <- id_to_pos[as.character(cell_data$id)]
year_positions <- year_to_col[as.character(cell_data$year)]
row_index_mat[cbind(cell_positions, year_positions)] <- seq_len(nrow(cell_data))


# --- Step 4: Compute neighbor stats for all variables -----------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  
  message("Processing neighbor stats for: ", var_name)
  
  # 4a. Reshape variable into cell × year matrix
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  val_mat[cbind(cell_positions, year_positions)] <- cell_data[[var_name]]
  
  # 4b. Pre-allocate output matrices (cell × year)
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # 4c. For each year, compute neighbor aggregates using STATIC neighbor index
  for (j in seq_len(n_years)) {
    year_vals <- val_mat[, j]  # length = n_cells
    
    for (i in seq_len(n_cells)) {
      nbrs <- cell_neighbor_idx[[i]]
      if (length(nbrs) == 0L) next
      
      nv <- year_vals[nbrs]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) next
      
      max_mat[i, j]  <- max(nv)
      min_mat[i, j]  <- min(nv)
      mean_mat[i, j] <- mean(nv)
    }
  }
  
  # 4d. Write results back to cell_data in original row order
  #     using the row_index_mat to map (cell, year) -> row in cell_data
  valid <- !is.na(row_index_mat)
  rows_out        <- row_index_mat[valid]
  cell_idx_valid  <- row(row_index_mat)[valid]
  year_idx_valid  <- col(row_index_mat)[valid]
  linear_idx      <- (year_idx_valid - 1L) * n_cells + cell_idx_valid
  
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  
  cell_data[[col_max]][rows_out]  <- max_mat[linear_idx]
  cell_data[[col_min]][rows_out]  <- min_mat[linear_idx]
  cell_data[[col_mean]][rows_out] <- mean_mat[linear_idx]
}

message("Neighbor feature computation complete.")
```

### Optional: Further Speedup with `vapply` Instead of Inner Loop

If the inner `for (i in ...)` loop over 344K cells is still slow (though it should complete in seconds per year), replace it with a vectorized C-level call:

```r
# Drop-in replacement for the inner cell loop within a single year.
# Uses vapply for modest speedup; for maximum speed, use Rcpp (shown below).

compute_year_stats <- function(year_vals, cell_neighbor_idx, n_cells) {
  stats <- vapply(seq_len(n_cells), function(i) {
    nbrs <- cell_neighbor_idx[[i]]
    if (length(nbrs) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    nv <- year_vals[nbrs]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  }, numeric(3))
  # stats is 3 × n_cells; transpose to n_cells × 3
  t(stats)
}
```

### Optional: Maximum Performance with Rcpp

```r
Rcpp::cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_stats_cpp(NumericVector vals, List nb_idx) {
  int n = nb_idx.size();
  NumericMatrix out(n, 3);  // columns: max, min, mean
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n; i++) {
    IntegerVector nbrs = nb_idx[i];
    int m = nbrs.size();
    if (m == 0) continue;

    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;
    int count = 0;

    for (int j = 0; j < m; j++) {
      double v = vals[nbrs[j] - 1];  // R is 1-indexed
      if (ISNA(v) || ISNAN(v)) continue;
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
      vsum += v;
      count++;
    }

    if (count > 0) {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / count;
    }
  }
  return out;
}
')

# Then inside the year loop, replace the inner cell loop with:
# stats <- neighbor_stats_cpp(year_vals, cell_neighbor_idx)
# max_mat[, j]  <- stats[, 1]
# min_mat[, j]  <- stats[, 2]
# mean_mat[, j] <- stats[, 3]
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup size | ~6.46M entries (row-level) | ~344K entries (cell-level, static) |
| Lookup construction | String paste + match per row | One-time integer list from `nb` object |
| Stat computation | Iterates 6.46M × 5 vars | Iterates 344K cells × 28 years × 5 vars (same total, but integer-indexed, no string ops) |
| Key overhead eliminated | Millions of `paste()` + named-vector lookups | Pure integer indexing into numeric vectors |
| Random Forest model | Untouched | Untouched |
| Numerical results | Identical `max`, `min`, `mean` | Identical — same arithmetic, same neighbor sets |
| Estimated runtime | 86+ hours | **5–15 minutes** (pure R); **1–3 minutes** (Rcpp) |