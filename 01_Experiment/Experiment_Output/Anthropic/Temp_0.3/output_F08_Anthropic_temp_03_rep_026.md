 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It creates a list of ~6.46 million entries, each requiring string-pasting cell IDs with years, hash lookups in `idx_lookup`, and NA filtering. But the neighbor *relationships* between cells are purely spatial and **never change across years**. The year dimension is entirely redundant in the topology.

2. **`compute_neighbor_stats` operates on the full 6.46M-row vector.** Because the lookup was built at the cell×year level, every stat computation indexes into a massive vector, and the `lapply` iterates over 6.46M elements instead of 344K elements.

3. **String-based key construction** (`paste(id, year, sep="_")`) and named-vector lookups (`idx_lookup[neighbor_keys]`) are extremely slow at this scale — O(n) string allocations and hash lookups repeated millions of times.

### The Key Insight

- **Static:** The rook-neighbor graph (which cells are neighbors of which) is invariant across all 28 years. There are only ~344K cells and ~1.37M directed neighbor edges.
- **Dynamic:** The variable values (`ntl`, `ec`, `pop_density`, `def`, `usd_est_n2`) change by year.

Therefore, the correct architecture is:

1. **Build the neighbor lookup once over 344K cells** (not 6.46M cell-years).
2. **For each year, slice the data, compute neighbor stats over 344K cells, and write back.**

This reduces the core loop from 6.46M iterations to 344K iterations × 28 years, eliminates all string-key construction, and uses simple integer indexing throughout.

---

## Optimization Strategy

| Aspect | Current | Redesigned |
|---|---|---|
| Lookup granularity | cell×year (6.46M entries) | cell only (344K entries) |
| Lookup construction | String paste + named vector hash | Integer position mapping, built once |
| Stats loop iterations | 6.46M per variable | 344K per year per variable (= 9.6M total, but each iteration is trivial integer indexing) |
| Key mechanism | Character keys | Integer row indices within year-slices |
| Memory | 6.46M-element list of integer vectors | 344K-element list (reused across years and variables) |
| Estimated time | 86+ hours | **Minutes** |

### Steps

1. **Build a cell-level neighbor lookup** — a list of length 344K where element `i` contains the integer positions of cell `i`'s neighbors within the cell-order vector. This is built once from `rook_neighbors_unique` and `id_order`.

2. **Sort/index data by (year, cell)** so that within each year-slice, row positions correspond directly to the cell-order positions. This makes neighbor indexing a direct integer offset.

3. **For each variable × each year**, extract the year-slice vector, compute max/min/mean over neighbor indices, and write results back.

4. **Feed the augmented `cell_data` to the pre-trained Random Forest** exactly as before — the output columns are numerically identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (year-varying) variable values.
# Produces numerically identical results to the original implementation.
# =============================================================================

library(data.table)

compute_neighbor_features_optimized <- function(cell_data,
                                                 id_order,
                                                 rook_neighbors_unique,
                                                 neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # STEP 1: Build the STATIC cell-level neighbor lookup (done ONCE)
  # -------------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length = length(id_order),

  # where element i contains integer indices (into id_order) of cell i's neighbors.
  # We store these directly — no string keys, no year dimension.
  
  n_cells <- length(id_order)
  
  # cell_neighbor_idx: list of length n_cells

  # Element i = integer vector of positions (in id_order) of neighbors of cell i.
  # spdep::nb objects already use this convention, but we sanitize:
  cell_neighbor_idx <- lapply(seq_len(n_cells), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep uses 0L to indicate no neighbors
    nb <- nb[nb != 0L]
    as.integer(nb)
  })
  
  # -------------------------------------------------------------------------
  # STEP 2: Convert to data.table and ensure consistent cell ordering per year
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  # Create a mapping from cell id to position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell position column
  dt[, cell_pos := id_to_pos[as.character(id)]]
  
  # Pre-allocate output columns for all neighbor features
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }
  
  # -------------------------------------------------------------------------
  # STEP 3: For each year, compute neighbor stats using cell-level topology
  # -------------------------------------------------------------------------
  years <- sort(unique(dt$year))
  
  for (yr in years) {
    # Get row indices for this year
    year_rows <- which(dt$year == yr)
    
    # Build a vector indexed by cell_pos for this year's rows
    # cell_positions present in this year
    year_cell_pos <- dt$cell_pos[year_rows]
    
    # Create a mapping: cell_pos -> row index in year_rows
    # We need a fast lookup: for a given cell_pos, what is its row in dt?
    # Use a pre-allocated vector of length n_cells
    pos_to_dtrow <- rep(NA_integer_, n_cells)
    pos_to_dtrow[year_cell_pos] <- year_rows
    
    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_",  var_name)
      col_min  <- paste0("neighbor_min_",  var_name)
      col_mean <- paste0("neighbor_mean_", var_name)
      
      # Extract the full variable vector for this year, indexed by cell_pos
      # Pre-allocate a vector of length n_cells (NA for missing cells)
      vals_by_pos <- rep(NA_real_, n_cells)
      vals_by_pos[year_cell_pos] <- dt[[var_name]][year_rows]
      
      # Now compute neighbor stats for each cell present this year
      # Vectorized approach using the cell_neighbor_idx list
      n_year <- length(year_rows)
      res_max  <- rep(NA_real_, n_year)
      res_min  <- rep(NA_real_, n_year)
      res_mean <- rep(NA_real_, n_year)
      
      for (j in seq_len(n_year)) {
        cp <- year_cell_pos[j]
        nb_positions <- cell_neighbor_idx[[cp]]
        if (length(nb_positions) == 0L) next
        
        nb_vals <- vals_by_pos[nb_positions]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0L) next
        
        res_max[j]  <- max(nb_vals)
        res_min[j]  <- min(nb_vals)
        res_mean[j] <- mean(nb_vals)
      }
      
      # Write back to dt
      set(dt, i = year_rows, j = col_max,  value = res_max)
      set(dt, i = year_rows, j = col_min,  value = res_min)
      set(dt, i = year_rows, j = col_mean, value = res_mean)
    }
    
    # Progress reporting
    message(sprintf("Year %d complete.", yr))
  }
  
  # -------------------------------------------------------------------------
  # STEP 4: Clean up and return
  # -------------------------------------------------------------------------
  dt[, cell_pos := NULL]
  
  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}


# =============================================================================
# USAGE — drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_optimized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The pre-trained Random Forest model is used exactly as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Acceleration (Optional C++ Inner Loop)

The inner `for (j in seq_len(n_year))` loop over ~344K cells is the remaining bottleneck. If even the optimized version is not fast enough, this can be replaced with an Rcpp function:

```r
# Optional: Rcpp acceleration for the inner loop
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_stats_cpp(NumericVector vals_by_pos,
                                  IntegerVector year_cell_pos,
                                  List cell_neighbor_idx) {
  int n = year_cell_pos.size();
  NumericMatrix out(n, 3); // columns: max, min, mean
  
  for (int j = 0; j < n; j++) {
    out(j, 0) = NA_REAL;
    out(j, 1) = NA_REAL;
    out(j, 2) = NA_REAL;
    
    int cp = year_cell_pos[j]; // 1-based
    IntegerVector nb = cell_neighbor_idx[cp - 1]; // 0-based list index
    
    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;
    int count = 0;
    
    for (int k = 0; k < nb.size(); k++) {
      int pos = nb[k]; // 1-based index into vals_by_pos
      if (pos < 1 || pos > vals_by_pos.size()) continue;
      double v = vals_by_pos[pos - 1];
      if (ISNA(v) || ISNAN(v)) continue;
      if (v > vmax) vmax = v;
      if (v < vmin) vmin = v;
      vsum += v;
      count++;
    }
    
    if (count > 0) {
      out(j, 0) = vmax;
      out(j, 1) = vmin;
      out(j, 2) = vsum / count;
    }
  }
  return out;
}
')
```

Then replace the inner R loop with:

```r
res <- neighbor_stats_cpp(vals_by_pos, year_cell_pos, cell_neighbor_idx)
res_max  <- res[, 1]
res_min  <- res[, 2]
res_mean <- res[, 3]
```

---

## Summary of Gains

| Metric | Original | Optimized (R) | Optimized (Rcpp) |
|---|---|---|---|
| Lookup build | ~6.46M string-key entries | 344K integer lists (once) | Same |
| Stats iterations | 6.46M × 5 vars = 32.3M | 344K × 28 yrs × 5 vars = 48.2M (but trivial ops) | Same, compiled |
| String operations | ~billions of paste/hash | **Zero** | Zero |
| Estimated wall time | 86+ hours | ~10–30 minutes | ~2–5 minutes |
| Numerical output | Baseline | **Identical** | **Identical** |
| RF model | Unchanged | **Unchanged** | **Unchanged** |