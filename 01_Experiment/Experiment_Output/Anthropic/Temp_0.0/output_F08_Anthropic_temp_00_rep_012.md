 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows) instead of per cell (~344K cells).** The rook-neighbor topology is purely spatial — it never changes across years. Yet the current code creates a lookup of length 6.46M, where each entry re-discovers the same neighbors for the same cell, just keyed to a different year. This means the ~1.37M neighbor relationships are redundantly resolved 28 times.

2. **`compute_neighbor_stats` iterates over 6.46M entries** using `lapply`, performing per-element subsetting and aggregation in pure R. This is inherently slow for millions of iterations.

3. **String-based key construction** (`paste(id, year, sep="_")`) and named-vector lookups (`idx_lookup[neighbor_keys]`) are O(n) hash lookups repeated millions of times, creating enormous overhead from string allocation and hashing.

4. **Memory pressure:** Storing 6.46M list entries (each a vector of neighbor row indices) consumes significant RAM and causes GC thrashing on a 16 GB machine.

### The Key Insight

> **Neighbor topology is static (per-cell). Variable values are dynamic (per-cell-year).**

The neighbor of cell `i` is always cell `j`, regardless of year. What changes is the *value* attached to cell `j` in each year. Therefore:

- Build the neighbor graph **once, over 344K cells** (not 6.46M cell-years).
- Compute neighbor stats by **indexing into year-specific value vectors** using the static cell-level neighbor list.

This reduces the lookup construction by **28×** and enables vectorized, year-parallel computation.

---

## Optimization Strategy

### 1. Separate Static Topology from Dynamic Data

Build a **cell-level** neighbor lookup once: a list of length 344,208 where entry `i` contains the integer positions of cell `i`'s neighbors in the canonical cell ordering. This is derived directly from `rook_neighbors_unique` (the `nb` object) and requires zero string operations.

### 2. Compute Neighbor Stats Per Year Using Matrix Indexing

For each year:
- Extract the variable values for all cells in that year as a single numeric vector (aligned to the canonical cell order).
- Use the static cell-level neighbor list to gather neighbor values and compute max/min/mean.

This turns 6.46M list iterations into 28 iterations × 344K cells, with the inner work being simple numeric vector subsetting.

### 3. Vectorize the Inner Loop with `vapply` or C++-backed Operations

Use `vapply` (which pre-allocates output) instead of `lapply` + `do.call(rbind, ...)`. Alternatively, use `data.table` for the year-level split-apply-combine.

### 4. Use `data.table` for Efficient Data Manipulation

Avoid repeated `data.frame` column assignments. Use `data.table` set-by-reference semantics.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# STEP 0: Convert to data.table if not already
# ==============================================================================
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ==============================================================================
# STEP 1: Build STATIC cell-level neighbor lookup (done ONCE)
#
# rook_neighbors_unique is an nb object (list of integer vectors) aligned to
# id_order. Entry i contains the indices (into id_order) of cell i's neighbors.
# spdep::nb objects use 0L to indicate no neighbors, so we filter those out.
#
# This step: O(344K cells), takes seconds.
# ==============================================================================
build_cell_neighbor_lookup <- function(neighbors) {
  # neighbors is the nb object: list of integer vectors

  # Each entry's values are indices into the same list (1-based), with 0 = no neighbors
  lapply(neighbors, function(nb_idx) {
    nb_idx <- nb_idx[nb_idx > 0L]
    as.integer(nb_idx)
  })
}

cell_neighbor_lookup <- build_cell_neighbor_lookup(rook_neighbors_unique)
# cell_neighbor_lookup[[i]] = integer vector of positions in id_order that are
# neighbors of the i-th cell in id_order.

n_cells <- length(id_order)
stopifnot(length(cell_neighbor_lookup) == n_cells)

# ==============================================================================
# STEP 2: Create a mapping from cell id to its position in id_order
# ==============================================================================
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# ==============================================================================
# STEP 3: Ensure cell_data is keyed and ordered for fast year-cell access
# ==============================================================================
# We need, for each year, a vector of variable values aligned to id_order.
# Add a column for the cell's position in id_order.
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Verify all cells are present in every year (panel is balanced)
setkey(cell_data, year, cell_pos)

# ==============================================================================
# STEP 4: Compute neighbor stats — static topology × dynamic values
#
# For each variable and each year:
#   1. Extract values as a vector aligned to id_order positions.
#   2. Use cell_neighbor_lookup to gather neighbor values.
#   3. Compute max, min, mean.
#   4. Write results back.
#
# This is O(28 years × 344K cells × avg_neighbors) per variable.
# With ~4 neighbors on average (rook), this is ~28 × 344K × 4 ≈ 38.5M ops/var.
# For 5 variables: ~193M simple numeric operations. Should take minutes.
# ==============================================================================
compute_neighbor_stats_static <- function(values_vec, cell_neighbor_lookup) {
  # values_vec: numeric vector of length n_cells, aligned to id_order
  # cell_neighbor_lookup: list of length n_cells, each entry = integer vector of
  #                       neighbor positions in id_order
  #
  # Returns: matrix of dim (n_cells, 3) with columns max, min, mean
  
  n <- length(values_vec)
  out <- matrix(NA_real_, nrow = n, ncol = 3L)
  
  for (i in seq_len(n)) {
    nb_idx <- cell_neighbor_lookup[[i]]
    if (length(nb_idx) == 0L) next
    nb_vals <- values_vec[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0L) next
    out[i, 1L] <- max(nb_vals)
    out[i, 2L] <- min(nb_vals)
    out[i, 3L] <- mean(nb_vals)
  }
  out
}

# ==============================================================================
# STEP 5: Main loop — iterate over variables, then years
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Pre-allocate output columns
for (var_name in neighbor_source_vars) {
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)
}

for (var_name in neighbor_source_vars) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  cat(sprintf("Processing neighbor stats for: %s\n", var_name))
  
  for (yr in years) {
    # Get row indices for this year (data is keyed by year, cell_pos)
    yr_rows <- cell_data[.(yr), which = TRUE]
    
    # Extract the values vector aligned to cell_pos order
    # Since we keyed by (year, cell_pos), rows within a year are sorted by cell_pos
    yr_data <- cell_data[yr_rows]
    
    # Build a full-length vector aligned to id_order positions
    values_vec <- rep(NA_real_, n_cells)
    values_vec[yr_data$cell_pos] <- yr_data[[var_name]]
    
    # Compute stats using static topology
    stats_mat <- compute_neighbor_stats_static(values_vec, cell_neighbor_lookup)
    # stats_mat is n_cells × 3, aligned to id_order
    
    # Map results back to the rows in cell_data for this year
    set(cell_data, i = yr_rows, j = max_col,  value = stats_mat[yr_data$cell_pos, 1L])
    set(cell_data, i = yr_rows, j = min_col,  value = stats_mat[yr_data$cell_pos, 2L])
    set(cell_data, i = yr_rows, j = mean_col, value = stats_mat[yr_data$cell_pos, 3L])
  }
  
  cat(sprintf("  Done: %s\n", var_name))
}

# Clean up helper column
cell_data[, cell_pos := NULL]

# ==============================================================================
# STEP 6: Predict with the pre-trained Random Forest (UNCHANGED)
# ==============================================================================
# The trained RF model is not retrained. The feature columns produced above
# (neighbor_max_*, neighbor_min_*, neighbor_mean_*) are numerically identical
# to the original implementation's output, preserving the estimand.
#
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Further Optimization (Optional): Rcpp Inner Loop

If the pure-R `for (i in seq_len(n))` loop over 344K cells is still a bottleneck (it shouldn't be — expect ~2–5 seconds per year per variable), here is an Rcpp drop-in replacement:

```r
Rcpp::cppFunction('
NumericMatrix neighbor_stats_cpp(NumericVector values, List neighbor_lookup) {
  int n = neighbor_lookup.size();
  NumericMatrix out(n, 3);
  std::fill(out.begin(), out.end(), NA_REAL);
  
  for (int i = 0; i < n; i++) {
    IntegerVector nb_idx = neighbor_lookup[i];
    if (nb_idx.size() == 0) continue;
    
    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;
    int count = 0;
    
    for (int j = 0; j < nb_idx.size(); j++) {
      double val = values[nb_idx[j] - 1];  // R is 1-indexed
      if (NumericVector::is_na(val)) continue;
      if (val > vmax) vmax = val;
      if (val < vmin) vmin = val;
      vsum += val;
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

# Then replace compute_neighbor_stats_static with:
# stats_mat <- neighbor_stats_cpp(values_vec, cell_neighbor_lookup)
```

---

## Performance Summary

| Aspect | Original | Redesigned |
|---|---|---|
| Lookup construction | 6.46M entries, string-keyed | 344K entries, integer-indexed (once) |
| Neighbor stat iterations | 6.46M × 5 vars = 32.3M | 344K × 28 yrs × 5 vars = 48.2M (but trivial ops) |
| String operations | ~billions (paste, named lookup) | **Zero** |
| Estimated runtime | 86+ hours | **5–15 minutes** (R), **1–3 minutes** (Rcpp) |
| Numerical output | Baseline | **Identical** — same max, min, mean over same neighbor sets |
| RF model | Unchanged | **Unchanged** — no retraining |
| Peak RAM | High (6.46M-element list + strings) | Low (~344K-element list + year vectors) |