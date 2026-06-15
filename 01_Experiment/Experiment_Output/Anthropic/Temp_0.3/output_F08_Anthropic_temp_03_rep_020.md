 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cell×year) neighbor indices.** It creates a list of ~6.46 million entries, each requiring string-pasting, hash lookups, and NA filtering. But the neighbor *relationships* are purely spatial — cell A neighbors cell B regardless of year. The only thing that changes across years is the *values* attached to cells. This means the lookup is doing 28× redundant work (once per year per cell, when the topology is year-invariant).

2. **`compute_neighbor_stats` iterates over ~6.46 million rows** with `lapply`, indexing into a flat cell-year vector. Each call pulls values from scattered positions in a 6.46M-length vector. This is cache-unfriendly and R-loop-heavy.

3. **String key construction** (`paste(id, year, sep="_")`) and named-vector lookups (`idx_lookup[neighbor_keys]`) are O(n) hash operations repeated millions of times — enormously wasteful.

### The Key Insight

- **Static:** The neighbor graph (which cell IDs neighbor which cell IDs) — 344,208 cells, ~1.37M directed edges. This never changes.
- **Dynamic:** The variable values attached to each cell — these change by year.

The redesign should: build the neighbor structure **once over cells only** (344K entries, not 6.46M), then for each year, slice the relevant variable column, compute neighbor max/min/mean using vectorized operations over the static cell-indexed neighbor list.

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** — a list of length 344,208 where each element contains integer indices (1-based positions in `id_order`) of that cell's neighbors. This is just a cleaned version of `rook_neighbors_unique` and costs essentially nothing.

2. **For each variable and each year**, subset the data to that year (or index into a cell-indexed vector), pull neighbor values using the static cell-level lookup, and compute max/min/mean with vectorized R or a fast compiled helper.

3. **Use `data.table`** for efficient subsetting, column assignment, and join-free indexing by cell position.

4. **Vectorize the inner loop** using `vapply` over 344K cells (not 6.46M rows) per year, and parallelize across years or variables if needed.

### Expected Speedup

- Lookup construction: 6.46M → 344K entries = **~19× faster**.
- Stats computation: operating on 344K cells × 28 years with vectorized column access instead of 6.46M string-keyed lookups = **~20-50× faster**.
- Overall: from ~86 hours to **~1-3 hours** on a 16 GB laptop, potentially under 1 hour with `data.table` and careful memory management.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Convert to data.table if not already
# ============================================================
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build STATIC cell-level neighbor lookup (once)
#
# rook_neighbors_unique is an nb object (list of integer vectors)
# indexed by position in id_order. We just need to clean it:
# remove 0L entries (spdep uses 0L for "no neighbors").
# ============================================================
build_cell_neighbor_lookup <- function(neighbors) {
  # neighbors is an nb object: list of integer index vectors

  # Each element i contains the positional indices (into id_order)
  # of the neighbors of cell id_order[i].
  # spdep encodes "no neighbors" as integer(0) or 0L.
  lapply(neighbors, function(nb_idx) {
    nb_idx <- nb_idx[nb_idx > 0L]
    as.integer(nb_idx)
  })
}

cell_neighbor_lookup <- build_cell_neighbor_lookup(rook_neighbors_unique)
# cell_neighbor_lookup[[i]] = integer vector of positional indices
# into id_order for the neighbors of cell id_order[i].

# ============================================================
# STEP 2: Build a mapping from cell id -> position in id_order
# ============================================================
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

# ============================================================
# STEP 3: Ensure cell_data is keyed and has a cell position column
# ============================================================
cell_data[, cell_pos := id_to_pos[as.character(id)]]

# Verify no NAs (every id in cell_data must be in id_order)
stopifnot(!anyNA(cell_data$cell_pos))

# Key by year and cell_pos for fast subsetting
setkey(cell_data, year, cell_pos)

# ============================================================
# STEP 4: Compute neighbor stats — static topology, dynamic values
# ============================================================
compute_neighbor_features <- function(dt, cell_nb_lookup, var_name,
                                      id_order_vec) {

  n_cells <- length(id_order_vec)
  max_col <- paste0("neighbor_max_", var_name)
  min_col <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Pre-allocate output columns
  dt[, (max_col) := NA_real_]
  dt[, (min_col) := NA_real_]
  dt[, (mean_col) := NA_real_]

  years <- sort(unique(dt$year))

  for (yr in years) {
    # --- Extract a cell-position-indexed vector of values for this year ---
    # Because we keyed by (year, cell_pos), subset is fast
    yr_rows <- dt[.(yr)]  # subset by year via key

    # Build a dense vector: position i -> value for cell at position i
    # Some cells may be missing for a year; those stay NA.
    vals_by_pos <- rep(NA_real_, n_cells)
    vals_by_pos[yr_rows$cell_pos] <- yr_rows[[var_name]]

    # --- Compute neighbor stats for each cell present this year ---
    cell_positions <- yr_rows$cell_pos

    # Vectorized computation over cells present this year
    stats <- vapply(cell_positions, function(cp) {
      nb_pos <- cell_nb_lookup[[cp]]
      if (length(nb_pos) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nv <- vals_by_pos[nb_pos]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nv), min(nv), mean(nv))
    }, numeric(3))
    # stats is 3 x length(cell_positions)

    # --- Write results back into the data.table ---
    # We need the row indices in the *original* dt, not in yr_rows.
    # Since dt is keyed by (year, cell_pos), we can use a join to assign.
    # But more directly, we can find the row indices:
    row_idx <- dt[.(yr), which = TRUE]

    set(dt, i = row_idx, j = max_col,  value = stats[1, ])
    set(dt, i = row_idx, j = min_col,  value = stats[2, ])
    set(dt, i = row_idx, j = mean_col, value = stats[3, ])

    if (interactive()) {
      cat(sprintf("  %s | year %d done\n", var_name, yr))
    }
  }

  invisible(dt)
}

# ============================================================
# STEP 5: Run for all neighbor source variables
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("Computing neighbor features for: %s\n", var_name))
  compute_neighbor_features(cell_data, cell_neighbor_lookup,
                            var_name, id_order)
}

# ============================================================
# STEP 6: Clean up helper column, restore original class if needed
# ============================================================
cell_data[, cell_pos := NULL]

# ============================================================
# STEP 7: Predict with the pre-trained Random Forest (unchanged)
# ============================================================
# The trained RF model object and prediction code remain exactly
# as they were. The output columns have identical names and
# identical numerical values (same estimand), so predictions
# are fully preserved.
#
# Example (unchanged from original pipeline):
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Aspect | Original | Redesigned |
|---|---|---|
| Neighbor definition | Rook contiguity from `rook_neighbors_unique` | Same `rook_neighbors_unique`, same positional indices |
| Neighbor values source | `data[[var_name]][idx]` where `idx` matched by `(id, year)` | `vals_by_pos[nb_pos]` where `vals_by_pos` is indexed by cell position for the same year |
| Aggregation functions | `max()`, `min()`, `mean()` with NA removal | Identical `max()`, `min()`, `mean()` with identical NA removal |
| Output column names | `neighbor_max_<var>`, `neighbor_min_<var>`, `neighbor_mean_<var>` | Identical |
| RF model | Pre-trained, not retrained | Pre-trained, not retrained — receives identical feature columns |

The computation is mathematically identical; only the execution strategy differs — exploiting the static-vs-dynamic decomposition to eliminate ~95% of redundant work.

### Optional Further Speedup: Rcpp Inner Loop

If the `vapply` over 344K cells is still a bottleneck (~15 min per variable×year), the inner loop can be pushed to C++ via Rcpp:

```r
Rcpp::cppFunction('
NumericMatrix neighbor_stats_cpp(NumericVector vals_by_pos,
                                 List cell_nb_lookup,
                                 IntegerVector cell_positions) {
  int n = cell_positions.size();
  NumericMatrix out(n, 3); // max, min, mean

  for (int i = 0; i < n; i++) {
    int cp = cell_positions[i] - 1; // 0-based
    IntegerVector nb = cell_nb_lookup[cp];
    int count = 0;
    double vmax = R_NegInf, vmin = R_PosInf, vsum = 0.0;

    for (int j = 0; j < nb.size(); j++) {
      double v = vals_by_pos[nb[j] - 1]; // 0-based
      if (!NumericVector::is_na(v)) {
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
        vsum += v;
        count++;
      }
    }
    if (count == 0) {
      out(i, 0) = NA_REAL;
      out(i, 1) = NA_REAL;
      out(i, 2) = NA_REAL;
    } else {
      out(i, 0) = vmax;
      out(i, 1) = vmin;
      out(i, 2) = vsum / count;
    }
  }
  return out;
}
')
```

This would replace the `vapply` call inside `compute_neighbor_features` and bring per-year-per-variable time down to seconds, reducing total runtime to **under 10 minutes**.