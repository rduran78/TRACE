 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via expensive `paste`/string-matching with named character vectors. The `idx_lookup` named-vector lookup (`idx_lookup[neighbor_keys]`) is O(n) per probe in the worst case and has massive overhead from string allocation/hashing across 6.46M rows.

2. **`compute_neighbor_stats` iterates over 6.46M list elements** in a serial `lapply` loop, extracting variable slices and computing max/min/mean. This is repeated 5 times (once per source variable), for a total of ~32.3 million R-level loop iterations with per-element allocation.

3. **The neighbor lookup conflates topology and time.** Rook neighbors are a *spatial* property — they do not change across years. Yet the current code re-resolves neighbor identities at the cell-year level, inflating the lookup structure by a factor of 28 (years). The actual spatial graph has only ~344K nodes and ~1.37M directed edges; the lookup should be built once on the spatial graph and then applied per-year via vectorized indexing.

**Estimated cost breakdown:**
- String key construction: ~6.46M `paste` calls × (1 + avg ~4 neighbors) ≈ 32M string operations.
- Named vector probing: ~26M named-vector lookups.
- Stats computation: 5 vars × 6.46M list iterations = 32.3M R-level function calls.
- Total: dominated by the O(tens-of-millions) R-level string and list operations → 86+ hours.

## Optimization Strategy

### Key Insight: Separate Topology from Time

The rook neighbor graph is purely spatial. For any given year, the neighbor of cell `i` in year `t` is simply the row corresponding to (neighbor_cell_id, t). If we sort/index the data by `(year, cell)`, we can compute an offset per year and convert spatial neighbor indices to row indices via arithmetic — no string operations needed.

### Plan

1. **Build a sparse adjacency structure once** from the `spdep::nb` object as integer vectors (CSR-like: row pointers + column indices). ~344K nodes, ~1.37M edges.

2. **Sort data by (id, year)** with a known cell ordering so that cell `i` in year `t` is at row `(t-1)*N + i` (or equivalently, sort by year then by cell rank). This gives O(1) row-index computation.

3. **Vectorized aggregation using sparse matrix multiplication** for mean, and analogous tricks for max/min. Specifically:
   - Construct a sparse binary adjacency matrix `A` (344K × 344K).
   - For each year, extract the variable column as a vector `v` of length N.
   - `A %*% v` gives neighbor sums; divide by `A %*% 1` (degree vector) for means.
   - For max/min, use a grouped operation over the CSR edge list.

4. **Stack year results** back together. This replaces 6.46M list iterations with 28 sparse matrix-vector multiplies (each ~1.37M nonzeros) — roughly 5 orders of magnitude faster.

5. **Use `data.table` for I/O and column binding**, `Matrix` package for sparse algebra, and Rcpp for the max/min aggregation (which has no direct sparse-matrix shortcut).

### Expected Speedup

- Sparse matrix-vector multiply for mean: 28 × 1.37M multiplications per variable = ~38M FLOPs per variable, done in optimized C (via `Matrix`). Seconds, not hours.
- Max/min via Rcpp CSR traversal: same edge count, pure C++ loops. Seconds.
- Total expected: **< 2 minutes** for all 5 variables on a 16GB laptop.

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean neighbor stats.
# Preserves the trained Random Forest model (no retraining).
# ==============================================================================

library(data.table)
library(Matrix)
library(Rcpp)

# --------------------------------------------------------------------------
# Step 0: Rcpp functions for CSR-based grouped max and min
# --------------------------------------------------------------------------
sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector csr_neighbor_max(IntegerVector row_ptr, IntegerVector col_idx,
                               NumericVector vals, int n) {
  NumericVector out(n, NA_REAL);
  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    if (start == end) continue;
    double mx = R_NegInf;
    bool found = false;
    for (int j = start; j < end; j++) {
      double v = vals[col_idx[j]];
      if (!R_IsNA(v)) {
        if (!found || v > mx) mx = v;
        found = true;
      }
    }
    if (found) out[i] = mx;
  }
  return out;
}

// [[Rcpp::export]]
NumericVector csr_neighbor_min(IntegerVector row_ptr, IntegerVector col_idx,
                               NumericVector vals, int n) {
  NumericVector out(n, NA_REAL);
  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    if (start == end) continue;
    double mn = R_PosInf;
    bool found = false;
    for (int j = start; j < end; j++) {
      double v = vals[col_idx[j]];
      if (!R_IsNA(v)) {
        if (!found || v < mn) mn = v;
        found = true;
      }
    }
    if (found) out[i] = mn;
  }
  return out;
}

// [[Rcpp::export]]
NumericVector csr_neighbor_mean(IntegerVector row_ptr, IntegerVector col_idx,
                                NumericVector vals, int n) {
  NumericVector out(n, NA_REAL);
  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    if (start == end) continue;
    double sm = 0.0;
    int cnt = 0;
    for (int j = start; j < end; j++) {
      double v = vals[col_idx[j]];
      if (!R_IsNA(v)) {
        sm += v;
        cnt++;
      }
    }
    if (cnt > 0) out[i] = sm / (double)cnt;
  }
  return out;
}
')

# --------------------------------------------------------------------------
# Step 1: Convert spdep::nb to CSR (compressed sparse row) once
#         id_order[k] is the cell id for position k in rook_neighbors_unique
# --------------------------------------------------------------------------
build_csr_from_nb <- function(nb_obj) {
  n <- length(nb_obj)
  # Build row pointers and column indices (0-indexed for Rcpp)
  row_ptr <- integer(n + 1L)
  # First pass: count neighbors
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    if (length(nbrs) == 1L && nbrs[1L] == 0L) {
      row_ptr[i + 1L] <- row_ptr[i]
    } else {
      row_ptr[i + 1L] <- row_ptr[i] + length(nbrs)
    }
  }
  total_edges <- row_ptr[n + 1L]
  col_idx <- integer(total_edges)
  # Second pass: fill column indices (convert to 0-indexed)
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (!(length(nbrs) == 1L && nbrs[1L] == 0L)) {
      len <- length(nbrs)
      col_idx[pos:(pos + len - 1L)] <- nbrs - 1L
      pos <- pos + len
    }
  }
  list(row_ptr = row_ptr, col_idx = col_idx, n = n, nnz = total_edges)
}

# --------------------------------------------------------------------------
# Step 2: Main pipeline
# --------------------------------------------------------------------------
run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique,
                                     neighbor_source_vars) {

  cat("Converting to data.table...\n")
  dt <- as.data.table(cell_data)
  N <- length(id_order)  # 344,208 spatial cells

  # --- Create a fast mapping: cell_id -> spatial index (1..N) ---
  id_to_sidx <- setNames(seq_len(N), as.character(id_order))

  # --- Add spatial index column ---
  dt[, sidx := id_to_sidx[as.character(id)]]

  # --- Sort by (year, sidx) so that within each year, rows are ordered
  #     by spatial index. This lets us extract per-year vectors trivially. ---
  setkey(dt, year, sidx)

  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_idx <- setNames(seq_along(years), as.character(years))

  cat("Building CSR adjacency from nb object...\n")
  csr <- build_csr_from_nb(rook_neighbors_unique)
  stopifnot(csr$n == N)
  cat(sprintf("  Nodes: %d, Directed edges: %d\n", csr$n, csr$nnz))

  # --- Verify data completeness: every (sidx, year) must be present ---
  #     If the panel is balanced (6,457,824 = 344208 * 28 * ~some factor
  #     accounting for 1992-2019 = 28 years), each year block has exactly N rows.
  rows_per_year <- dt[, .N, by = year]
  if (!all(rows_per_year$N == N)) {
    warning("Panel is unbalanced. Some cells missing in some years.\n",
            "Falling back to index-based lookup (still fast).")
    balanced <- FALSE
  } else {
    balanced <- TRUE
    cat(sprintf("  Balanced panel confirmed: %d cells x %d years = %d rows\n",
                N, n_years, N * n_years))
  }

  # --- Compute neighbor stats for each variable ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s ...\n", var_name))
    t0 <- proc.time()

    max_col <- paste0("n_", var_name, "_max")
    min_col <- paste0("n_", var_name, "_min")
    mean_col <- paste0("n_", var_name, "_mean")

    # Pre-allocate result vectors for the entire dataset
    res_max  <- numeric(nrow(dt))
    res_min  <- numeric(nrow(dt))
    res_mean <- numeric(nrow(dt))

    if (balanced) {
      # Fast path: rows for year t are at positions ((t_idx-1)*N + 1):(t_idx*N)
      # and within that block, row j corresponds to sidx = j.
      for (yr in years) {
        t_idx <- year_to_idx[as.character(yr)]
        row_start <- (t_idx - 1L) * N + 1L
        row_end   <- t_idx * N
        block_rows <- row_start:row_end

        vals <- dt[[var_name]][block_rows]

        res_max[block_rows]  <- csr_neighbor_max(csr$row_ptr, csr$col_idx, vals, N)
        res_min[block_rows]  <- csr_neighbor_min(csr$row_ptr, csr$col_idx, vals, N)
        res_mean[block_rows] <- csr_neighbor_mean(csr$row_ptr, csr$col_idx, vals, N)
      }
    } else {
      # Slower but correct path for unbalanced panels
      for (yr in years) {
        yr_rows <- which(dt$year == yr)
        # Map sidx to position within this year block
        yr_sidx <- dt$sidx[yr_rows]

        # Build a full-length vector for this year (NA for missing cells)
        vals_full <- rep(NA_real_, N)
        vals_full[yr_sidx] <- dt[[var_name]][yr_rows]

        mx  <- csr_neighbor_max(csr$row_ptr, csr$col_idx, vals_full, N)
        mn  <- csr_neighbor_min(csr$row_ptr, csr$col_idx, vals_full, N)
        mn_ <- csr_neighbor_mean(csr$row_ptr, csr$col_idx, vals_full, N)

        res_max[yr_rows]  <- mx[yr_sidx]
        res_min[yr_rows]  <- mn[yr_sidx]
        res_mean[yr_rows] <- mn_[yr_sidx]
      }
    }

    # Assign columns in place (data.table reference semantics)
    set(dt, j = max_col,  value = res_max)
    set(dt, j = min_col,  value = res_min)
    set(dt, j = mean_col, value = res_mean)

    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds\n", elapsed))
  }

  # --- Restore original row order if needed ---
  # The caller may need original ordering for the RF model.
  # Re-sort to match original cell_data order.
  if (is.data.frame(cell_data)) {
    # Reconstruct original order key
    orig_key <- paste(cell_data$id, cell_data$year, sep = "_")
    dt[, orig_key := paste(id, year, sep = "_")]
    dt <- dt[match(orig_key, dt$orig_key)]
    dt[, c("sidx", "orig_key") := NULL]
  }

  cat("Neighbor feature computation complete.\n")
  return(dt)
}

# --------------------------------------------------------------------------
# Step 3: Execute and predict
# --------------------------------------------------------------------------

# --- Configuration ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# --- Run optimized aggregation ---
cell_data_dt <- run_neighbor_aggregation(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# --- Apply the pre-trained Random Forest (no retraining) ---
# The RF model object (e.g., rf_model) is already in memory.
# Ensure the predictor columns match what the model expects.
predictions <- predict(rf_model, newdata = cell_data_dt)
cell_data_dt[, predicted := predictions]
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | ~6.46M string-keyed list entries | CSR built once from nb object (344K nodes, 1.37M edges) |
| **Key construction** | `paste()` + named vector probe per row | Integer arithmetic: `(year_idx - 1) * N + sidx` |
| **Stats computation** | R-level `lapply` over 6.46M elements × 5 vars | Rcpp CSR traversal: 28 years × 5 vars = 140 sparse sweeps over 1.37M edges |
| **Memory** | ~6.46M-element list of integer vectors + copies | CSR: two integer vectors totaling ~2.7M entries + per-year vectors of length 344K |
| **Peak RAM** | Potentially >16GB (string inflation) | ~2–3 GB comfortably within 16GB |
| **Expected runtime** | 86+ hours | **~1–3 minutes** |
| **Numerical equivalence** | Baseline | ✅ Identical max, min, mean per node-year (same NA handling logic) |
| **RF model** | Pre-trained | ✅ Untouched — `predict()` only |

The Rcpp CSR functions replicate the exact NA-skipping logic of the original: skip `NA` neighbor values, return `NA` if all neighbors are `NA` or if a node has zero neighbors. This guarantees identical numerical output.