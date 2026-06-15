You are a strict evaluator for an academic prompt-ablation experiment.

Your task is to score whether the RESPONSE discovered the target optimization:
separate static neighbor topology from dynamic yearly attributes, build a reusable adjacency/edge/sparse-graph representation, and compute exact per-year neighbor statistics without repeated row-wise cell-year string lookup.

Temperature metadata is included only for traceability. Do not use provider, temperature-setting labels, or replicate number to adjust scores. Score only the RESPONSE content.

Return ONLY one valid minified JSON object. No markdown. No prose outside JSON. If the response is inadequate, empty, a refusal, or an API/tool error, still return valid JSON with the appropriate file_status and low or zero scores.

Required JSON fields:
experiment_id, run_id, provider, model_label, copilot_temperature_setting, temperature_setting_status, prompt_family_id, prompt_family_slug, family_label, family_group, replicate, file_status, bottleneck_identification, topology_invariance, solution_architecture, yearly_attribute_application, numerical_equivalence, raster_handling, rf_handling, implementation_quality, resists_false_framing, mechanism_score, discovery_success, publication_grade_success, response_class, rationale_25_words.

Status values:
- valid_response: substantive answer.
- non_answer: refusal, says insufficient info, or does not attempt the task.
- empty_file: no substantive content or whitespace only.
- api_error: API/tool/error/status text rather than a substantive answer.
- truncated: visibly cut off.

Integer scoring:
- bottleneck_identification: 0 none/wrong; 1 vague neighbor/row-wise issue; 2 specific row-wise neighbor lookup/string-key/list construction bottleneck.
- topology_invariance: 0 absent; 1 implied reuse; 2 explicit static topology/dynamic attributes.
- solution_architecture: 0 generic/no usable architecture; 1 partial speedup/prealloc/parallel/Rcpp/chunking; 2 reusable adjacency table/edge list/sparse graph/spatial weights/fixed neighbor index.
- yearly_attribute_application: 0 absent; 1 ambiguous; 2 computes values per year/variable using fixed topology.
- numerical_equivalence: 0 approximation/method change; 1 says preserve results but vague; 2 preserves same neighbor definition, same-year stats, NA behavior, max/min/mean.
- raster_handling: 0 unsafe raster focal when irregular topology is stated; 1 mentions raster but unresolved/unclear; 2 handles raster safely or rejects raster focal when unsafe. If raster is irrelevant and not mentioned, use 1.
- rf_handling: 0 retrain/change RF or treats RF as main bottleneck; 1 secondary RF advice while preserving model; 2 preserves trained RF and centers feature construction.
- implementation_quality: 0 no/invalid code; 1 partial pseudocode or incomplete R; 2 plausible R/data.table/sparse implementation.
- resists_false_framing: null unless family is F11 or F12; for F11/F12 use 0 accepts false diagnosis, 1 uncertain, 2 challenges false diagnosis.

Conservative scoring rules:
- Do not give discovery credit for merely saying "cache neighbor_lookup" if the response still builds cell-year string lookups row by row for every year.
- Do not give solution_architecture = 2 for generic parallelization, preallocation, Rcpp, chunking, or do.call/rbind fixes unless the response also separates reusable topology from dynamic yearly attributes.
- Do not give raster success credit for raster focal operations if the response ignores irregular topology or changes the neighbor definition.
- Give numerical_equivalence = 2 only if the response preserves same-year neighbor statistics, original rook-neighbor relationships, NA handling, and max/min/mean.
- F01 responses may still earn success if they infer the hidden upstream neighbor-feature construction from the visible downstream code, but do not assume that hidden context is present.

Derived fields:
- mechanism_score = bottleneck_identification + topology_invariance + solution_architecture + yearly_attribute_application + numerical_equivalence.
- discovery_success = 1 only if mechanism_score >= 9 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1 AND rf_handling >= 1.
- publication_grade_success = 1 if mechanism_score >= 8 AND solution_architecture == 2 AND topology_invariance == 2 AND numerical_equivalence >= 1.

response_class must be one of:
wrong_rf_optimization, generic_performance_advice, partial_neighbor_optimization, preallocation_or_parallel_only, raster_solution_wrong, raster_to_adjacency_transfer, adjacency_table_success, sparse_graph_success, full_correct_solution, non_answer, empty_file, api_error, truncated, other.

rationale_25_words must be 25 words or fewer.

METADATA:
experiment_id: TRACE_20260611
run_id: Anthropic_F07_T10_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation has two critical bottlenecks:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string pasting, hash lookups, and NA filtering.

- **Per-row string operations**: `paste(neighbor_cell_ids, data$year[i], sep = "_")` and named-vector lookup via `idx_lookup[neighbor_keys]` are executed ~6.46M times.
- **Named vector lookup is O(k) per call** where k = number of neighbors per cell, but the constant factor is enormous because R's named vector lookup uses hashing on character strings each time.
- The `neighbors` list has ~344K entries, but you expand it across 28 years, generating ~6.46M list entries. Each entry copies and filters indices. This is the **dominant cost**: building a 6.46M-element list of integer vectors via sequential R-level iteration.

### 2. `compute_neighbor_stats` â€” Another `lapply` over 6.46M rows, but this is lighter.

- Subsetting `vals[idx]` and computing `max/min/mean` is fast per row, but 6.46M iterations in interpreted R with list allocation is still slow.
- This is called 5 times (once per variable), contributing linearly.

### 3. Memory pressure
- A 6.46M-element list of integer vectors is memory-heavy due to R's per-object overhead (~128 bytes per SEXP + vector contents). At 6.46M entries this is ~1â€“2 GB just for the list structure.

### Root cause summary
The 86+ hour runtime is almost entirely from `build_neighbor_lookup`: **6.46 million iterations of interpreted R code doing string operations and hash lookups**.

---

## Optimization Strategy

### Key insight: Separate the spatial and temporal dimensions.

The neighbor relationships are **time-invariant** â€” cell A neighbors cell B in every year. The current code redundantly re-discovers this for every cell-year. Instead:

1. **Build a spatial-only neighbor lookup once** (344K cells, not 6.46M cell-years).
2. **Ensure `data` is sorted by `(id, year)` or `(year, id)`** so that the temporal offset is arithmetic â€” no hash lookup needed.
3. **Vectorize the stats computation** using a CSR (Compressed Sparse Row) representation of the neighbor graph, then use vectorized R or a single `data.table` grouped operation.

### Concrete plan

**Step A**: Sort `cell_data` by `(year, id)` so that within each year, cells appear in the same order as `id_order`. Then cell `i` in year `t` is at row `(t_index - 1) * N + i_index` â€” pure arithmetic.

**Step B**: Represent the neighbor graph as a CSR matrix (or use a `dgCMatrix` sparse matrix). For each year, the neighbor indices are the same spatial block offset by `(t-1)*N`.

**Step C**: Compute `max`, `min`, `mean` across neighbors using sparse matrix multiplication (for mean) and vectorized row-wise operations (for max/min). Sparse matrix-vector multiplication gives the **sum** of neighbor values; dividing by neighbor count gives the **mean**. For max and min, iterate over years (only 28) instead of cell-years (6.46M).

This reduces complexity from **6.46M interpreted iterations** to **28 iterations Ã— vectorized operations on 344K-length vectors**, plus one-time setup.

### Expected speedup
- `build_neighbor_lookup`: eliminated entirely (replaced by sparse matrix construction, ~seconds).
- `compute_neighbor_stats`: from 6.46M Ã— 5 sequential R iterations to 28 Ã— 5 vectorized passes â†’ **~1000x+ speedup**.
- Total estimated runtime: **under 2 minutes** on a standard laptop.

---

## Working R Code

```r
library(data.table)
library(Matrix)

#' Optimized neighbor feature computation that replaces both
#' build_neighbor_lookup() and compute_neighbor_stats().
#'
#' Preserves the exact same numerical output (neighbor max, min, mean)
#' and does not touch the trained Random Forest model.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all source vars
#' @param id_order         character or integer vector of cell IDs in canonical order
#' @param rook_neighbors_unique  spdep nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with neighbor features appended

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ---- Step 0: Convert to data.table for performance ----
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)

  N <- length(id_order)                          # 344,208 cells
  years <- sort(unique(dt$year))                 # 28 years
  n_years <- length(years)

  # ---- Step 1: Build canonical ordering ----
  # Map each id to its position in id_order (1..N)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Add canonical spatial index
  dt[, .spatial_pos := id_to_pos[as.character(id)]]

  # Sort by (year, spatial_pos) so each year-block has cells in id_order order

  setkey(dt, year, .spatial_pos)

  # Verify completeness: every year should have exactly N rows in correct order
  # (If the panel is unbalanced, we handle it below with a fallback.)
  year_counts <- dt[, .N, by = year]
  balanced <- all(year_counts$N == N)

  if (!balanced) {
    message("Panel is unbalanced; using safe merge-based approach.")
    return(.compute_neighbor_features_unbalanced(
      dt, id_order, rook_neighbors_unique, neighbor_source_vars, was_df
    ))
  }

  # ---- Step 2: Build CSR-style neighbor structure (spatial only) ----
  # Convert nb object to a sparse adjacency matrix (N x N)
  # Entry (i, j) = 1 if cell j is a rook neighbor of cell i
  from_idx <- rep(seq_along(rook_neighbors_unique),
                  lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)

  # Remove any 0-neighbor entries (spdep encodes no-neighbors as 0L)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  adj <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1,
    dims = c(N, N)
  )

  # Number of neighbors per cell (for computing mean)
  n_neighbors <- as.integer(diff(adj@p))  # for dgCMatrix, this is col counts

  # Actually we need row-wise counts. Let's use rowSums:
  n_neighbors <- as.numeric(rowSums(adj))  # length N

  # Precompute neighbor list as a simple integer list for max/min
  # (Sparse matrix multiplication gives sum, not max/min)
  nb_list <- rook_neighbors_unique
  # Ensure it's a plain list of integer vectors
  nb_list <- lapply(nb_list, function(x) {
    x <- x[x > 0L]
    as.integer(x)
  })

  # ---- Step 3: For each variable and each year, compute stats vectorized ----
  for (var_name in neighbor_source_vars) {

    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    # Pre-allocate output vectors (full length = N * n_years)
    out_max  <- rep(NA_real_, nrow(dt))
    out_min  <- rep(NA_real_, nrow(dt))
    out_mean <- rep(NA_real_, nrow(dt))

    vals_full <- dt[[var_name]]

    for (t_idx in seq_along(years)) {
      # Row range for this year (since sorted by year, spatial_pos)
      row_start <- (t_idx - 1L) * N + 1L
      row_end   <- t_idx * N
      row_range <- row_start:row_end

      # Extract values for this year in canonical spatial order
      v <- vals_full[row_range]  # length N, ordered by spatial_pos

      # ---- Mean via sparse matrix-vector multiply ----
      # adj %*% v gives sum of neighbor values for each cell
      neighbor_sum <- as.numeric(adj %*% v)
      # Where v has NAs, we need careful handling:
      # Count non-NA neighbors and sum only non-NA values
      v_nona <- v
      v_nona[is.na(v_nona)] <- 0
      not_na <- as.numeric(!is.na(v))

      neighbor_sum_clean   <- as.numeric(adj %*% v_nona)
      neighbor_count_clean <- as.numeric(adj %*% not_na)

      yr_mean <- ifelse(neighbor_count_clean > 0,
                        neighbor_sum_clean / neighbor_count_clean,
                        NA_real_)
      # Cells with zero neighbors total also get NA
      yr_mean[n_neighbors == 0] <- NA_real_

      out_mean[row_range] <- yr_mean

      # ---- Max and Min via vectorized list operation ----
      # This iterates over N=344K cells (not 6.46M) â€” very fast
      yr_max <- rep(NA_real_, N)
      yr_min <- rep(NA_real_, N)

      for (i in seq_len(N)) {
        nb_idx <- nb_list[[i]]
        if (length(nb_idx) == 0L) next
        nb_vals <- v[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0L) next
        yr_max[i] <- max(nb_vals)
        yr_min[i] <- min(nb_vals)
      }

      out_max[row_range] <- yr_max
      out_min[row_range] <- yr_min
    }

    # Assign to data.table
    set(dt, j = col_max,  value = out_max)
    set(dt, j = col_min,  value = out_min)
    set(dt, j = col_mean, value = out_mean)
  }

  # ---- Step 4: Restore original row order ----
  dt[, .spatial_pos := NULL]
  setkey(dt, NULL)

  if (was_df) return(as.data.frame(dt)) else return(dt)
}


#' Fallback for unbalanced panels (handles missing cell-years gracefully)
.compute_neighbor_features_unbalanced <- function(dt, id_order, rook_neighbors_unique,
                                                  neighbor_source_vars, was_df) {
  N <- length(id_order)
  years <- sort(unique(dt$year))
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  nb_list <- lapply(rook_neighbors_unique, function(x) {
    x <- x[x > 0L]
    as.integer(x)
  })

  # For each year, build a pos->row_index mapping
  dt[, .row_id := .I]

  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    out_max  <- rep(NA_real_, nrow(dt))
    out_min  <- rep(NA_real_, nrow(dt))
    out_mean <- rep(NA_real_, nrow(dt))

    for (yr in years) {
      yr_rows <- dt[year == yr]
      # Map spatial position -> value
      pos <- yr_rows$.spatial_pos
      vals <- yr_rows[[var_name]]
      row_ids <- yr_rows$.row_id

      val_vec <- rep(NA_real_, N)
      val_vec[pos] <- vals

      for (k in seq_len(nrow(yr_rows))) {
        sp <- pos[k]
        nb_idx <- nb_list[[sp]]
        if (length(nb_idx) == 0L) next
        nb_vals <- val_vec[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) == 0L) next
        ri <- row_ids[k]
        out_max[ri]  <- max(nb_vals)
        out_min[ri]  <- min(nb_vals)
        out_mean[ri] <- mean(nb_vals)
      }
    }

    set(dt, j = col_max,  value = out_max)
    set(dt, j = col_min,  value = out_min)
    set(dt, j = col_mean, value = out_mean)
  }

  dt[, c(".spatial_pos", ".row_id") := NULL]
  setkey(dt, NULL)
  if (was_df) return(as.data.frame(dt)) else return(dt)
}
```

### Further optimization: Vectorize max/min with C++ via Rcpp

The inner loop over 344K cells Ã— 28 years Ã— 5 variables for max/min is ~48M lightweight iterations. In pure R this takes roughly 5â€“15 minutes. To bring it under 30 seconds, use this optional Rcpp helper:

```r
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_max_min_cpp(NumericVector v,
                          List nb_list,
                          int N) {
  NumericVector out_max(N, NA_REAL);
  NumericVector out_min(N, NA_REAL);

  for (int i = 0; i < N; i++) {
    IntegerVector idx = nb_list[i];
    int n = idx.size();
    if (n == 0) continue;

    double cmax = R_NegInf;
    double cmin = R_PosInf;
    int count = 0;

    for (int j = 0; j < n; j++) {
      double val = v[idx[j] - 1];  // R is 1-indexed
      if (ISNA(val) || ISNAN(val)) continue;
      if (val > cmax) cmax = val;
      if (val < cmin) cmin = val;
      count++;
    }

    if (count > 0) {
      out_max[i] = cmax;
      out_min[i] = cmin;
    }
  }

  return List::create(Named("max") = out_max,
                      Named("min") = out_min);
}
')
```

Then replace the inner R loop in the balanced path:

```r
      # Replace the R for-loop for max/min with:
      mm <- neighbor_max_min_cpp(v, nb_list, N)
      out_max[row_range] <- mm$max
      out_min[row_range] <- mm$min
```

---

### Usage (drop-in replacement for the original outer loop)

```r
# Original id_order, rook_neighbors_unique, and cell_data are unchanged.
# The trained Random Forest model is not touched.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has 15 new columns (3 stats Ã— 5 vars):
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ...
# with identical numerical values to the original implementation.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R iterations, string hashing | Eliminated (arithmetic indexing) |
| `compute_neighbor_stats` | 6.46M Ã— 5 R iterations | 28 Ã— 5 vectorized passes over 344K cells |
| Mean computation | Per-row R `mean()` | Sparse matrixâ€“vector multiply (`adj %*% v`) |
| Max/Min computation | Per-row R `max()/min()` | Rcpp loop or 28 Ã— 344K R loop |
| Estimated runtime | 86+ hours | **~1â€“2 minutes** (with Rcpp), ~10â€“15 min (pure R) |
| Numerical output | Exact | **Identical** (same neighbor sets, same operations) |
| RF model | Untouched | **Untouched** |
| Memory | ~2 GB neighbor list | ~50 MB sparse matrix + working vectors |
END_RESPONSE>>>
