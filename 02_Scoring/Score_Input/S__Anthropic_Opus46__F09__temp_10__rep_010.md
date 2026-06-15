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
run_id: Anthropic_F09_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F09_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) for three compounding reasons:

### 1. `build_neighbor_lookup` is O(NÂ²)-class in practice
It iterates over **all 6.46 million cell-year rows** with `lapply`, and for each row it:
- Performs character-based hash lookups (`paste`, `setNames`, named-vector indexing).
- Rebuilds neighbor keys **per row per year**, even though the **neighbor topology is identical across all 28 years**. This means the same spatial neighbor relationships are redundantly resolved ~28 times per cell, totaling ~6.46M iterations with string operations.

### 2. `compute_neighbor_stats` iterates row-wise over 6.46M rows (Ã—5 variables)
Each call to `lapply` over the lookup table touches every row, subsetting and computing `max/min/mean` in pure R. That's ~32.3M R-level function calls across the 5 variables.

### 3. The neighbor lookup itself is a list of 6.46M integer vectors
This is a massive R list object, consuming significant RAM and preventing vectorized operations.

**Root cause summary:** The spatial topology (which cell neighbors which) is **time-invariant**, but the code entangles it with the time dimension, blowing up a ~344K-cell spatial problem into a ~6.46M-row problem at the lookup-construction stage.

---

## Optimization Strategy

**Core idea:** Separate the time-invariant spatial adjacency from the time-varying attributes. Build a sparse adjacency matrix **once** (344K Ã— 344K), then use sparse matrixâ€“dense matrix multiplication to compute neighbor sums and counts, from which max/min/mean are derived efficiently.

Specifically:

| Step | What | Complexity |
|------|------|------------|
| 1 | Convert `spdep::nb` â†’ sparse binary adjacency matrix **W** (344K Ã— 344K). Done once. | O(cells + edges) â‰ˆ seconds |
| 2 | For each variable, reshape the 6.46M-row column into a 344K Ã— 28 matrix (cells Ã— years). | O(N) |
| 3 | Compute `neighbor_mean = W %*% value_matrix / W %*% ones_matrix` via sparse matrix multiplication. | O(edges Ã— years) â‰ˆ seconds per variable |
| 4 | For `max` and `min`, iterate over cells (not cell-years) using the `nb` object â€” only 344K iterations, each touching ~4 neighbors Ã— 28 years. | O(cells Ã— neighbors Ã— years) |
| 5 | Reshape results back to the long panel and column-bind onto `cell_data`. | O(N) |

**Expected speedup:** From ~86 hours to **< 5 minutes** on a standard laptop.

**Preservation guarantees:**
- The trained Random Forest model is not retouched.
- The numerical outputs (neighbor max, min, mean) are identical to the original implementation.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ===========================================================================
# STEP 0: Convert cell_data to data.table for fast manipulation
# ===========================================================================
cell_dt <- as.data.table(cell_data)

# Ensure stable ordering: cells (id) Ã— years
#   id_order  = vector of 344,208 unique cell IDs in the spdep::nb index order
#   years     = 1992:2019
years <- sort(unique(cell_dt$year))
n_cells <- length(id_order)
n_years <- length(years)

# Create integer indices for cells and years
cell_dt[, cell_idx := match(id, id_order)]
cell_dt[, year_idx := match(year, years)]

# Sort so that matrix filling is straightforward
setorder(cell_dt, cell_idx, year_idx)

# Verify dimensions
stopifnot(nrow(cell_dt) == n_cells * n_years)

# ===========================================================================
# STEP 1: Build sparse adjacency matrix W from spdep::nb object (ONCE)
# ===========================================================================
build_sparse_adjacency <- function(nb_obj) {
  # nb_obj is a list of length n_cells; nb_obj[[i]] gives neighbor indices
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)

  # Remove the 0-neighbor sentinel that spdep uses (integer(0) is fine, but

  # some nb objects store 0L for "no neighbours")
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]

  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

W <- build_sparse_adjacency(rook_neighbors_unique)   # 344208 Ã— 344208, ~1.37M non-zeros
neighbor_counts <- as.vector(W %*% rep(1, n_cells))  # number of neighbors per cell

# ===========================================================================
# STEP 2 & 3: For each variable, compute neighbor mean, max, min
# ===========================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Helper: reshape a long vector (already sorted by cell_idx, year_idx) into
#         a cells Ã— years matrix
to_matrix <- function(vec) {
  matrix(vec, nrow = n_cells, ncol = n_years, byrow = FALSE)
}

# Helper: reshape a cells Ã— years matrix back to a long vector
to_long <- function(mat) {
  as.vector(mat)   # column-major = cell_idx-major, matching our sort order
}

for (var_name in neighbor_source_vars) {

  cat("Processing neighbor features for:", var_name, "\n")

  # --- 2a. Reshape variable into matrix (cells Ã— years) ---
  val_mat <- to_matrix(cell_dt[[var_name]])

  # --- 2b. Neighbor MEAN via sparse matrix multiplication ---
  #     For each cell i and year t:
  #       neighbor_sum[i,t]  = sum over j in N(i) of val[j,t]
  #       neighbor_mean[i,t] = neighbor_sum[i,t] / |N(i)|
  #
  #     W %*% val_mat computes all neighbor sums simultaneously.

  neighbor_sum_mat <- as.matrix(W %*% val_mat)              # 344K Ã— 28

  # Handle NA propagation: count non-NA neighbors per cell per year
  not_na_mat      <- to_matrix(as.numeric(!is.na(cell_dt[[var_name]])))
  neighbor_nna_mat <- as.matrix(W %*% not_na_mat)           # count of non-NA neighbors

  # Replace val NAs with 0 for sum, then correct
  val_mat_0 <- val_mat
  val_mat_0[is.na(val_mat_0)] <- 0
  neighbor_sum_clean <- as.matrix(W %*% val_mat_0)

  neighbor_mean_mat <- neighbor_sum_clean / neighbor_nna_mat
  neighbor_mean_mat[neighbor_nna_mat == 0] <- NA_real_

  # --- 2c. Neighbor MAX and MIN ---
  #     These are not expressible as matrix products.  We iterate over
  #     344K cells (NOT 6.46M rows), which is fast.
  neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[i]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) next
    # Extract the neighbor sub-matrix: |N(i)| Ã— n_years
    nb_vals <- val_mat[nb_idx, , drop = FALSE]
    # Compute column-wise (year-wise) max and min, ignoring NAs
    neighbor_max_mat[i, ] <- apply(nb_vals, 2, max, na.rm = TRUE)
    neighbor_min_mat[i, ] <- apply(nb_vals, 2, min, na.rm = TRUE)
  }

  # Fix Inf/-Inf from all-NA columns
  neighbor_max_mat[is.infinite(neighbor_max_mat)] <- NA_real_
  neighbor_min_mat[is.infinite(neighbor_min_mat)] <- NA_real_

  # --- 2d. Write results back to cell_dt ---
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  cell_dt[, (col_max)  := to_long(neighbor_max_mat)]
  cell_dt[, (col_min)  := to_long(neighbor_min_mat)]
  cell_dt[, (col_mean) := to_long(neighbor_mean_mat)]

  cat("  Done:", col_max, col_min, col_mean, "\n")
}

# ===========================================================================
# STEP 4: Clean up helper columns, restore original row order if needed
# ===========================================================================
cell_dt[, c("cell_idx", "year_idx") := NULL]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# ===========================================================================
# STEP 5: Predict with the EXISTING trained Random Forest (unchanged)
# ===========================================================================
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Further Speedup for Max/Min (Optional)

The `for` loop over 344K cells with `apply` for max/min is already manageable (~1â€“3 minutes), but if you want it even faster, you can replace it with **Rcpp**:

```r
# Optional: Rcpp version for neighbor max/min
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix neighbor_max_cpp(NumericMatrix val_mat, List nb_list) {
  int n_cells = val_mat.nrow();
  int n_years = val_mat.ncol();
  NumericMatrix out(n_cells, n_years);
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    IntegerVector nb = nb_list[i];
    int nn = nb.size();
    if (nn == 0) continue;
    for (int t = 0; t < n_years; t++) {
      double mx = R_NegInf;
      bool found = false;
      for (int k = 0; k < nn; k++) {
        int j = nb[k] - 1;  // 0-indexed
        if (j < 0) continue;
        double v = val_mat(j, t);
        if (!NumericVector::is_na(v)) {
          if (v > mx) mx = v;
          found = true;
        }
      }
      if (found) out(i, t) = mx;
    }
  }
  return out;
}

// [[Rcpp::export]]
NumericMatrix neighbor_min_cpp(NumericMatrix val_mat, List nb_list) {
  int n_cells = val_mat.nrow();
  int n_years = val_mat.ncol();
  NumericMatrix out(n_cells, n_years);
  std::fill(out.begin(), out.end(), NA_REAL);

  for (int i = 0; i < n_cells; i++) {
    IntegerVector nb = nb_list[i];
    int nn = nb.size();
    if (nn == 0) continue;
    for (int t = 0; t < n_years; t++) {
      double mn = R_PosInf;
      bool found = false;
      for (int k = 0; k < nn; k++) {
        int j = nb[k] - 1;
        if (j < 0) continue;
        double v = val_mat(j, t);
        if (!NumericVector::is_na(v)) {
          if (v < mn) mn = v;
          found = true;
        }
      }
      if (found) out(i, t) = mn;
    }
  }
  return out;
}
')

# Usage (replaces the R for-loop):
# neighbor_max_mat <- neighbor_max_cpp(val_mat, rook_neighbors_unique)
# neighbor_min_mat <- neighbor_min_cpp(val_mat, rook_neighbors_unique)
```

---

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Lookup construction | 6.46M string-paste iterations | One-time 344K sparse matrix (~2 sec) |
| Neighbor mean | 6.46M `lapply` calls Ã— 5 vars | Sparse matrix multiply: `W %*% M` (~seconds) |
| Neighbor max/min | 6.46M `lapply` calls Ã— 5 vars | 344K-iteration loop over cells (~1â€“3 min), or Rcpp (~10 sec) |
| Total time | ~86 hours | **~2â€“5 minutes** |
| Trained RF model | Preserved âœ“ | Preserved âœ“ |
| Numerical results | Identical âœ“ | Identical âœ“ |
| RAM peak | Huge list of 6.46M vectors | ~1.5 GB (sparse matrix + dense matrices) |
END_RESPONSE>>>
