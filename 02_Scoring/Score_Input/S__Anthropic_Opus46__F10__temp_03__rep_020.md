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
run_id: Anthropic_F10_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M names â€” this alone is catastrophically slow. The function does ~6.46M Ã— avg_neighbors hash lookups against a 6.46M-length named vector.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting variable values and computing `max/min/mean` in pure R. This is repeated 5 times (once per source variable), totaling ~32.3 million R-level function calls.

3. **The neighbor lookup is year-aware but redundant**: rook neighbors are a *spatial* relationship that is identical across all 28 years. The current code rebuilds year-specific row indices by pasting strings, but the topology is static. The lookup should be built once at the spatial level and then broadcast across years via vectorized indexing.

**Root cause summary**: The 86+ hour runtime comes from (a) O(nÂ²)-like named-vector lookups during graph construction, and (b) millions of interpreted R loop iterations during aggregation. Both can be replaced with vectorized sparse-matrix operations.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** (344,208 Ã— 344,208) from the `nb` object â€” ~1.37M non-zero entries. This is tiny in memory (~16 MB as a `dgCMatrix`).

2. **Reshape each variable into a matrix** of shape (344,208 cells Ã— 28 years). Each column is one year.

3. **Compute neighbor aggregates via sparse matrixâ€“dense matrix multiplication and element-wise operations**:
   - **Mean**: `A %*% X / degree` (where `degree` = number of neighbors per node).
   - **Max / Min**: Use a single pass over the sparse matrix's CSC structure in C++ (via `Rcpp`) or use `{Matrix}` tricks with repeated sparse operations.

4. **Flatten back** to the original long-format data frame and bind columns.

This reduces the entire pipeline from ~86 hours to **minutes** (sparse matrix ops on matrices of this size are near-instantaneous).

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean neighbor stats.
# =============================================================================

library(Matrix)
library(Rcpp)

# ---- Step 0: One-time C++ helper for sparse row-wise max/min ----
# We compile a small Rcpp function that, given a CSC sparse matrix A and a
# dense matrix X, computes row-wise max, min, and sum of neighbor values.
# This avoids any R-level loops over 6.46M rows.

sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_stats(
    IntegerVector Ap,    // CSC column pointers (length ncol+1)
    IntegerVector Ai,    // CSC row indices
    NumericMatrix X,     // dense matrix: nrow = n_cells, ncol = n_years
    IntegerVector degree // number of neighbors per row (length n_cells)
) {
  int n = X.nrow();
  int T = X.ncol();

  NumericMatrix out_max(n, T);
  NumericMatrix out_min(n, T);
  NumericMatrix out_mean(n, T);

  // Initialize max to -Inf, min to +Inf, sum to 0
  double posInf = R_PosInf;
  double negInf = R_NegInf;
  for (int i = 0; i < n; i++) {
    for (int t = 0; t < T; t++) {
      out_max(i, t) = negInf;
      out_min(i, t) = posInf;
      out_mean(i, t) = 0.0;
    }
  }

  // CSC traversal: for each column j (= neighbor source node),
  // iterate over rows i that have an edge from j -> i (i.e., j is neighbor of i).
  // Accumulate stats for row i using X[j, t].
  int ncol_A = Ap.size() - 1;
  for (int j = 0; j < ncol_A; j++) {
    for (int ptr = Ap[j]; ptr < Ap[j + 1]; ptr++) {
      int i = Ai[ptr];  // row i has neighbor j
      for (int t = 0; t < T; t++) {
        double val = X(j, t);
        if (NumericMatrix::is_na(val)) continue;
        if (val > out_max(i, t)) out_max(i, t) = val;
        if (val < out_min(i, t)) out_min(i, t) = val;
        out_mean(i, t) += val;
      }
    }
  }

  // Finalize: replace sentinel values with NA; compute mean = sum / count
  // We need non-NA neighbor counts per (i, t). For simplicity and speed,
  // if all neighbor values for a row are NA, degree effectively = 0.
  // We track valid counts via a second pass or by noting that
  // if out_max is still -Inf, no valid neighbor was found.
  // For mean, we need valid counts. We do a second sparse pass for NA counting.

  // Count valid (non-NA) neighbors per (i, t)
  IntegerMatrix valid_count(n, T);
  for (int j = 0; j < ncol_A; j++) {
    for (int ptr = Ap[j]; ptr < Ap[j + 1]; ptr++) {
      int i = Ai[ptr];
      for (int t = 0; t < T; t++) {
        double val = X(j, t);
        if (!NumericMatrix::is_na(val)) {
          valid_count(i, t) += 1;
        }
      }
    }
  }

  for (int i = 0; i < n; i++) {
    for (int t = 0; t < T; t++) {
      if (valid_count(i, t) == 0) {
        out_max(i, t) = NA_REAL;
        out_min(i, t) = NA_REAL;
        out_mean(i, t) = NA_REAL;
      } else {
        out_mean(i, t) = out_mean(i, t) / valid_count(i, t);
      }
    }
  }

  return List::create(
    Named("max")  = out_max,
    Named("min")  = out_min,
    Named("mean") = out_mean
  );
}
')

# ---- Step 1: Build sparse adjacency matrix from nb object (once) ----
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of length n, each element is integer vector of neighbor indices
  # Builds a sparse n x n matrix A where A[i,j] = 1 means j is a neighbor of i
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove 0-entries (spdep uses 0 to indicate no neighbors)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

# ---- Step 2: Ensure data is sorted by (id, year) and build index maps ----
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   neighbor_source_vars, rf_model) {

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  cat("Building spatial adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  # Convert to dgCMatrix (CSC) for our Rcpp function
  A <- as(A, "dgCMatrix")

  # Build mapping: cell id -> matrix row index (1..n_cells)
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  # Build mapping: year -> matrix column index (1..n_years)
  year_to_col <- setNames(seq_along(years), as.character(years))

  # Map each row of cell_data to (cell_row, year_col)
  cat("Mapping data to cell x year matrix indices...\n")
  cell_row_idx <- id_to_row[as.character(cell_data$id)]
  year_col_idx <- year_to_col[as.character(cell_data$year)]
  # Linear index into n_cells x n_years matrix (column-major)
  lin_idx <- cell_row_idx + (year_col_idx - 1L) * n_cells

  # ---- Step 3: For each variable, reshape to matrix, compute stats, reshape back ----
  cat("Computing neighbor statistics for", length(neighbor_source_vars), "variables...\n")

  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")

    # Reshape to n_cells x n_years matrix
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[lin_idx] <- cell_data[[var_name]]

    # Compute neighbor max, min, mean via sparse aggregation
    stats <- sparse_neighbor_stats(A@p, A@i, X, diff(A@p))

    # Extract results back to long format using the same linear index
    max_vals  <- stats$max[lin_idx]
    min_vals  <- stats$min[lin_idx]
    mean_vals <- stats$mean[lin_idx]

    # Add columns to cell_data (matching original naming convention)
    cell_data[[paste0("neighbor_max_",  var_name)]] <- max_vals
    cell_data[[paste0("neighbor_min_",  var_name)]] <- min_vals
    cell_data[[paste0("neighbor_mean_", var_name)]] <- mean_vals
  }

  cat("Neighbor feature engineering complete.\n")

  # ---- Step 4: Predict with pre-trained Random Forest (no retraining) ----
  cat("Generating predictions with pre-trained Random Forest model...\n")
  cell_data$prediction <- predict(rf_model, newdata = cell_data)

  return(cell_data)
}

# ---- Usage ----
# cell_data <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   rf_model               = trained_rf_model
# )
```

## Performance Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup construction** | ~6.46M string-paste + named-vector lookups | One-time sparse matrix construction from `nb` object (~0.5 sec) |
| **Aggregation per variable** | 6.46M `lapply` iterations in R | Single C++ pass over ~1.37M sparse entries Ã— 28 year columns |
| **Total aggregation passes** | 5 variables Ã— 6.46M R calls = 32.3M calls | 5 Ã— 1 sparse-matrix traversal in compiled C++ |
| **Memory** | 6.46M-element list of integer vectors + 6.46M named character vector | ~16 MB sparse matrix + 5 Ã— (344K Ã— 28) dense matrices (~39 MB each) |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Numerical equivalence** | Baseline | âœ… Identical max/min/mean (same NA handling, same arithmetic) |
| **RF model** | Unchanged | âœ… Predict-only, no retraining |

### Why numerical equivalence is preserved

- The sparse matrix `A[i,j] = 1` encodes exactly the same rook-neighbor relationships as the original `nb` object.
- For each cell `i` and year `t`, the C++ code collects `X[j, t]` for all `j` where `A[i,j] = 1`, skips `NA` values, and computes `max`, `min`, and `sum/count` â€” identical operations to the original `compute_neighbor_stats`.
- The `valid_count` tracking ensures the mean denominator matches the original's `length(neighbor_vals[!is.na(neighbor_vals)])`.
- Cells with zero valid neighbors produce `NA` for all three statistics, matching the original `c(NA, NA, NA)` return.
END_RESPONSE>>>
