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
run_id: Anthropic_F10_T00_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector â€” this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a pure-R `lapply`**, extracting subsets of a numeric vector and computing `max/min/mean`. The per-element overhead of R function calls, subsetting, and `NA` removal dominates.

3. **The topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Rook neighbors are a spatial property â€” the same adjacency holds for every year. The current code re-resolves neighbor cell IDs into row indices per cell-year, which is redundant.

**Estimated complexity:** ~6.46M Ã— (string operations + named vector lookups + R-level aggregation) â‰ˆ 86+ hours.

## Optimization Strategy

1. **Build the sparse adjacency matrix once (344K Ã— 344K).** Convert the `nb` object to a sparse `dgCMatrix` using `spdep::nb2Matrix` or manual construction. This encodes the ~1.37M directed edges.

2. **Reshape each variable into a dense 344,208 Ã— 28 matrix (cells Ã— years).** This allows column-wise (per-year) sparse matrixâ€“vector multiplication.

3. **Compute neighbor sums and counts via sparse matrix multiplication.** For `mean`: `A %*% X` gives neighbor sums; a count vector `A %*% (non-NA indicator)` gives the denominator. For `max` and `min`: use a single pass over the CSC/CSR structure in C++ via `Rcpp`.

4. **Vectorize max/min via Rcpp.** Sparse matrix multiplication naturally gives sum (and thus mean), but max/min require explicit iteration over neighbor entries. A small Rcpp function over the sparse matrix columns handles this in seconds.

5. **Reshape results back to long format and column-bind.** This preserves the original data layout and numerical equivalence.

**Expected speedup:** From 86+ hours to ~2â€“5 minutes. Memory: the sparse matrix is ~22 MB (1.37M entries); each dense matrix is ~77 MB (344K Ã— 28 doubles). Total peak well under 16 GB.

## Working R Code

```r
# =============================================================================
# Optimized neighbor-feature pipeline
# Preserves numerical equivalence with the original max / min / mean statistics
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# ---- 1. Rcpp helper: sparse neighbor max, min, mean per column -----------

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// A is a dgCMatrix (CSC format), X is a dense matrix (n_nodes x n_years)
// Returns a list of three matrices: max, min, mean  (same dims as X)
// [[Rcpp::export]]
List sparse_neighbor_stats(S4 A, NumericMatrix X) {
  // CSC components
  IntegerVector Ap = A.slot("p");   // column pointers (length n+1)
  IntegerVector Ai = A.slot("i");   // row indices
  // Note: dgCMatrix is column-compressed. For row i, its neighbors j are
  // found where Ai == i within each column. But we need neighbors OF row i,
  // i.e., columns j where A[i,j] = 1.  For CSC that means we need the
  // transpose: cols of A^T = rows of A.  So we transpose A first.
  // Actually, we receive A as the adjacency matrix where A[i,j]=1 means
  // j is a neighbor of i (i.e., row i has entries in columns = neighbors).
  // In CSC, iterating by column j gives us all rows i that have A[i,j]=1.
  // We want: for each row i, gather X[j,] for all j where A[i,j]=1.
  // Strategy: transpose A to get At where At[j,i]=1, then At in CSC
  // lets us iterate column i to get all j.  But transposing in C++ is
  // complex.  Instead, we require the caller to pass t(A).
  // So: A_input is t(adj), CSC.  Column i of A_input lists the neighbors of i.

  int n = X.nrow();
  int T = X.ncol();

  NumericMatrix out_max(n, T);
  NumericMatrix out_min(n, T);
  NumericMatrix out_mean(n, T);

  for (int i = 0; i < n; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    int degree = end - start;

    if (degree == 0) {
      for (int t = 0; t < T; t++) {
        out_max(i, t)  = NA_REAL;
        out_min(i, t)  = NA_REAL;
        out_mean(i, t) = NA_REAL;
      }
      continue;
    }

    for (int t = 0; t < T; t++) {
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int    cnt  = 0;

      for (int p = start; p < end; p++) {
        int j = Ai[p];          // neighbor index
        double val = X(j, t);
        if (!R_IsNA(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          vsum += val;
          cnt++;
        }
      }

      if (cnt == 0) {
        out_max(i, t)  = NA_REAL;
        out_min(i, t)  = NA_REAL;
        out_mean(i, t) = NA_REAL;
      } else {
        out_max(i, t)  = vmax;
        out_min(i, t)  = vmin;
        out_mean(i, t) = vsum / cnt;
      }
    }
  }

  return List::create(
    Named("nb_max")  = out_max,
    Named("nb_min")  = out_min,
    Named("nb_mean") = out_mean
  );
}
')

# ---- 2. Build sparse adjacency matrix from nb object (once) ---------------

build_adjacency_matrix <- function(nb_obj) {
  # nb_obj: list of length n_nodes; nb_obj[[i]] = integer vector of neighbor
  # indices (1-based), with class "nb".  0L means no neighbors.
  n <- length(nb_obj)
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel

  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  # A[i,j] = 1 means j is a neighbor of i

  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(A)
}

# ---- 3. Reshape long panel to node Ã— year matrix --------------------------

long_to_matrix <- function(dt, var_name, id_idx, year_idx) {
  # dt:       data.table with columns id, year, <var_name>
  # id_idx:   named integer vector: id_idx[as.character(cell_id)] -> row position
  # year_idx: named integer vector: year_idx[as.character(year)] -> col position
  # Returns:  numeric matrix  n_nodes x n_years
  n <- length(id_idx)
  T_ <- length(year_idx)
  mat <- matrix(NA_real_, nrow = n, ncol = T_)
  ri <- id_idx[as.character(dt$id)]
  ci <- year_idx[as.character(dt$year)]
  mat[cbind(ri, ci)] <- dt[[var_name]]
  return(mat)
}

# ---- 4. Reshape node Ã— year matrix back to long column --------------------

matrix_to_long_col <- function(mat, id_idx, year_idx, n_rows, dt_id, dt_year) {
  # Reverse of long_to_matrix: extract values aligned to the original row order
  ri <- id_idx[as.character(dt_id)]
  ci <- year_idx[as.character(dt_year)]
  mat[cbind(ri, ci)]
}

# ---- 5. Main pipeline -----------------------------------------------------

run_neighbor_feature_pipeline <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table for speed (non-destructive)
  dt <- as.data.table(cell_data)

  # --- Build adjacency matrix once ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique)
  # Transpose for CSC column-iteration = neighbor gathering
  At <- t(A)
  # Force to dgCMatrix (CSC) for the Rcpp function

  At <- as(At, "dgCMatrix")

  # --- Index maps ---
  unique_ids   <- id_order                          # length 344,208
  unique_years <- sort(unique(dt$year))             # 1992:2019, length 28
  id_idx   <- setNames(seq_along(unique_ids),   as.character(unique_ids))
  year_idx <- setNames(seq_along(unique_years), as.character(unique_years))

  # --- Loop over source variables ---
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))

    # Reshape to matrix
    X <- long_to_matrix(dt, var_name, id_idx, year_idx)

    # Compute neighbor max, min, mean via Rcpp sparse aggregation
    stats <- sparse_neighbor_stats(At, X)

    # Map back to long-format columns aligned with dt row order
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := matrix_to_long_col(stats$nb_max,  id_idx, year_idx,
                                           .N, id, year)]
    dt[, (min_col)  := matrix_to_long_col(stats$nb_min,  id_idx, year_idx,
                                           .N, id, year)]
    dt[, (mean_col) := matrix_to_long_col(stats$nb_mean, id_idx, year_idx,
                                           .N, id, year)]

    cat(sprintf("  -> Added %s, %s, %s\n", max_col, min_col, mean_col))
  }

  cat("Neighbor feature pipeline complete.\n")
  return(dt)
}

# ---- 6. Execute and predict ------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_feature_pipeline(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Predict with the pre-trained Random Forest (no retraining)
# rf_model is already loaded in the environment
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Summary of Guarantees

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The Rcpp kernel computes `max`, `min`, `sum/count` over exactly the same neighbor sets with the same `NA` handling (`!is.na` filter, return `NA` when all neighbors missing or degree-0). Results are bit-identical to the original R code. |
| **Trained RF preserved** | `rf_model` is never modified or retrained â€” only `predict()` is called. |
| **Memory** | Sparse matrix: ~22 MB. Five 344KÃ—28 dense matrices: ~385 MB. data.table overhead: ~5.7 GB for 6.46M Ã— 110 cols. Peak â‰ˆ 8â€“10 GB, fits in 16 GB. |
| **Speed** | Adjacency build: ~1 s. Each variable: ~2â€“5 s in Rcpp (6.46M nodes Ã— ~4 neighbors Ã— 28 years). Total: **~1â€“3 minutes** vs. 86+ hours. |
| **Topology reuse** | The sparse matrix `At` is built once and reused for all 5 variables and all 28 years â€” no per-year or per-cell string operations. |
END_RESPONSE>>>
