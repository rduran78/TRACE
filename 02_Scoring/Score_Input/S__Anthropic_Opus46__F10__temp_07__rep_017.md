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
run_id: Anthropic_F10_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`paste` + `idx_lookup[neighbor_keys]`). This is O(NÃ—K) with enormous constant factors from R's string operations and named-vector indexing (hash lookups on character keys). With ~6.46M rows and an average of ~4 rook neighbors per cell, this is ~25.8M string constructions and hash lookups.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a numeric vector, removing NAs, and computing three summary statistics. This is called 5 times (once per source variable), so ~32.3M R-level function calls with per-element allocation overhead.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** Rook contiguity depends only on spatial position, not time. The current code redundantly encodes the same spatial adjacency structure for every year by embedding year into the lookup keys, inflating the problem from 344,208 spatial edges to 6.46M row-level entries.

**Root cause:** The implementation treats the problem as a flat row-level operation instead of exploiting the panel structure (spatial topology Ã— time). The graph has 344,208 nodes with ~1.37M directed edges â€” this topology is **identical across all 28 years**. The code should build the spatial adjacency once and apply it per-year via vectorized sparse matrix operations.

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from `rook_neighbors_unique` (344,208 Ã— 344,208, ~1.37M nonzeros). This is a `dgCMatrix` from the `Matrix` package.

2. **For each variable and each year**, extract the variable vector for that year's cells (length 344,208), then compute neighbor max/min/mean via sparse matrix operations:
   - **Mean**: `A %*% x / degree` (one sparse matrix-vector multiply).
   - **Max/Min**: Use a modified sparse matrix approach â€” replace structural zeros with `-Inf`/`Inf` and use `rowMaxs`/`rowMins` from the `matrixStats`-compatible sparse operations, or iterate over the CSC/CSR structure in C++ via `Rcpp`.

3. **Vectorize across years** by reshaping each variable into a 344,208 Ã— 28 matrix and performing sparse matrix Ã— dense matrix multiplication for the mean. For max/min, use an `Rcpp` kernel that walks the sparse adjacency structure.

4. **Memory**: The sparse matrix is ~1.37M entries Ã— 12 bytes â‰ˆ 16 MB. Each dense matrix is 344,208 Ã— 28 Ã— 8 bytes â‰ˆ 77 MB. Total working memory well under 4 GB.

5. **Preserve the trained Random Forest model**: No retraining. We only reconstruct the identical predictor columns with the same names and semantics.

6. **Numerical equivalence**: Sparse matrix multiplication for mean is algebraically identical. The Rcpp kernel for max/min uses the same `max`/`min` over the same neighbor sets. Cells with zero neighbors produce `NA`, matching the original.

## Optimized R Code

```r
# ==============================================================================
# Optimized spatial neighbor feature computation
# Sparse graph topology Ã— panel data, vectorized across years
# ==============================================================================

library(Matrix)
library(Rcpp)
library(data.table)

# ------------------------------------------------------------------------------
# Step 1: Build sparse adjacency matrix from spdep nb object (done ONCE)
# ------------------------------------------------------------------------------

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: list of integer vectors (neighbor indices), length n
  # Returns: n x n sparse dgCMatrix with 1s at neighbor positions
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove the 0-neighbor sentinel that spdep uses
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n), repr = "C")
}

# ------------------------------------------------------------------------------
# Step 2: Rcpp kernel for sparse neighbor max and min across columns
# ------------------------------------------------------------------------------

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(
    IntegerVector Ap,    // CSR row pointers (length n+1, 0-based)
    IntegerVector Aj,    // CSR column indices (0-based)
    NumericMatrix X,     // n x T matrix of values
    IntegerVector degree // row degrees
) {
  int n = X.nrow();
  int T = X.ncol();
  NumericMatrix maxMat(n, T);
  NumericMatrix minMat(n, T);

  for (int i = 0; i < n; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    int deg   = end - start;

    if (deg == 0) {
      for (int t = 0; t < T; t++) {
        maxMat(i, t) = NA_REAL;
        minMat(i, t) = NA_REAL;
      }
      continue;
    }

    for (int t = 0; t < T; t++) {
      double mx = R_NegInf;
      double mn = R_PosInf;
      int valid_count = 0;

      for (int p = start; p < end; p++) {
        int j = Aj[p];
        double v = X(j, t);
        if (!R_IsNA(v) && !ISNAN(v)) {
          if (v > mx) mx = v;
          if (v < mn) mn = v;
          valid_count++;
        }
      }

      if (valid_count == 0) {
        maxMat(i, t) = NA_REAL;
        minMat(i, t) = NA_REAL;
      } else {
        maxMat(i, t) = mx;
        minMat(i, t) = mn;
      }
    }
  }

  return List::create(Named("max") = maxMat, Named("min") = minMat);
}
')

# Also handle NA-aware mean via Rcpp for full consistency
cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_neighbor_mean(
    IntegerVector Ap,
    IntegerVector Aj,
    NumericMatrix X
) {
  int n = X.nrow();
  int T = X.ncol();
  NumericMatrix meanMat(n, T);

  for (int i = 0; i < n; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    int deg   = end - start;

    if (deg == 0) {
      for (int t = 0; t < T; t++) {
        meanMat(i, t) = NA_REAL;
      }
      continue;
    }

    for (int t = 0; t < T; t++) {
      double sum = 0.0;
      int valid_count = 0;

      for (int p = start; p < end; p++) {
        int j = Aj[p];
        double v = X(j, t);
        if (!R_IsNA(v) && !ISNAN(v)) {
          sum += v;
          valid_count++;
        }
      }

      if (valid_count == 0) {
        meanMat(i, t) = NA_REAL;
      } else {
        meanMat(i, t) = sum / valid_count;
      }
    }
  }

  return meanMat;
}
')

# ------------------------------------------------------------------------------
# Step 3: Convert dgCMatrix (CSC) to CSR format for row-wise traversal
# ------------------------------------------------------------------------------

csc_to_csr <- function(A) {
  # Transpose CSC gives CSR representation
  At <- t(A)  # dgCMatrix transpose
  list(
    Ap = At@p,   # row pointers (0-based, length n+1)
    Aj = At@i,   # column indices (0-based)
    Ax = At@x    # values (all 1s for adjacency)
  )
}

# ------------------------------------------------------------------------------
# Step 4: Main pipeline
# ------------------------------------------------------------------------------

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  # --- Convert to data.table for fast manipulation ---
  dt <- as.data.table(cell_data)
  
  # --- Ensure consistent ordering: cells within each year in id_order order ---
  # Create cell index mapping
  n_cells <- length(id_order)
  id_to_idx <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Get sorted unique years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))
  
  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(dt)))
  
  # --- Build sparse adjacency matrix (ONCE) ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  csr <- csc_to_csr(A)
  degree <- diff(csr$Ap)
  cat(sprintf("Adjacency: %d nonzeros, avg degree: %.2f\n", length(csr$Aj), mean(degree)))
  
  # --- Map each row to (cell_index, year_index) ---
  dt[, cell_idx := id_to_idx[as.character(id)]]
  dt[, year_idx := year_to_col[as.character(year)]]
  
  # Verify complete panel or handle sparse panel
  # Create a linear index for fast matrix population
  dt[, lin_idx := cell_idx + (year_idx - 1L) * n_cells]
  
  # --- Process each neighbor source variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # Reshape variable into n_cells x n_years matrix
    X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    X[dt$lin_idx] <- dt[[var_name]]
    
    # Compute neighbor stats via Rcpp (walks CSR structure)
    maxmin <- sparse_neighbor_maxmin(csr$Ap, csr$Aj, X, as.integer(degree))
    mean_mat <- sparse_neighbor_mean(csr$Ap, csr$Aj, X)
    
    # Extract results back to data.table rows
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := maxmin$max[lin_idx]]
    dt[, (min_col)  := maxmin$min[lin_idx]]
    dt[, (mean_col) := mean_mat[lin_idx]]
    
    cat(sprintf("  Done: %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  # --- Clean up helper columns ---
  dt[, c("cell_idx", "year_idx", "lin_idx") := NULL]
  
  # --- Return as data.frame to match original pipeline expectations ---
  as.data.frame(dt)
}

# ------------------------------------------------------------------------------
# Step 5: Execute and predict
# ------------------------------------------------------------------------------

# Run the optimized pipeline
cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)

# Apply the pre-trained Random Forest (no retraining)
# rf_model is already loaded in the environment
cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string pastes + hash lookups | One sparse matrix build (344K Ã— 344K, 1.37M nnz) |
| **Stats computation** | 5 Ã— 6.46M R-level `lapply` iterations | 5 Ã— Rcpp CSR traversal over 344K Ã— 28 matrix |
| **Memory (lookup)** | ~6.46M-element nested list (~2-4 GB) | ~33 MB sparse matrix (CSR) |
| **Time complexity** | O(N_rows Ã— K Ã— string_ops) | O(N_cells Ã— K Ã— T) with C++ inner loop |
| **Estimated runtime** | 86+ hours | **~2-5 minutes** |
| **Numerical result** | Reference | Identical (same neighbor sets, same max/min/mean, same NA handling) |
| **RF model** | Preserved | Preserved (predict only, no retraining) |

The key insight: the spatial graph topology is **year-invariant**. By factoring the problem into *topology* (sparse matrix, built once) Ã— *attributes* (dense matrix, cells Ã— years), we eliminate all redundant string operations and leverage cache-friendly C++ traversal of the CSR structure. The 5 neighbor variables produce 15 new columns (max/min/mean Ã— 5), numerically identical to the original.
END_RESPONSE>>>
