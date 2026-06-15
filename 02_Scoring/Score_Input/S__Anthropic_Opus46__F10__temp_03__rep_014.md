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
run_id: Anthropic_F10_T03_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a ~6.46M-element character vector â€” this alone is catastrophically slow.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in a serial `lapply`**, extracting subsets of a numeric vector. The per-element R-level overhead (function call, subsetting, `is.na` check, aggregation) dominates when repeated millions of times.

3. **The neighbor lookup is year-aware but the graph topology is year-invariant.** The 344,208 cells have the same rook neighbors every year. The code redundantly encodes 28 copies of the same adjacency structure (one per year-slice), inflating the lookup from ~1.37M edges to ~38.5M entries across all cell-years.

**Root cause summary:** The algorithm is O(N_rows Ã— avg_neighbors) with enormous R-interpreter overhead per operation, and the data structure design prevents vectorized/compiled-code execution.

## Optimization Strategy

1. **Build the sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208 sparse matrix). This is the graph topology â€” year-invariant.

2. **Reshape each variable into a 344,208 Ã— 28 dense matrix** (cells Ã— years). This enables column-wise (year-wise) sparse matrixâ€“vector operations.

3. **Compute neighbor aggregates via sparse matrix algebra:**
   - **Mean:** `A %*% X / degree` (where `A` is the binary adjacency matrix, `X` is the variable matrix, and `degree` is the row-sum vector).
   - **Max and Min:** Use a single pass in C++ via `Rcpp` over the sparse matrix CSR structure â€” unavoidable since max/min are not linear and can't be expressed as matrix multiplication.

4. **Avoid all string-key lookups, all `lapply` over millions of elements, and all year-level redundancy.**

5. **Memory:** Sparse matrix with ~1.37M non-zeros â‰ˆ 33 MB. Dense matrices 344,208 Ã— 28 â‰ˆ 77 MB each. Total for 5 variables Ã— 3 stats Ã— 77 MB â‰ˆ 3.2 GB peak â€” fits in 16 GB.

6. **Time:** Sparse matrixâ€“dense matrix multiply for mean: seconds. Rcpp loop for max/min over ~1.37M edges Ã— 28 years: seconds. Total: **minutes, not hours.**

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(Matrix)
library(Rcpp)
library(data.table)

# ---- Step 0: Compile the C++ workhorse for max/min (runs once) ----

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// Computes row-wise max and min of X[neighbors, ] using CSR sparse structure.
// p, j are 0-based CSR arrays from a dgRMatrix.
// X is n_cells x n_years matrix.
// Returns a list with two matrices: max_mat and min_mat (same dims as X).

// [[Rcpp::export]]
List sparse_row_maxmin(IntegerVector p, IntegerVector j,
                       NumericMatrix X) {
  int n = X.nrow();
  int nyears = X.ncol();
  NumericMatrix max_mat(n, nyears);
  NumericMatrix min_mat(n, nyears);

  // Initialize to NA
  double na_val = NA_REAL;
  std::fill(max_mat.begin(), max_mat.end(), na_val);
  std::fill(min_mat.begin(), min_mat.end(), na_val);

  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue;  // no neighbors -> stays NA

    for (int yr = 0; yr < nyears; yr++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int k = start; k < end; k++) {
        double val = X(j[k], yr);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(i, yr) = cur_max;
        min_mat(i, yr) = cur_min;
      }
      // else stays NA
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# Also need NA-aware mean via sparse ops. We handle NA by zeroing out NAs
# and tracking valid counts.

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_row_mean(IntegerVector p, IntegerVector j,
                              NumericMatrix X) {
  int n = X.nrow();
  int nyears = X.ncol();
  NumericMatrix mean_mat(n, nyears);

  double na_val = NA_REAL;
  std::fill(mean_mat.begin(), mean_mat.end(), na_val);

  for (int i = 0; i < n; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue;

    for (int yr = 0; yr < nyears; yr++) {
      double sum_val = 0.0;
      int valid = 0;

      for (int k = start; k < end; k++) {
        double val = X(j[k], yr);
        if (!R_IsNA(val)) {
          sum_val += val;
          valid++;
        }
      }

      if (valid > 0) {
        mean_mat(i, yr) = sum_val / (double)valid;
      }
    }
  }

  return mean_mat;
}
')


# =============================================================================
# Step 1: Build sparse adjacency matrix from nb object (once)
# =============================================================================

build_adjacency_csr <- function(nb_obj) {
  # nb_obj is a list of length n_cells; nb_obj[[i]] contains integer neighbor

  # indices (1-based). A zero-element vector or 0L means no neighbors.
  n <- length(nb_obj)

  # Build COO triplets
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  from_vec <- unlist(from_list, use.names = FALSE)
  to_vec   <- unlist(to_list, use.names = FALSE)

  # Create dgRMatrix (CSR) via dgTMatrix -> dgCMatrix -> dgRMatrix
  A <- sparseMatrix(i = from_vec, j = to_vec, x = 1,
                    dims = c(n, n), repr = "T")
  A <- as(as(A, "CsparseMatrix"), "RsparseMatrix")  # CSR format
  return(A)
}


# =============================================================================
# Step 2: Reshape panel data into cell Ã— year matrices
# =============================================================================

reshape_to_matrix <- function(dt, id_order, years, var_name) {
  # dt: data.table with columns id, year, <var_name>
  # id_order: integer vector of cell IDs in canonical order (length n_cells)
  # years: sorted integer vector of years
  # Returns: n_cells x n_years numeric matrix

  n_cells <- length(id_order)
  n_years <- length(years)

  id_map   <- setNames(seq_along(id_order), as.character(id_order))
  year_map <- setNames(seq_along(years), as.character(years))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_idx <- id_map[as.character(dt$id)]
  col_idx <- year_map[as.character(dt$year)]

  mat[cbind(row_idx, col_idx)] <- dt[[var_name]]
  return(mat)
}


# =============================================================================
# Step 3: Compute all neighbor features and write back to data
# =============================================================================

run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique) {

  # Convert to data.table for speed

  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  years   <- sort(unique(cell_data$year))
  n_cells <- length(id_order)
  n_years <- length(years)

  cat("Building CSR adjacency matrix...\n")
  A_csr <- build_adjacency_csr(rook_neighbors_unique)
  # Extract CSR components (0-based for C++)
  csr_p <- A_csr@p        # length n_cells + 1, 0-based row pointers
  csr_j <- A_csr@j        # 0-based column indices

  # Precompute row/col index mapping for writing results back
  id_map   <- setNames(seq_along(id_order), as.character(id_order))
  year_map <- setNames(seq_along(years), as.character(years))
  row_idx  <- id_map[as.character(cell_data$id)]
  col_idx  <- year_map[as.character(cell_data$year)]
  lin_idx  <- (col_idx - 1L) * n_cells + row_idx  # linear index into matrix

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))

    # Reshape to matrix
    X <- reshape_to_matrix(cell_data, id_order, years, var_name)

    # Compute max, min via C++
    maxmin <- sparse_row_maxmin(csr_p, csr_j, X)
    max_mat <- maxmin$max_mat  # n_cells x n_years
    min_mat <- maxmin$min_mat

    # Compute mean via C++
    mean_mat <- sparse_row_mean(csr_p, csr_j, X)

    # Write results back to cell_data using linear indexing
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    set(cell_data, j = max_col,  value = max_mat[lin_idx])
    set(cell_data, j = min_col,  value = min_mat[lin_idx])
    set(cell_data, j = mean_col, value = mean_mat[lin_idx])

    # Free memory
    rm(X, max_mat, min_mat, mean_mat, maxmin)
    gc()
  }

  return(cell_data)
}


# =============================================================================
# Step 4: Execute and predict
# =============================================================================

# --- Run the optimized pipeline ---
cell_data <- run_neighbor_aggregation(cell_data, id_order, rook_neighbors_unique)

# --- Apply the pre-trained Random Forest (unchanged) ---
# rf_model is already loaded; do NOT retrain.
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

## Why This Is Correct and Numerically Equivalent

| Original operation | Optimized equivalent | Equivalence guarantee |
|---|---|---|
| `build_neighbor_lookup` finds row indices of neighbors sharing the same year | CSR row `i` contains column indices of spatial neighbors; year dimension is handled by the matrix column | Same neighbor set per cell-year: topology is year-invariant, so `X[neighbor, year_col]` retrieves exactly the same values |
| `max(neighbor_vals[!is.na(...)])` | `sparse_row_maxmin` skips `NA` values identically | Exact same `max` over same non-NA values |
| `min(neighbor_vals[!is.na(...)])` | Same C++ function | Identical |
| `mean(neighbor_vals[!is.na(...)])` | `sparse_row_mean` sums non-NA values and divides by count | Identical to R's `mean()` on the same non-NA subset |
| Returns `NA` when no valid neighbors | Both C++ functions return `NA` when `valid == 0` or no neighbors exist | Identical |

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (string ops on 6.46M rows) | ~2 seconds (CSR construction from nb object) |
| Per-variable aggregation | ~17 hours Ã— 5 vars | ~5â€“15 seconds Ã— 5 vars (C++ over 1.37M edges Ã— 28 years) |
| **Total** | **86+ hours** | **< 5 minutes** |
| Peak RAM | Unbounded list growth | ~4â€“5 GB (fits 16 GB) |

The speedup factor is approximately **1,000Ã—**, achieved by eliminating R-interpreter overhead via compiled C++ loops over a compact CSR sparse structure, and by separating the time-invariant graph topology from the time-varying node attributes.
END_RESPONSE>>>
