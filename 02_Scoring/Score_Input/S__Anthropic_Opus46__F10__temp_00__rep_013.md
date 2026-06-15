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
run_id: Anthropic_F10_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_013.md

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

1. **Build the sparse adjacency structure once as a CSR (Compressed Sparse Row) matrix** using `Matrix::sparseMatrix`. The 344,208 cells and ~1.37M directed edges form a sparse binary adjacency matrix `A` (344,208 Ã— 344,208).

2. **Reshape each variable into a dense matrix** of shape (344,208 cells Ã— 28 years). This separates topology from time.

3. **Compute neighbor statistics via sparse matrixâ€“dense matrix multiplication and sparse row operations:**
   - **Mean:** `A %*% X / degree` (where degree = row sums of A).
   - **Max and Min:** Use a single pass over the CSR structure in C++ via `Rcpp`, or use a clever sparse-matrix trick with repeated masking. The most efficient approach is a small `Rcpp` function that iterates over CSR row pointers.

4. **Map results back** to the original long-format data.frame, preserving exact numerical equivalence.

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**. The sparse matrix multiply for mean is near-instantaneous. The Rcpp loop for max/min over ~1.37M edges Ã— 28 years is also very fast.

## Working R Code

```r
# =============================================================================
# Optimized neighborâ€statistics pipeline
# Preserves numerical equivalence with the original compute_neighbor_stats
# =============================================================================

library(Matrix)
library(Rcpp)
library(data.table)

# ---- Step 0: Rcpp workhorse for row-wise max / min over sparse adjacency ----
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
#include <cmath>
#include <limits>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_row_maxmin(IntegerVector Ap,      // CSR row pointers (length n+1, 0-based)
                       IntegerVector Aj,      // CSR column indices (0-based)
                       NumericMatrix X,        // dense matrix n x T
                       IntegerVector degree) { // row degrees
  int n = X.nrow();
  int TT = X.ncol();
  NumericMatrix out_max(n, TT);
  NumericMatrix out_min(n, TT);

  for (int i = 0; i < n; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    if (start == end) {
      // no neighbors
      for (int t = 0; t < TT; t++) {
        out_max(i, t) = NA_REAL;
        out_min(i, t) = NA_REAL;
      }
      continue;
    }
    for (int t = 0; t < TT; t++) {
      double vmax = -std::numeric_limits<double>::infinity();
      double vmin =  std::numeric_limits<double>::infinity();
      int valid = 0;
      for (int p = start; p < end; p++) {
        int j = Aj[p];
        double val = X(j, t);
        if (!ISNA(val) && !ISNAN(val)) {
          if (val > vmax) vmax = val;
          if (val < vmin) vmin = val;
          valid++;
        }
      }
      if (valid == 0) {
        out_max(i, t) = NA_REAL;
        out_min(i, t) = NA_REAL;
      } else {
        out_max(i, t) = vmax;
        out_min(i, t) = vmin;
      }
    }
  }
  return List::create(Named("max") = out_max,
                      Named("min") = out_min);
}

// [[Rcpp::export]]
NumericMatrix sparse_row_mean_na(IntegerVector Ap,
                                  IntegerVector Aj,
                                  NumericMatrix X) {
  // Computes row-wise mean of neighbor values, skipping NAs,
  // exactly matching: mean(neighbor_vals[!is.na(neighbor_vals)])
  int n = X.nrow();
  int TT = X.ncol();
  NumericMatrix out(n, TT);

  for (int i = 0; i < n; i++) {
    int start = Ap[i];
    int end   = Ap[i + 1];
    if (start == end) {
      for (int t = 0; t < TT; t++) out(i, t) = NA_REAL;
      continue;
    }
    for (int t = 0; t < TT; t++) {
      double s = 0.0;
      int valid = 0;
      for (int p = start; p < end; p++) {
        double val = X(Aj[p], t);
        if (!ISNA(val) && !ISNAN(val)) {
          s += val;
          valid++;
        }
      }
      out(i, t) = (valid == 0) ? NA_REAL : s / valid;
    }
  }
  return out;
}
')

# ---- Step 1: Build the sparse adjacency matrix once --------------------------

build_adjacency_csr <- function(id_order, nb_object) {
  # id_order: vector of 344,208 cell IDs in the order matching nb_object
  # nb_object: spdep nb list (rook_neighbors_unique), 1-indexed into id_order
  n <- length(id_order)
  stopifnot(length(nb_object) == n)

  # Build COO triplets
  from_list <- vector("list", n)
  to_list   <- vector("list", n)
  for (i in seq_len(n)) {
    nbrs <- nb_object[[i]]
    # spdep nb objects use integer(0) or 0L for no-neighbor; filter
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_list[[i]] <- rep.int(i, length(nbrs))
      to_list[[i]]   <- nbrs
    }
  }
  from_idx <- unlist(from_list, use.names = FALSE)
  to_idx   <- unlist(to_list,   use.names = FALSE)

  # Sparse matrix (dgRMatrix would be ideal but we build dgCMatrix then convert)
  A <- sparseMatrix(i = from_idx, j = to_idx, x = 1,
                    dims = c(n, n), repr = "C")  # CSC
  # Convert to CSR (dgRMatrix) for row-wise access
  A_csr <- as(A, "RsparseMatrix")

  list(
    Ap     = A_csr@p,        # row pointers, 0-based, length n+1
    Aj     = A_csr@j,        # column indices, 0-based
    n      = n,
    degree = diff(A_csr@p)   # number of neighbors per node
  )
}

# ---- Step 2: Reshape long data to cell Ã— year matrix -------------------------

long_to_matrix <- function(dt, var_name, cell_idx_col, year_col, years) {
  # dt: data.table with columns cell_idx_col (integer 1..N), year_col, var_name
  # Returns N x T matrix
  n_cells <- max(dt[[cell_idx_col]])
  n_years <- length(years)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  year_map <- setNames(seq_along(years), as.character(years))
  ci <- dt[[cell_idx_col]]
  yi <- year_map[as.character(dt[[year_col]])]
  mat[cbind(ci, yi)] <- dt[[var_name]]
  mat
}

# ---- Step 3: Master pipeline -------------------------------------------------

add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars,
                                      years = 1992:2019) {
  message("Converting to data.table...")
  dt <- as.data.table(cell_data)

  # Map cell IDs to 1..N indices matching id_order / nb_object order
  id_map <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_map[as.character(id)]]

  message("Building sparse adjacency (CSR)...")
  adj <- build_adjacency_csr(id_order, rook_neighbors_unique)
  Ap <- as.integer(adj$Ap)
  Aj <- as.integer(adj$Aj)
  deg <- as.integer(adj$degree)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing variable: %s", var_name))

    # Reshape to N x T matrix
    X <- long_to_matrix(dt, var_name, "cell_idx", "year", years)

    # Compute max, min via Rcpp CSR traversal
    maxmin <- sparse_row_maxmin(Ap, Aj, X, deg)
    # Compute mean via Rcpp CSR traversal (NA-aware, exact match)
    M <- sparse_row_mean_na(Ap, Aj, X)

    # Map back to long format using (cell_idx, year_idx)
    year_map <- setNames(seq_along(years), as.character(years))
    yi <- year_map[as.character(dt$year)]
    ci <- dt$cell_idx
    lin_idx <- cbind(ci, yi)

    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    dt[, (max_col)  := maxmin$max[lin_idx]]
    dt[, (min_col)  := maxmin$min[lin_idx]]
    dt[, (mean_col) := M[lin_idx]]

    # Free memory
    rm(X, maxmin, M); gc()
  }

  dt[, cell_idx := NULL]
  message("Done.")
  as.data.frame(dt)
}

# ---- Step 4: Execute ---------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars,
  years = 1992:2019
)

# ---- Step 5: Predict with the pre-trained Random Forest (unchanged) ----------
# The RF model is NOT retrained. We simply call predict() with the enriched data.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Numerically Equivalent

| Original | Optimized | Match? |
|---|---|---|
| `max(neighbor_vals[!is.na(neighbor_vals)])` | Rcpp loop: track running max, skip `NA` | âœ… Exact |
| `min(neighbor_vals[!is.na(neighbor_vals)])` | Rcpp loop: track running min, skip `NA` | âœ… Exact |
| `mean(neighbor_vals[!is.na(neighbor_vals)])` | Rcpp loop: accumulate sum and count, skip `NA`, divide | âœ… Exact (IEEE 754 summation order is row-sequential in both) |
| No neighbors â†’ `c(NA, NA, NA)` | `start == end` â†’ `NA_REAL` | âœ… Exact |
| All neighbor values `NA` â†’ `c(NA, NA, NA)` | `valid == 0` â†’ `NA_REAL` | âœ… Exact |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Topology construction | ~6.46M string pastes + named vector lookups | One-time `sparseMatrix` build (~1.37M edges) |
| Per-variable aggregation | 6.46M R-level `lapply` iterations | Single Rcpp pass over CSR Ã— 28 columns |
| Total estimated time | **86+ hours** | **~2â€“5 minutes** |
| Peak RAM | Large list-of-lists (~6.46M entries) | ~344K Ã— 28 dense matrix + CSR arrays â‰ˆ **< 2 GB** |
| RF model | Unchanged | Unchanged |
| Numerical result | Baseline | **Identical** |
END_RESPONSE>>>
