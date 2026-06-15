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
run_id: Anthropic_F07_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row character key construction and hash-table lookup** â€” `paste()` and named-vector indexing are called ~6.46 million times inside an `lapply`, each time creating small character vectors and doing partial matching against `idx_lookup` (a named vector of length 6.46M). Named-vector lookup in R is O(n) per probe in the worst case (it's a linear scan unless R internally hashes it, which for 6.46M names is unreliable and memory-heavy).

2. **Redundant recomputation across years** â€” The neighbor *topology* is fixed across all 28 years (rook neighbors don't change). Yet the function re-discovers neighbors for every cell-year row independently. For 344,208 cells Ã— 28 years, the same neighbor list is looked up 28 times per cell.

3. **`compute_neighbor_stats`** is also slow because it loops over 6.46M elements in R-level `lapply`, extracting subsets of a vector each time.

**Estimated cost**: ~6.46M iterations Ã— expensive string operations + 6.46M hash lookups against a 6.46M-entry table = 86+ hours.

## Optimization Strategy

### Key Insight: Separate Topology from Time

The neighbor graph is **time-invariant**. Instead of building a 6.46M-element row-level lookup, we:

1. **Build a sparse adjacency matrix `W`** (344,208 Ã— 344,208) from `rook_neighbors_unique` once. This is a binary CSC/CSR matrix â€” trivially constructed from an `nb` object via `spdep::nb2listw` â†’ `as_dgRMatrix` or directly.

2. **Reshape each variable into a matrix** of dimension (344,208 cells Ã— 28 years). Each column is one year.

3. **Compute neighbor stats via sparse matrixâ€“dense matrix multiplication**:
   - **Neighbor sum** = `W %*% X` (sparse Ã— dense, highly optimized in C via the `Matrix` package).
   - **Neighbor count** = `W %*% (!is.na(X))` (to handle NAs correctly).
   - **Neighbor mean** = sum / count.
   - **Neighbor max and min** require a grouped operation, but can be done efficiently column-by-column using the sparse structure of `W` iterated in C++ via `Rcpp`, or via a vectorized row-wise approach on the sparse matrix.

4. **Melt back** to the long panel and join.

This replaces 6.46M R-level iterations with a handful of sparse-matrix operations (each taking seconds) and one Rcpp loop for max/min.

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| Lookup build | O(6.46M Ã— string ops) | O(1.37M) sparse matrix build, once |
| Mean per variable | O(6.46M) R-level lapply | O(nnz Ã— 28) sparse matmul (~seconds) |
| Max/Min per variable | O(6.46M) R-level lapply | O(nnz Ã— 28) Rcpp loop (~seconds) |
| **Total estimated time** | **86+ hours** | **~2â€“5 minutes** |

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Preserves the original numerical estimand exactly.
# Preserves the trained Random Forest model (no retraining).
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# ---- 1. Build sparse binary adjacency matrix from nb object (once) ----------

build_adjacency_matrix <- function(nb_obj, n) {

  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial units (length of nb_obj)
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)
  # Remove zero-neighbor entries (spdep uses integer(0) or 0L)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]
  sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
}

n_cells <- length(rook_neighbors_unique)  # 344,208
W <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# ---- 2. Rcpp function for sparse-neighbor max and min ----------------------

cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(IntegerVector Wp,     // CSC column pointers (length n+1)
                            IntegerVector Wi,     // CSC row indices
                            NumericMatrix X,      // n_cells x n_years
                            int n, int nyears) {
  // W is n x n in CSC. Column j lists the rows i such that W[i,j]=1,
  // i.e., j is a neighbor of i.  But we built W so that W[i,j]=1 means
  // j is a neighbor of i.  In CSC, iterating column j gives rows i where
  // W[i,j]=1.  We need, for each row i, the values X[j,] for all j in
  // neighbors(i).  That is iterating ROW i of W.  CSC is efficient for
  // column iteration, so we transpose: iterate columns of W^T = rows of W.
  // Actually, since W is built symmetrically for rook neighbors, W = W^T.
  // So iterating column i of W gives the neighbors of i.

  NumericMatrix maxMat(n, nyears);
  NumericMatrix minMat(n, nyears);

  // Initialize
  for (int i = 0; i < n; i++) {
    for (int t = 0; t < nyears; t++) {
      maxMat(i, t) = NA_REAL;
      minMat(i, t) = NA_REAL;
    }
  }

  for (int i = 0; i < n; i++) {
    int p_start = Wp[i];
    int p_end   = Wp[i + 1];
    if (p_start == p_end) continue;  // no neighbors

    for (int t = 0; t < nyears; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int    count   = 0;
      for (int p = p_start; p < p_end; p++) {
        int j = Wi[p];
        double val = X(j, t);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          count++;
        }
      }
      if (count > 0) {
        maxMat(i, t) = cur_max;
        minMat(i, t) = cur_min;
      }
    }
  }

  return List::create(Named("max") = maxMat,
                      Named("min") = minMat);
}
')

# ---- 3. Main function: compute all neighbor stats for one variable ----------

compute_neighbor_features_fast <- function(cell_dt, var_name, W, id_order, years) {
  # cell_dt:  data.table with columns id, year, <var_name>
  # W:        sparse adjacency matrix (n_cells x n_cells), CSC
  # id_order: vector of cell IDs in the order matching W rows/cols
  # years:    sorted vector of unique years

  n_cells <- length(id_order)
  n_years <- length(years)

  # Map cell id -> matrix row index
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))

  # Map year -> matrix column index
  year_to_col <- setNames(seq_along(years), as.character(years))

  # Build the n_cells x n_years matrix X from the long data
  row_idx <- id_to_row[as.character(cell_dt$id)]
  col_idx <- year_to_col[as.character(cell_dt$year)]
  vals    <- cell_dt[[var_name]]

  X <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  X[cbind(row_idx, col_idx)] <- vals

  # --- Neighbor mean via sparse matrix algebra ---
  # Handle NAs: replace NA with 0 for summation, track counts separately
  X_nona <- X
  X_nona[is.na(X)] <- 0
  not_na <- (!is.na(X)) * 1.0  # indicator matrix

  neighbor_sum   <- as.matrix(W %*% X_nona)       # n_cells x n_years
  neighbor_count <- as.matrix(W %*% not_na)        # n_cells x n_years

  neighbor_mean <- neighbor_sum / neighbor_count
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- Neighbor max and min via Rcpp ---
  # Ensure W is in dgCMatrix (CSC) format
  W_csc <- as(W, "dgCMatrix")
  maxmin <- sparse_neighbor_maxmin(W_csc@p, W_csc@i, X, n_cells, n_years)
  neighbor_max <- maxmin$max   # n_cells x n_years matrix
  neighbor_min <- maxmin$min

  # --- Melt back to long format and join ---
  # Create a data.table of results indexed by (id, year)
  result_dt <- data.table(
    id   = rep(id_order, times = n_years),
    year = rep(years, each = n_cells),
    nb_max  = as.vector(neighbor_max),
    nb_min  = as.vector(neighbor_min),
    nb_mean = as.vector(neighbor_mean)
  )

  # Rename columns to match original naming convention
  setnames(result_dt,
           c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean")))

  result_dt
}

# ---- 4. Outer loop: apply to all 5 variables and join to cell_data ----------

# Convert to data.table for fast joins
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Sorted unique years
years <- sort(unique(cell_data$year))

# id_order must match the ordering used when building rook_neighbors_unique.
# This is typically the order of cell IDs in the original spatial object.
# Verify: length(id_order) == length(rook_neighbors_unique) == 344208

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")

  result_dt <- compute_neighbor_features_fast(
    cell_dt  = cell_data,
    var_name = var_name,
    W        = W,
    id_order = id_order,
    years    = years
  )

  setkey(result_dt, id, year)

  # Remove old columns if they exist (idempotent re-runs)
  old_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  for (col in old_cols) {
    if (col %in% names(cell_data)) cell_data[, (col) := NULL]
  }

  # Join new columns
  cell_data <- result_dt[cell_data, on = .(id, year)]

  cat("  Done.\n")
}

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_data)

# ---- 5. Predict with the existing trained Random Forest (unchanged) ---------
# The trained RF model object is used as-is. No retraining.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Detail |
|---|---|
| **Root cause** | 6.46M R-level iterations with per-row string hashing against a 6.46M-entry named vector; neighbor topology redundantly recomputed for every year. |
| **Fix** | Factor the problem into time-invariant topology (sparse matrix `W`, built once) Ã— time-varying values (dense matrix `X`, one column per year). Mean via sparse matmul; max/min via Rcpp over the CSC structure. |
| **Numerical equivalence** | Guaranteed â€” same neighbor sets, same arithmetic (sum/count for mean, elementwise max/min). NA handling is identical. |
| **RAM** | Peak â‰ˆ 344K Ã— 28 Ã— 8 bytes Ã— ~6 matrices â‰ˆ **460 MB**, well within 16 GB. The sparse matrix `W` with 1.37M entries â‰ˆ 22 MB. |
| **Expected runtime** | ~2â€“5 minutes total for all 5 variables (vs. 86+ hours). |
| **RF model** | Untouched â€” column names and values are identical, so `predict()` works as before. |
END_RESPONSE>>>
