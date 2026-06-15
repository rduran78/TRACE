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
run_id: Anthropic_F10_T00_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list entries**, each constructed via string-pasting, named-vector lookup, and `NA` filtering. This is an O(n) loop over millions of rows with expensive string operations (`paste`, named vector indexing). The string-keyed lookup `idx_lookup` is a named character vector of length ~6.46M â€” each lookup into it is O(n) in the worst case because R named vectors use linear hashing that degrades at scale.

2. **`compute_neighbor_stats` iterates over ~6.46 million list elements** per variable, subsetting a numeric vector and computing `max/min/mean`. With 5 variables, that's ~32.3 million R-level list iterations, each with allocation overhead.

3. **The neighbor lookup is row-indexed (cellÃ—year)**, but the graph topology is **year-invariant** â€” every year has the same rook adjacency. The current code redundantly encodes the same spatial graph 28 times (once per year), inflating the lookup from ~1.37M edges to ~38.4M edge references.

**Key insight**: The adjacency graph is purely spatial (rook neighbors between cells). It does not change across years. The current code entangles topology with the panel structure by building a row-level lookup. We should separate topology from temporal indexing.

---

## Optimization Strategy

1. **Build a sparse adjacency matrix once** from the `nb` object (~344K Ã— 344K, ~1.37M non-zero entries). This is a CSC/CSR matrix from the `Matrix` package â€” native C-level sparse operations.

2. **Reshape each variable into a cell Ã— year matrix** (344,208 rows Ã— 28 columns). Each column is one year's values for all cells.

3. **Compute neighbor aggregates via sparse matrix multiplication**:
   - **Mean**: `A_norm %*% X` where `A_norm` is the row-normalized adjacency matrix (each row sums to the number of neighbors, then divided).
   - **Sum**: `A %*% X` gives neighbor sums; combined with neighbor count, yields mean.
   - **Max/Min**: Use a custom sparse row-wise extrema function that iterates over the CSC structure in C++ via `Rcpp`, or use an R-level vectorized approach over the sparse structure.

4. **For max and min**, sparse matrix algebra doesn't directly apply (they're not linear). We use an efficient vectorized approach: extract the (i, j) pairs from the sparse matrix once, then for each variable-year column, do a `data.table` grouped aggregation on the neighbor pairs â€” or better, use `Rcpp` to iterate over the CSR structure.

5. **Bind results back** to the panel `data.table` and score with the existing Random Forest.

**Expected speedup**: From ~86 hours to **minutes**. The sparse matrixâ€“vector product for mean runs in O(nnz) â‰ˆ 1.37M multiplications per year per variable. Max/min via Rcpp CSR traversal is similarly O(nnz). Total work: 5 vars Ã— 28 years Ã— 3 stats Ã— O(1.37M) â‰ˆ ~576M simple operations â€” trivially fast.

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original max/min/mean neighbor stats.
# =============================================================================

library(data.table)
library(Matrix)
library(Rcpp)

# ---- Step 0: Compile the Rcpp sparse row-wise max/min kernel ----------------

sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// Compute row-wise max and min over a CSR sparse matrix applied to a dense
// matrix X (n x k). For each row i of the adjacency and each column c of X,
// we compute max and min of X[j, c] over all neighbors j of i.
//
// CSR representation: row_ptr (length n+1), col_idx (length nnz), both 0-based.
// X is n x k (column-major as R stores it).
// Returns a list with "max_mat" and "min_mat", each n x k.

// [[Rcpp::export]]
List csr_neighbor_maxmin(IntegerVector row_ptr,
                         IntegerVector col_idx,
                         NumericMatrix X) {
  int n = X.nrow();
  int k = X.ncol();
  int nnz = col_idx.size();

  NumericMatrix max_mat(n, k);
  NumericMatrix min_mat(n, k);

  // Initialize to NA
  double na_val = NA_REAL;
  std::fill(max_mat.begin(), max_mat.end(), na_val);
  std::fill(min_mat.begin(), min_mat.end(), na_val);

  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    if (start == end) continue;  // no neighbors -> stays NA

    for (int c = 0; c < k; c++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid_count = 0;

      for (int p = start; p < end; p++) {
        int j = col_idx[p];
        double val = X(j, c);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid_count++;
        }
      }

      if (valid_count == 0) {
        max_mat(i, c) = na_val;
        min_mat(i, c) = na_val;
      } else {
        max_mat(i, c) = cur_max;
        min_mat(i, c) = cur_min;
      }
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}

// Row-wise mean with NA handling via CSR, applied to dense matrix X (n x k).
// [[Rcpp::export]]
NumericMatrix csr_neighbor_mean(IntegerVector row_ptr,
                                IntegerVector col_idx,
                                NumericMatrix X) {
  int n = X.nrow();
  int k = X.ncol();
  NumericMatrix mean_mat(n, k);
  double na_val = NA_REAL;
  std::fill(mean_mat.begin(), mean_mat.end(), na_val);

  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    if (start == end) continue;

    for (int c = 0; c < k; c++) {
      double sum_val = 0.0;
      int valid_count = 0;

      for (int p = start; p < end; p++) {
        int j = col_idx[p];
        double val = X(j, c);
        if (!R_IsNA(val)) {
          sum_val += val;
          valid_count++;
        }
      }

      if (valid_count == 0) {
        mean_mat(i, c) = na_val;
      } else {
        mean_mat(i, c) = sum_val / valid_count;
      }
    }
  }

  return mean_mat;
}
')

# ---- Step 1: Build sparse adjacency in CSR format (once) --------------------

build_csr_from_nb <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer vectors of neighbor indices, 1-based)
  # n: number of spatial cells (length of nb_obj)
  #
  # Returns list(row_ptr, col_idx) in 0-based indexing for Rcpp.

  # Count neighbors per node
  n_neighbors <- vapply(nb_obj, function(x) {
    # spdep nb encodes "no neighbors" as a single 0L
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1))

  nnz <- sum(n_neighbors)
  row_ptr <- integer(n + 1L)
  row_ptr[1L] <- 0L
  for (i in seq_len(n)) {
    row_ptr[i + 1L] <- row_ptr[i] + n_neighbors[i]
  }

  col_idx <- integer(nnz)
  pos <- 1L
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    if (!(length(nbrs) == 1L && nbrs[1L] == 0L)) {
      len <- length(nbrs)
      col_idx[pos:(pos + len - 1L)] <- nbrs - 1L  # 0-based
      pos <- pos + len
    }
  }

  list(row_ptr = row_ptr, col_idx = col_idx)
}

# ---- Step 2: Reshape panel to cell x year matrix ----------------------------

panel_to_matrix <- function(dt, var_name, id_order, year_order) {
  # dt: data.table with columns id, year, and var_name
  # id_order: integer vector of cell IDs defining row order (length = n_cells)
  # year_order: sorted integer vector of years defining column order
  #
  # Returns an n_cells x n_years numeric matrix.

  n_cells <- length(id_order)
  n_years <- length(year_order)

  # Map id -> row index, year -> col index
  id_map   <- setNames(seq_along(id_order), as.character(id_order))
  year_map <- setNames(seq_along(year_order), as.character(year_order))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_i <- id_map[as.character(dt$id)]
  col_j <- year_map[as.character(dt$year)]

  mat[cbind(row_i, col_j)] <- dt[[var_name]]
  mat
}

# ---- Step 3: Flatten matrix back to panel column ----------------------------

matrix_to_panel_col <- function(mat, dt, id_order, year_order) {
  # Reverse of panel_to_matrix: extract values aligned to dt rows.
  id_map   <- setNames(seq_along(id_order), as.character(id_order))
  year_map <- setNames(seq_along(year_order), as.character(year_order))

  row_i <- id_map[as.character(dt$id)]
  col_j <- year_map[as.character(dt$year)]

  mat[cbind(row_i, col_j)]
}

# ---- Step 4: Main pipeline --------------------------------------------------

run_neighbor_aggregation <- function(cell_data, id_order, rook_neighbors_unique,
                                     neighbor_source_vars) {
  # cell_data: data.table (or data.frame) with columns: id, year, and all vars
  # id_order: vector of unique cell IDs in the order matching rook_neighbors_unique

  # rook_neighbors_unique: spdep nb object
  # neighbor_source_vars: character vector of variable names

  # Convert to data.table for efficiency
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

  n_cells  <- length(id_order)
  year_order <- sort(unique(cell_data$year))
  n_years  <- length(year_order)

  cat("Building CSR adjacency structure...\n")
  csr <- build_csr_from_nb(rook_neighbors_unique, n_cells)
  cat(sprintf("  Cells: %d | Years: %d | Edges (nnz): %d\n",
              n_cells, n_years, length(csr$col_idx)))

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))

    # Reshape to cell x year matrix
    X <- panel_to_matrix(cell_data, var_name, id_order, year_order)

    # Compute neighbor max, min via Rcpp CSR kernel
    maxmin <- csr_neighbor_maxmin(csr$row_ptr, csr$col_idx, X)

    # Compute neighbor mean via Rcpp CSR kernel
    mean_mat <- csr_neighbor_mean(csr$row_ptr, csr$col_idx, X)

    # Map back to panel rows
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)

    cell_data[, (max_col_name)  := matrix_to_panel_col(maxmin$max_mat, cell_data,
                                                        id_order, year_order)]
    cell_data[, (min_col_name)  := matrix_to_panel_col(maxmin$min_mat, cell_data,
                                                        id_order, year_order)]
    cell_data[, (mean_col_name) := matrix_to_panel_col(mean_mat, cell_data,
                                                        id_order, year_order)]

    cat(sprintf("  Added: %s, %s, %s\n", max_col_name, min_col_name, mean_col_name))
  }

  cell_data
}

# ---- Step 5: Execute and predict --------------------------------------------

# Load pre-trained model and data (user-specific paths)
# rf_model            <- readRDS("path/to/trained_rf_model.rds")
# cell_data           <- readRDS("path/to/cell_data.rds")
# id_order            <- readRDS("path/to/id_order.rds")
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors_unique.rds")

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_aggregation(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# Score with the pre-trained Random Forest (no retraining)
cell_data$prediction <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Original | Optimized | Equivalence |
|---|---|---|
| `max(neighbor_vals[!is.na(...)])` | `csr_neighbor_maxmin` skips `NA`, takes max | Identical: same values, same `NA` propagation when all neighbors are `NA` or absent |
| `min(neighbor_vals[!is.na(...)])` | Same kernel, min branch | Identical |
| `mean(neighbor_vals[!is.na(...)])` | `csr_neighbor_mean` sums non-`NA` / count | Identical: `sum/count` with `NA` removal matches R's `mean(x, na.rm=FALSE)` on the filtered vector |
| No neighbors â†’ `c(NA, NA, NA)` | `start == end` â†’ row stays `NA` | Identical |
| All neighbor values `NA` â†’ `c(NA, NA, NA)` | `valid_count == 0` â†’ `NA` | Identical |

The adjacency is built from the same `rook_neighbors_unique` `nb` object, preserving the exact same directed edge set. The cell ordering is governed by `id_order`, which is the same vector used in the original `build_neighbor_lookup`.

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | ~6.46M string pastes + named vector lookups | One-time CSR build from `nb` (~344K nodes) |
| Per-variable work | 6.46M R list iterations | Single Rcpp pass over 1.37M edges Ã— 28 year-columns |
| Total R-level iterations | ~32.3M (5 vars Ã— 6.46M Ã— list ops) | 5 Rcpp calls (max/min combined + mean) |
| Memory | 6.46M-element list of integer vectors | CSR: two integer vectors (~5.5 MB) + one 344KÃ—28 matrix (~77 MB) per var |
| Estimated wall time | ~86+ hours | **~2â€“5 minutes** |
| RAM peak | Potentially >16 GB (string bloat) | ~2â€“3 GB comfortably within 16 GB |

The trained Random Forest model is never modified or retrained â€” it is only used for `predict()` after the neighbor features are attached.
END_RESPONSE>>>
