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
run_id: Anthropic_F10_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(nÂ²)-scale neighbor lookup construction**: `build_neighbor_lookup` iterates over all ~6.46M rows, performing string-paste key creation and named-vector lookups (`idx_lookup[neighbor_keys]`) for every single row. Named vector lookups in R are hash-based but still carry significant overhead at this scale. The string concatenation of `paste(id, year, sep="_")` for ~6.46M keys, then querying them ~6.46M Ã— avg_neighbors times, is extremely expensive.

2. **Row-wise `lapply` in `compute_neighbor_stats`**: Iterating over 6.46M list elements in R, each invoking `max`, `min`, `mean` on small vectors, produces massive interpreter overhead. This is called 5 times (once per variable), totaling ~32.3M R-level function calls.

3. **Redundant topology**: The neighbor graph is **time-invariant** â€” rook adjacency depends on spatial position, not year. Yet the lookup is built at the cell-year level, inflating a ~344K-node spatial graph to a ~6.46M-node spatiotemporal lookup. The same adjacency structure is needlessly replicated 28 times.

**Why 86+ hours**: ~6.46M list elements Ã— 5 variables Ã— (string ops + R-level aggregation) â‰ˆ billions of interpreted R operations.

## Optimization Strategy

1. **Separate topology from time**: Build a sparse adjacency structure once over the 344,208 spatial cells only. The rook neighbor object already provides this.

2. **Convert the `nb` object to a sparse matrix**: Represent adjacency as a `dgCMatrix` (compressed sparse column) from the `Matrix` package. Sparse matrixâ€“dense matrix multiplication computes neighborhood sums in one vectorized BLAS call. Neighbor counts give means; row-wise operations give max/min.

3. **Vectorized aggregation via sparse matrix operations**:
   - **Mean**: `(A %*% X) / neighbor_count` where A is the adjacency matrix and X is the variable matrix (cells Ã— years).
   - **Max / Min**: Use grouped operations via the sparse matrix structure â€” extract column indices per row and compute max/min in C++ via `Rcpp`, or use `data.table` grouped operations on the edge list.

4. **Memory layout**: Reshape each variable into a 344,208 Ã— 28 matrix (cells Ã— years). Sparse-matrix Ã— dense-matrix multiplication is cache-friendly and leverages optimized BLAS.

5. **Preserve numerical equivalence**: The sparse matrix approach computes the identical sum, count, max, and min over the identical neighbor sets, producing bit-identical results (for mean: identical to floating-point precision of `sum/count`).

6. **Do not retrain the RF**: Only the feature-engineering step is replaced; the pre-trained model is loaded and applied via `predict()` as before.

## Working R Code

```r
# ==============================================================================
# Optimized Neighbor Feature Engineering Pipeline
# ==============================================================================
# Requirements: Matrix, data.table, Rcpp (all standard, no exotic dependencies)
# ==============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# --------------------------------------------------------------------------
# Step 0: Inline C++ for sparse-row-wise max and min
# --------------------------------------------------------------------------
cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// Compute row-wise max and min of neighbor values using CSR representation.
// p: row pointers (length nrow+1, 0-indexed)
// j: column indices (0-indexed)
// valmat: dense matrix (ncol_cells x nyears), column-major
// Returns a list with two matrices: max_mat and min_mat (nrow x nyears)
// [[Rcpp::export]]
List rowwise_maxmin_sparse(IntegerVector p, IntegerVector j,
                           NumericMatrix valmat, int nrow_out) {
  int nyears = valmat.ncol();
  int ncells = nrow_out;

  NumericMatrix max_mat(ncells, nyears);
  NumericMatrix min_mat(ncells, nyears);

  // Initialize with NA
  double na_val = NA_REAL;
  std::fill(max_mat.begin(), max_mat.end(), na_val);
  std::fill(min_mat.begin(), min_mat.end(), na_val);

  for (int i = 0; i < ncells; i++) {
    int start = p[i];
    int end   = p[i + 1];
    if (start == end) continue; // no neighbors

    for (int t = 0; t < nyears; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid = 0;

      for (int k = start; k < end; k++) {
        double v = valmat(j[k], t);
        if (!R_IsNA(v)) {
          if (v > cur_max) cur_max = v;
          if (v < cur_min) cur_min = v;
          valid++;
        }
      }

      if (valid > 0) {
        max_mat(i, t) = cur_max;
        min_mat(i, t) = cur_min;
      }
      // else remains NA
    }
  }

  return List::create(Named("max_mat") = max_mat,
                      Named("min_mat") = min_mat);
}
')

# --------------------------------------------------------------------------
# Step 1: Build spatial adjacency matrix ONCE (344,208 x 344,208 sparse)
# --------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj: spdep nb object (list of integer vectors of neighbor indices)
  # n: number of spatial cells
  # Returns: sparse dgCMatrix (n x n) with 1s at neighbor positions

  # Build COO triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj)

  # Remove any 0-neighbor entries (empty integer(0) elements produce nothing)
  valid <- to > 0L
  from  <- from[valid]
  to    <- to[valid]

  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n),
                    repr = "C")  # CSC format
  return(A)
}

# --------------------------------------------------------------------------
# Step 2: Reshape long panel to cell x year matrix for one variable
# --------------------------------------------------------------------------
reshape_to_matrix <- function(dt, var_name, cell_idx, year_idx, n_cells, n_years) {
  # dt: data.table with columns id, year, and var_name
  # cell_idx: named integer vector mapping cell id -> row position (1..n_cells)
  # year_idx: named integer vector mapping year -> col position (1..n_years)
  # Returns: n_cells x n_years numeric matrix

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  ri  <- cell_idx[as.character(dt$id)]
  ci  <- year_idx[as.character(dt$year)]
  mat[cbind(ri, ci)] <- dt[[var_name]]
  return(mat)
}

# --------------------------------------------------------------------------
# Step 3: Compute neighbor mean via sparse matrix multiplication
# --------------------------------------------------------------------------
compute_neighbor_mean <- function(A, val_mat, neighbor_counts) {
  # A: sparse adjacency matrix (n_cells x n_cells)
  # val_mat: dense matrix (n_cells x n_years)
  # neighbor_counts: integer vector length n_cells (number of neighbors per cell)
  # Returns: n_cells x n_years matrix of neighbor means

  # Replace NA with 0 for summation, track valid counts
  is_valid  <- !is.na(val_mat)  # logical matrix
  val_clean <- val_mat
  val_clean[!is_valid] <- 0

  # Neighbor sums (sparse %*% dense is highly optimized)
  sum_mat   <- as.matrix(A %*% val_clean)    # n_cells x n_years
  count_mat <- as.matrix(A %*% (is_valid * 1))  # valid neighbor counts per cell-year

  mean_mat <- sum_mat / count_mat  # NaN where count==0
  mean_mat[count_mat == 0] <- NA_real_
  return(mean_mat)
}

# --------------------------------------------------------------------------
# Step 4: Compute neighbor max and min via C++ with CSR structure
# --------------------------------------------------------------------------
compute_neighbor_maxmin <- function(A_csr, val_mat) {
  # A_csr: sparse dgRMatrix (CSR) adjacency matrix
  # val_mat: dense matrix (n_cells x n_years)
  # Returns: list with max_mat and min_mat

  # dgRMatrix stores: @p (row pointers), @j (column indices, 0-based)
  rowwise_maxmin_sparse(A_csr@p, A_csr@j, val_mat, nrow(val_mat))
}

# --------------------------------------------------------------------------
# Step 5: Flatten matrix back to long format column
# --------------------------------------------------------------------------
flatten_matrix_to_long <- function(mat, cell_idx, year_idx, dt) {
  ri <- cell_idx[as.character(dt$id)]
  ci <- year_idx[as.character(dt$year)]
  mat[cbind(ri, ci)]
}

# ==========================================================================
# MAIN PIPELINE
# ==========================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
  # Convert to data.table for speed
  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  cat("Cells:", n_cells, " Years:", n_years, " Rows:", nrow(dt), "\n")

  # --- Build mappings ---
  cell_idx <- setNames(seq_along(id_order), as.character(id_order))
  year_idx <- setNames(seq_along(years), as.character(years))

  # --- Step 1: Build adjacency matrix ONCE ---
  cat("Building sparse adjacency matrix...\n")
  A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  cat("  Adjacency matrix:", nrow(A), "x", ncol(A),
      " nnz:", nnzero(A), "\n")

  # Convert to CSR (dgRMatrix) for row-wise max/min in C++
  A_csr <- as(A, "RsparseMatrix")

  # --- Compute neighbor features for each source variable ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")
    t0 <- proc.time()

    # Reshape to matrix
    val_mat <- reshape_to_matrix(dt, var_name, cell_idx, year_idx,
                                 n_cells, n_years)

    # Compute mean via sparse matmul
    mean_mat <- compute_neighbor_mean(A, val_mat, neighbor_counts = NULL)

    # Compute max and min via C++
    maxmin   <- compute_neighbor_maxmin(A_csr, val_mat)
    max_mat  <- maxmin$max_mat
    min_mat  <- maxmin$min_mat

    # Flatten back to long format and add columns
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := flatten_matrix_to_long(max_mat, cell_idx, year_idx, dt)]
    dt[, (min_col)  := flatten_matrix_to_long(min_mat, cell_idx, year_idx, dt)]
    dt[, (mean_col) := flatten_matrix_to_long(mean_mat, cell_idx, year_idx, dt)]

    elapsed <- (proc.time() - t0)[3]
    cat("  Done in", round(elapsed, 1), "seconds\n")

    # Free intermediate matrices
    rm(val_mat, mean_mat, max_mat, min_mat, maxmin)
  }

  cat("All neighbor features computed.\n")

  # --- Apply pre-trained Random Forest (no retraining) ---
  cat("Generating predictions with pre-trained RF model...\n")
  dt[, prediction := predict(rf_model, newdata = dt)]

  cat("Pipeline complete.\n")
  return(dt)
}

# ==========================================================================
# EXECUTION
# ==========================================================================
# Assuming these objects are already in the environment:
#   cell_data              - data.frame/data.table with id, year, variables
#   id_order               - integer vector of cell IDs in nb-object order
#   rook_neighbors_unique  - spdep nb object (list of neighbor index vectors)
#   trained_rf_model       - pre-trained randomForest / ranger model object

# result <- run_optimized_pipeline(cell_data, id_order,
#                                  rook_neighbors_unique, trained_rf_model)
```

## Performance Analysis

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~hours (6.46M string ops) | ~2 sec (sparse matrix from nb) | ~1000Ã— |
| Mean (per variable) | ~hours (6.46M lapply calls) | ~10â€“30 sec (sparse matmul, BLAS) | ~200Ã— |
| Max/Min (per variable) | ~hours (6.46M R-level loops) | ~30â€“60 sec (C++ CSR traversal) | ~100Ã— |
| Total (5 variables) | ~86+ hours | **~5â€“10 minutes** | ~500â€“1000Ã— |
| Peak RAM | ~8â€“12 GB (6.46M-element list) | ~4â€“6 GB (sparse mat + dense matrices) | ~2Ã— less |

**Key design principles applied:**

1. **Build topology once in sparse matrix form** â€” the `nb` â†’ `dgCMatrix` conversion happens once and the same matrix is reused for all 5 variables and all 28 years simultaneously.

2. **Vectorized mean via sparse Ã— dense matmul** â€” `A %*% X` computes all neighbor sums across all cells and all years in a single BLAS call. NA handling uses a parallel valid-count matrix.

3. **C++ for max/min** â€” These are not expressible as linear algebra, so a compiled CSR traversal replaces 6.46M R-level `lapply` calls. The CSR representation (`dgRMatrix`) gives contiguous memory access per row.

4. **Year-axis parallelism for free** â€” By reshaping each variable into a (344,208 Ã— 28) matrix, the sparse matmul processes all 28 years simultaneously without any explicit loop over years.

5. **Numerical equivalence**: The neighbor sets are identical (same `nb` object â†’ same adjacency entries). Sum/count â†’ mean, element-wise max/min over the same neighbor values â†’ bit-identical results. The pre-trained RF model is applied via `predict()` without modification.
END_RESPONSE>>>
