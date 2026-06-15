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
run_id: Anthropic_F10_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F10_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million R list elements**, each via an anonymous function call with string concatenation (`paste`), named-vector lookups, and NA filtering. This is an O(n) loop over millions of rows with expensive string operations at each step.

2. **`compute_neighbor_stats` iterates over ~6.46 million list elements** per variable, calling `max`, `min`, `mean` inside an `lapply`. With 5 variables, that's ~32.3 million R-level function invocations with per-element allocation.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it's year-specific.** Each cell has the same rook neighbors every year. The `paste(id, year)` key construction entangles topology with time, preventing vectorized exploitation of the panel's regular structure.

**Root cause:** The design treats the problem as a flat row-level operation instead of exploiting the fact that the adjacency structure is **static across years** and the panel is **balanced** (every cell appears in every year). This means neighbor aggregation can be expressed as a **sparse matrixâ€“dense matrix multiplication**, which is O(nnz) and runs in optimized C/Fortran code.

---

## Optimization Strategy

### Key Insight
If `A` is the `N_cells Ã— N_cells` sparse adjacency matrix (rook neighbors), and `X` is an `N_cells Ã— N_years` matrix of a variable (one column per year), then:

- **Neighbor sum** = `A %*% X` (sparse matrix multiply, O(nnz Ã— T))
- **Neighbor count** = `A %*% (non-NA indicator matrix)` (same cost)
- **Neighbor mean** = sum / count
- **Neighbor max/min** require a grouped operation, but can be vectorized via the CSR representation of `A`

For **mean**, sparse matrix multiplication gives us exact numerical equivalence. For **max** and **min**, we iterate over the CSR row pointers in C++ via `Rcpp`, which is O(nnz Ã— T) with no R-level per-element overhead.

### Complexity Comparison

| | Original | Optimized |
|---|---|---|
| Lookup build | O(R) string ops, R â‰ˆ 6.46M | O(N) integer ops, N = 344K (once) |
| Mean (per var) | O(R) R-level loops | O(nnz Ã— T) BLAS sparse multiply |
| Max/Min (per var) | O(R) R-level loops | O(nnz Ã— T) compiled C++ |
| Total R-level calls | ~32M+ | ~5 (one per variable) |
| Estimated time | 86+ hours | **Minutes** |

---

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with original compute_neighbor_stats.
# Requires: Matrix, Rcpp, data.table
# ==============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# --------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix from spdep nb object (ONCE)
# --------------------------------------------------------------------------
build_adjacency_matrix <- function(nb_object, n_cells) {
  # nb_object: list of length n_cells, each element is integer vector of neighbor indices
  # Builds a sparse CSR-compatible matrix (dgRMatrix) via COO -> dgCMatrix -> dgRMatrix
  
  from <- integer(0)
  to   <- integer(0)
  
  for (i in seq_len(n_cells)) {
    nbrs <- nb_object[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from <- c(from, rep.int(i, length(nbrs)))
      to   <- c(to, nbrs)
    }
  }
  
  A <- sparseMatrix(
    i    = from,
    j    = to,
    x    = rep.int(1, length(from)),
    dims = c(n_cells, n_cells),
    repr = "C"   # CSC format initially
  )
  
  return(A)
}

# --------------------------------------------------------------------------
# STEP 2: Rcpp function for sparse row-wise max and min over a dense matrix
#          Operates on CSR representation for cache-friendly row traversal.
# --------------------------------------------------------------------------
cppFunction('
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_row_maxmin(IntegerVector row_ptr,    // length n_rows + 1, 0-based
                       IntegerVector col_idx,    // length nnz, 0-based
                       NumericMatrix X,          // n_cells x n_years
                       int n_rows) {
  int n_cols = X.ncol();
  NumericMatrix out_max(n_rows, n_cols);
  NumericMatrix out_min(n_rows, n_cols);
  
  // Initialize to NA
  double na_val = NA_REAL;
  std::fill(out_max.begin(), out_max.end(), na_val);
  std::fill(out_min.begin(), out_min.end(), na_val);
  
  for (int i = 0; i < n_rows; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    
    if (start == end) continue;  // no neighbors, stays NA
    
    for (int t = 0; t < n_cols; t++) {
      double cur_max = R_NegInf;
      double cur_min = R_PosInf;
      int valid_count = 0;
      
      for (int jj = start; jj < end; jj++) {
        int neighbor = col_idx[jj];
        double val = X(neighbor, t);
        if (!R_IsNA(val)) {
          if (val > cur_max) cur_max = val;
          if (val < cur_min) cur_min = val;
          valid_count++;
        }
      }
      
      if (valid_count > 0) {
        out_max(i, t) = cur_max;
        out_min(i, t) = cur_min;
      }
      // else stays NA
    }
  }
  
  return List::create(Named("max_mat") = out_max,
                      Named("min_mat") = out_min);
}
')

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor mean via sparse matrix multiplication
#          Handles NAs correctly: mean = sum_of_non_NA / count_of_non_NA
# --------------------------------------------------------------------------
compute_neighbor_mean_sparse <- function(A, X_mat) {
  # X_mat: n_cells x n_years, may contain NAs
  # Replace NAs with 0 for summation, track non-NA indicator
  
  X_clean       <- X_mat
  na_mask       <- is.na(X_mat)
  X_clean[na_mask] <- 0
  
  indicator     <- matrix(1, nrow = nrow(X_mat), ncol = ncol(X_mat))
  indicator[na_mask] <- 0
  
  # Sparse multiply: A (n_cells x n_cells) %*% X_clean (n_cells x n_years)
  neighbor_sum   <- A %*% X_clean
  neighbor_count <- A %*% indicator
  
  # Convert to dense
  neighbor_sum   <- as.matrix(neighbor_sum)
  neighbor_count <- as.matrix(neighbor_count)
  
  # mean = sum / count; where count == 0, result is NA
  result <- neighbor_sum / neighbor_count
  result[neighbor_count == 0] <- NA_real_
  
  return(result)
}

# --------------------------------------------------------------------------
# STEP 4: Master function â€” reshape, aggregate, reshape back
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, 
                                           id_order, 
                                           rook_neighbors_unique,
                                           neighbor_source_vars) {
  
  n_cells <- length(id_order)
  
  cat("Building sparse adjacency matrix...\n")
  A_csc <- build_adjacency_matrix(rook_neighbors_unique, n_cells)
  
  # Convert to CSR (dgRMatrix) for the C++ max/min kernel
  A_csr <- as(A_csc, "RsparseMatrix")
  
  # Build cell-id to matrix-row mapping
  id_to_row <- setNames(seq_len(n_cells), as.character(id_order))
  
  # Convert to data.table for fast reshaping
  dt <- as.data.table(cell_data)
  
  # Get sorted unique years for consistent column ordering
  years_sorted <- sort(unique(dt$year))
  n_years      <- length(years_sorted)
  year_to_col  <- setNames(seq_len(n_years), as.character(years_sorted))
  
  # Map each row to its (matrix_row, matrix_col) position
  dt[, mat_row := id_to_row[as.character(id)]]
  dt[, mat_col := year_to_col[as.character(year)]]
  
  # Extract CSR components for C++ (0-based indexing)
  # dgRMatrix slots: @p (row pointers), @j (column indices), both 0-based
  row_ptr <- A_csr@p        # integer, length n_cells + 1, 0-based
  col_idx <- A_csr@j        # integer, length nnz, 0-based
  
  cat(sprintf("Adjacency: %d cells, %d directed edges, %d years\n",
              n_cells, length(col_idx), n_years))
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing variable: %s\n", var_name))
    
    # --- Reshape variable into n_cells x n_years dense matrix ---
    X_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    vals  <- dt[[var_name]]
    rows  <- dt$mat_row
    cols  <- dt$mat_col
    X_mat[cbind(rows, cols)] <- vals
    
    # --- Compute neighbor mean via sparse matmul ---
    mean_mat <- compute_neighbor_mean_sparse(A_csc, X_mat)
    
    # --- Compute neighbor max and min via C++ CSR traversal ---
    maxmin <- sparse_row_maxmin(row_ptr, col_idx, X_mat, n_cells)
    max_mat <- maxmin$max_mat
    min_mat <- maxmin$min_mat
    
    # --- Map results back to the original row order ---
    idx <- cbind(rows, cols)
    
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col_name)  := max_mat[idx]]
    dt[, (min_col_name)  := min_mat[idx]]
    dt[, (mean_col_name) := mean_mat[idx]]
    
    cat(sprintf("  Done: %s, %s, %s\n", max_col_name, min_col_name, mean_col_name))
  }
  
  # Remove helper columns
  dt[, c("mat_row", "mat_col") := NULL]
  
  return(as.data.frame(dt))
}

# ==========================================================================
# EXECUTION
# ==========================================================================

# --- Compute all neighbor features ---
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# --- Apply the pre-trained Random Forest (NO retraining) ---
# The model object (e.g., rf_model) is already in memory or loaded from disk.
# Predict using the enriched cell_data which now has all ~110 predictor columns.
cell_data$prediction <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

---

## Numerical Equivalence Verification

Run this once on a small subset to confirm the optimized code produces identical results:

```r
# --------------------------------------------------------------------------
# VERIFICATION: Compare original vs optimized on a sample
# --------------------------------------------------------------------------
verify_equivalence <- function(cell_data_original, cell_data_optimized, 
                                neighbor_source_vars, n_check = 1000) {
  set.seed(42)
  check_rows <- sample(nrow(cell_data_original), min(n_check, nrow(cell_data_original)))
  
  all_ok <- TRUE
  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col <- paste0("neighbor_", stat, "_", var_name)
      orig <- cell_data_original[[col]][check_rows]
      optm <- cell_data_optimized[[col]][check_rows]
      
      both_na <- is.na(orig) & is.na(optm)
      both_val <- !is.na(orig) & !is.na(optm)
      
      na_match <- all(both_na == (is.na(orig) | is.na(optm)))  
      # More precisely:
      na_mismatch <- sum(is.na(orig) != is.na(optm))
      val_match   <- all(abs(orig[both_val] - optm[both_val]) < 1e-10)
      
      if (na_mismatch > 0 || !val_match) {
        cat(sprintf("MISMATCH: %s â€” NA mismatches: %d, value match: %s\n",
                    col, na_mismatch, val_match))
        all_ok <- FALSE
      }
    }
  }
  
  if (all_ok) cat("All checks passed: numerical equivalence confirmed.\n")
  return(all_ok)
}
```

---

## Performance Summary

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| Topology construction | ~6.46M string ops | ~1.37M integer COO entries â†’ sparse matrix | ~100Ã— |
| Mean aggregation (per var) | 6.46M R `lapply` calls | 1 sparse matmul (CHOLMOD/BLAS) | ~500Ã— |
| Max/Min aggregation (per var) | 6.46M R `lapply` calls | 1 C++ CSR pass | ~200Ã— |
| Memory pattern | ~6.46M R list elements + string keys | Dense matrix (344K Ã— 28 â‰ˆ 77MB per var) + sparse matrix (~22MB) | ~3â€“5Ã— less |
| **Total estimated runtime** | **86+ hours** | **5â€“15 minutes** | **~500Ã—** |

The trained Random Forest model is never modified or retrained â€” it is applied as-is via `predict()` on the enriched dataset.
END_RESPONSE>>>
