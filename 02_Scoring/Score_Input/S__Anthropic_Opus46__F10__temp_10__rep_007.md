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
run_id: Anthropic_F10_T10_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical performance bottlenecks:

1. **O(nÂ²)-scale string-key lookup construction**: `build_neighbor_lookup` creates a named character vector (`idx_lookup`) of ~6.46 million paste-concatenated keys, then for each of the ~6.46M rows, performs character matching into that vector. Named vector lookup in R is linear-scan or hash-based but with enormous overhead at this scale.

2. **Row-level R `lapply` over 6.46M rows Ã— 5 variables**: `compute_neighbor_stats` calls an R closure 6.46M times per variable. Each invocation subsets, removes NAs, and computes three summary statistics. The per-call overhead of R function dispatch, subsetting, and GC pressure is massive. Over 5 variables, this is ~32.3 million R-level function calls.

3. **Redundant topology encoding**: The neighbor graph is purely spatial (rook contiguity) and is invariant across years. But `build_neighbor_lookup` expands it into a year-specific row-index lookup by pasting year suffixes â€” replicating the same ~1.37M edge topology 28 times into a list of ~6.46M entries. This consumes enormous memory and time.

**Why 86+ hours**: The dominant cost is the ~32.3M R-level `lapply` iterations with per-element subsetting, plus the initial ~6.46M string-matching operations. R's interpreted loop overhead makes this intractable.

---

## Optimization Strategy

### Core insight: Separate spatial topology from temporal indexing

The rook-neighbor graph is static across years. Instead of building a 6.46M-row lookup, we:

1. **Build the sparse adjacency structure once** from the `nb` object â€” just 344,208 nodes, ~1.37M edges.
2. **Organize data as a matrix** with rows = cells, columns = years, for each variable.
3. **Use sparse matrixâ€“dense matrix multiplication** (via the `Matrix` package) to compute neighbor sums, counts, maxima, and minima in vectorized operations across all cells and all years simultaneously.

### Specific techniques

- **CSR sparse adjacency matrix** `A` (344,208 Ã— 344,208, ~1.37M nonzeros): built once from the `nb` object.
- **Neighbor mean**: If `X` is the (cells Ã— years) value matrix, then `A %*% X` gives neighbor sums, and `A %*% (!is.na(X))` gives neighbor counts. Mean = sum/count.
- **Neighbor max/min**: Cannot be done by matrix multiplication directly (max/min are not linear). We use a **row-wise sparse iteration in C++ via Rcpp** over the CSR structure â€” but crucially this is compiled C++ iterating over the sparse structure, not 6.46M R function calls.
- **Memory**: Each (344,208 Ã— 28) double matrix is ~77 MB. With ~5 variables Ã— 3 stats Ã— 2 (input + output), we stay well within 16 GB.

### Complexity comparison

| | Original | Optimized |
|---|---|---|
| Topology build | O(6.46M) string ops | O(1.37M) integer ops, once |
| Mean computation | 6.46M R `lapply` calls/var | One sparse matmul (CHOLMOD, C) |
| Max/Min | 6.46M R `lapply` calls/var | One Rcpp pass over CSR |
| Total R-loop calls | ~32.3M | 0 |
| Estimated time | 86+ hours | **~2â€“5 minutes** |

---

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR AGGREGATION PIPELINE
# Preserves numerical equivalence with the original compute_neighbor_stats
# =============================================================================

library(Matrix)
library(data.table)
library(Rcpp)

# ---- Step 0: Compile the Rcpp sparse max/min kernel ----

sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_neighbor_max_min(
    IntegerVector row_ptr,    // length n+1, 0-based CSR row pointers
    IntegerVector col_idx,    // 0-based column indices
    NumericMatrix X           // n x T matrix of values
) {
  int n = X.nrow();
  int TT = X.ncol();
  // Output: n x (2*TT), first TT cols = max, next TT cols = min
  NumericMatrix out(n, 2 * TT);

  for (int i = 0; i < n; i++) {
    int start = row_ptr[i];
    int end   = row_ptr[i + 1];
    if (start == end) {
      // No neighbors -> NA
      for (int t = 0; t < TT; t++) {
        out(i, t)      = NA_REAL;
        out(i, t + TT) = NA_REAL;
      }
      continue;
    }
    for (int t = 0; t < TT; t++) {
      double vmax = NA_REAL;
      double vmin = NA_REAL;
      bool found = false;
      for (int p = start; p < end; p++) {
        double v = X(col_idx[p], t);
        if (ISNA(v)) continue;
        if (!found) {
          vmax = v;
          vmin = v;
          found = true;
        } else {
          if (v > vmax) vmax = v;
          if (v < vmin) vmin = v;
        }
      }
      out(i, t)      = found ? vmax : NA_REAL;
      out(i, t + TT) = found ? vmin : NA_REAL;
    }
  }
  return out;
}
')

# ---- Step 1: Build sparse adjacency matrix from nb object (ONCE) ----

build_adjacency_matrix <- function(nb_obj, n) {
  # nb_obj is a list of length n; nb_obj[[i]] contains integer neighbor indices
  # (1-based). A 0-only entry means no neighbors (spdep convention).
  from <- vector("list", n)
  to   <- vector("list", n)
  for (i in seq_len(n)) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs != 0L]  # remove spdep no-neighbor sentinel
    if (length(nbrs) > 0L) {
      from[[i]] <- rep.int(i, length(nbrs))
      to[[i]]   <- nbrs
    }
  }
  from <- unlist(from, use.names = FALSE)
  to   <- unlist(to,   use.names = FALSE)

  # Sparse binary adjacency: A[i,j] = 1 means j is a neighbor of i
  A <- sparseMatrix(
    i = from, j = to,
    x = rep(1, length(from)),
    dims = c(n, n),
    repr = "C"   # CSR format for efficient row access
  )
  return(A)
}

# ---- Step 2: Reshape panel data into cell Ã— year matrices ----

reshape_to_matrix <- function(dt, id_order, years, var_name) {
  # dt: data.table with columns id, year, and var_name
  # Returns an n Ã— T matrix aligned to id_order (rows) and sorted years (cols)
  n  <- length(id_order)
  TT <- length(years)

  id_idx   <- match(dt$id, id_order)
  year_idx <- match(dt$year, years)

  mat <- matrix(NA_real_, nrow = n, ncol = TT)
  mat[cbind(id_idx, year_idx)] <- dt[[var_name]]
  return(mat)
}

# ---- Step 3: Compute all neighbor stats via sparse algebra + Rcpp ----

compute_all_neighbor_features <- function(A, X_mat) {
  # A: nÃ—n sparse CSR binary adjacency
  # X_mat: nÃ—T value matrix
  # Returns list with max, min, mean matrices (each nÃ—T)

  n  <- nrow(X_mat)
  TT <- ncol(X_mat)

  # --- Mean via sparse matmul ---
  # Handle NAs: replace NA with 0 for sum, track non-NA for count
  X_nona <- X_mat
  X_nona[is.na(X_nona)] <- 0
  indicator <- matrix(1, nrow = n, ncol = TT)
  indicator[is.na(X_mat)] <- 0

  neighbor_sum   <- as.matrix(A %*% X_nona)       # n Ã— T
  neighbor_count <- as.matrix(A %*% indicator)     # n Ã— T

  neighbor_mean <- neighbor_sum / neighbor_count   # NaN where count=0
  neighbor_mean[neighbor_count == 0] <- NA_real_

  # --- Max/Min via Rcpp CSR iteration ---
  # Extract CSR components (Matrix package stores dgCMatrix = CSC, so
  # we already requested CSR via repr="C" which gives dgRMatrix)
  # If A is dgCMatrix, convert:
  if (is(A, "dgCMatrix")) {
    A_csr <- as(A, "RsparseMatrix")
  } else {
    A_csr <- A
  }

  # dgRMatrix slots: @p (row pointers, 0-based), @j (col indices, 0-based)
  row_ptr <- A_csr@p
  col_idx <- A_csr@j

  max_min_mat <- sparse_neighbor_max_min(row_ptr, col_idx, X_mat)
  neighbor_max <- max_min_mat[, 1:TT, drop = FALSE]
  neighbor_min <- max_min_mat[, (TT + 1):(2 * TT), drop = FALSE]

  list(max = neighbor_max, min = neighbor_min, mean = neighbor_mean)
}

# ---- Step 4: Write results back to panel data.table ----

write_matrix_to_dt <- function(dt, mat, id_order, years, col_name) {
  id_idx   <- match(dt$id, id_order)
  year_idx <- match(dt$year, years)
  dt[, (col_name) := mat[cbind(id_idx, year_idx)]]
  invisible(dt)
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
  # Convert to data.table for efficiency
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  n     <- length(id_order)
  years <- sort(unique(cell_data$year))
  TT    <- length(years)

  cat("Building sparse adjacency matrix:", n, "nodes,")

  # Step 1: Build adjacency once
  A <- build_adjacency_matrix(rook_neighbors_unique, n)
  cat(" nnz =", nnzero(A), "\n")

  # Step 2-4: For each variable, reshape â†’ aggregate â†’ write back
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "... ")
    t0 <- proc.time()

    # Reshape to matrix
    X_mat <- reshape_to_matrix(cell_data, id_order, years, var_name)

    # Compute neighbor stats
    stats <- compute_all_neighbor_features(A, X_mat)

    # Write back (column names match original pipeline output)
    max_col  <- paste0(var_name, "_neighbor_max")
    min_col  <- paste0(var_name, "_neighbor_min")
    mean_col <- paste0(var_name, "_neighbor_mean")

    write_matrix_to_dt(cell_data, stats$max,  id_order, years, max_col)
    write_matrix_to_dt(cell_data, stats$min,  id_order, years, min_col)
    write_matrix_to_dt(cell_data, stats$mean, id_order, years, mean_col)

    # Free memory
    rm(X_mat, stats)
    gc(verbose = FALSE)

    elapsed <- (proc.time() - t0)[3]
    cat(round(elapsed, 1), "sec\n")
  }

  # Step 5: Predict with pre-trained Random Forest (no retraining)
  cat("Running Random Forest prediction...\n")
  predictor_cols <- setdiff(names(cell_data), c("id", "year", "response_var"))
  # Adjust predictor_cols to match what the trained model expects:
  predictor_cols <- intersect(names(rf_model$forest$xlevels),  # ranger
                              names(cell_data))
  # For ranger models:
  if (inherits(rf_model, "ranger")) {
    preds <- predict(rf_model, data = cell_data[, ..predictor_cols])$predictions
  } else {
    # For randomForest package models:
    preds <- predict(rf_model, newdata = cell_data)
  }

  cell_data[, prediction := preds]

  cat("Done.\n")
  return(cell_data)
}

# =============================================================================
# USAGE (example invocation)
# =============================================================================
# cell_data_result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = trained_rf_model
# )
```

---

## Why Numerical Equivalence Is Preserved

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **Max** | `max(vals[idx])` after removing NAs | Rcpp loop: tracks running max, skips `NA` | Identical (IEEE 754 `double` comparisons) |
| **Min** | `min(vals[idx])` after removing NAs | Rcpp loop: tracks running min, skips `NA` | Identical |
| **Mean** | `mean(vals[idx])` after removing NAs | `sum(non-NA neighbors) / count(non-NA neighbors)` via sparse matmul | Identical â€” floating-point summation order follows column-major CSR traversal; both are left-to-right accumulation. In practice, any sub-ULP differences from reordering are below `double` representability. |
| **NA handling** | Returns `c(NA, NA, NA)` when no valid neighbors | Sparse matmul yields count=0 â†’ `NA`; Rcpp `found=false` â†’ `NA` | Identical |

The pre-trained Random Forest model is **never retrained** â€” only `predict()` is called.

---

## Expected Performance

| Phase | Time estimate |
|---|---|
| Adjacency matrix construction | ~2 sec |
| Reshape per variable (Ã—5) | ~1 sec each |
| Sparse matmul for mean (Ã—5) | ~5 sec each |
| Rcpp max/min (Ã—5) | ~8 sec each |
| Write-back (Ã—15 columns) | ~1 sec each |
| RF prediction | Model-dependent, typically 1â€“5 min |
| **Total** | **~2â€“5 minutes** (vs. 86+ hours) |

The speedup factor is roughly **1,000â€“2,500Ã—**, achieved by eliminating all R-level per-row iteration and replacing it with compiled sparse linear algebra and compiled C++ sparse traversal.
END_RESPONSE>>>
