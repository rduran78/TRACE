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
run_id: Anthropic_F10_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries** by iterating row-by-row through every cell-year observation. Since the rook-neighbor topology is *time-invariant* (the same 344,208 cells have the same neighbors every year), this duplicates the spatial graph 28 times. The `paste()`/`setNames()`/lookup pattern for ~6.46M rows is extremely expensive in R.

2. **`compute_neighbor_stats` iterates over ~6.46M list elements per variable** (5 variables Ã— 6.46M = ~32.3M R-level `lapply` iterations), each doing subsetting, `is.na` filtering, and three summary statistics. R's interpreted loop overhead on 32M iterations is enormous.

3. **Memory pressure**: storing a 6.46M-element list of integer vectors (the neighbor lookup) plus intermediate copies of the data frame causes heavy GC thrashing on a 16 GB machine.

**Root cause**: The code treats the problem as a *cell-year* graph (~6.46M nodes, ~38M directed edges after temporal replication) instead of a *cell* graph (344,208 nodes, ~1.37M directed edges) applied independently within each year slice.

## Optimization Strategy

1. **Build the sparse adjacency structure once at the cell level** (344,208 Ã— 344,208) using a sparse matrix (`Matrix::dgCMatrix`). This encodes all ~1.37M directed rook edges.

2. **Reshape each variable into a 344,208 Ã— 28 dense matrix** (cells Ã— years). This allows vectorized column-wise (per-year) sparse matrixâ€“dense vector operations.

3. **Compute neighbor statistics via sparse matrix algebra**:
   - **Mean**: `A %*% X / degree` (where `A` is the adjacency matrix, `X` is the attribute matrix, and `degree` is the row-sum of valid neighbors per cell-year).
   - **Max and Min**: Use a single pass over the CSR structure in C++ (via `Rcpp`) to compute row-wise max/min of neighbor values. This avoids R-level loops entirely.

4. **Handle NAs correctly**: Mask NAs so they don't contaminate sums/counts, and return NA for cells with zero valid neighbors (preserving numerical equivalence).

5. **Write results back** to the long-format `cell_data` data frame, with identical column names, so the pre-trained Random Forest model's `predict()` call works unchanged.

**Expected speedup**: From ~86+ hours to **~2â€“5 minutes**. The sparse matrixâ€“vector product for mean is O(nnz) â‰ˆ 1.37M per year per variable. The Rcpp row-wise max/min is the same complexity. Total: 5 vars Ã— 28 years Ã— 3 stats Ã— O(1.37M) â‰ˆ 576M simple operations â€” trivial for modern hardware.

## Working R Code

```r
# ==============================================================================
# Prerequisites
# ==============================================================================
# install.packages(c("Matrix", "data.table", "Rcpp"))
library(Matrix)
library(data.table)
library(Rcpp)

# ==============================================================================
# Step 0: Rcpp function for sparse row-wise max and min (handles NAs)
# ==============================================================================
Rcpp::sourceCpp(code = '
#include <Rcpp.h>
#include <limits>
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix sparse_row_maxmin(IntegerVector p, IntegerVector j,
                                NumericVector x, int nrow) {
  // p: CSR row pointers (length nrow+1), j: column indices, x: values
  // Returns nrow x 2 matrix: col0 = max, col1 = min
  NumericMatrix out(nrow, 2);
  for (int i = 0; i < nrow; i++) {
    double rmax = NA_REAL;
    double rmin = NA_REAL;
    bool found = false;
    for (int k = p[i]; k < p[i + 1]; k++) {
      double val = x[k];
      if (!ISNA(val) && !ISNAN(val)) {
        if (!found) {
          rmax = val;
          rmin = val;
          found = true;
        } else {
          if (val > rmax) rmax = val;
          if (val < rmin) rmin = val;
        }
      }
    }
    out(i, 0) = found ? rmax : NA_REAL;
    out(i, 1) = found ? rmin : NA_REAL;
  }
  return out;
}
')

# ==============================================================================
# Step 1: Build the spatial adjacency matrix ONCE (cell-level, time-invariant)
# ==============================================================================
build_adjacency_matrix <- function(id_order, rook_neighbors) {
  # id_order: vector of cell IDs in the order used by the nb object
  # rook_neighbors: spdep nb object (list of integer index vectors)
  n <- length(id_order)
  
  # Build COO triplets
  from_idx <- integer(0)
  to_idx   <- integer(0)
  
  for (i in seq_len(n)) {
    nb <- rook_neighbors[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb <- nb[nb != 0L]
    if (length(nb) > 0L) {
      from_idx <- c(from_idx, rep(i, length(nb)))
      to_idx   <- c(to_idx, nb)
    }
  }
  
  # Sparse adjacency matrix (row i has 1s in columns that are i's neighbors)
  A <- sparseMatrix(
    i = from_idx, j = to_idx, x = 1,
    dims = c(n, n), repr = "C"   # CSC format for Matrix package
  )
  
  return(A)
}

cat("Building adjacency matrix...\n")
A <- build_adjacency_matrix(id_order, rook_neighbors_unique)
n_cells <- length(id_order)
cat(sprintf("  Adjacency matrix: %d x %d, %d nonzeros\n",
            nrow(A), ncol(A), nnz(A)))

# ==============================================================================
# Step 2: Create cell-index and year-index mappings
# ==============================================================================
cat("Building index mappings...\n")

# Convert to data.table for speed (non-destructive; keeps cell_data intact)
dt <- as.data.table(cell_data)

# Map cell IDs to matrix row indices (1-based, matching id_order)
id_to_row <- setNames(seq_along(id_order), as.character(id_order))
dt[, cell_idx := id_to_row[as.character(id)]]

# Sorted unique years
years <- sort(unique(dt$year))
n_years <- length(years)
year_to_col <- setNames(seq_along(years), as.character(years))
dt[, year_idx := year_to_col[as.character(year)]]

# Keep track of original row order for writing back
dt[, orig_row := .I]

# ==============================================================================
# Step 3: Convert CSC adjacency to CSR for Rcpp row-wise operations
# ==============================================================================
# Matrix package stores dgCMatrix in CSC. We need CSR (dgRMatrix) for row ops.
# Transpose CSC gives us CSR of the original matrix's rows as columns of A^T.
# Actually, easier: convert to dgRMatrix directly.

A_csr <- as(A, "RsparseMatrix")  # dgRMatrix: CSR format

# ==============================================================================
# Step 4: For each variable, compute neighbor max, min, mean
# ==============================================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_features <- function(dt, A, A_csr, id_to_row,
                                          years, n_cells, n_years,
                                          var_names) {
  
  for (var_name in var_names) {
    cat(sprintf("  Processing variable: %s\n", var_name))
    
    # ------------------------------------------------------------------
    # 4a. Pivot variable into a dense matrix: cells (rows) x years (cols)
    # ------------------------------------------------------------------
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(dt$cell_idx, dt$year_idx)] <- dt[[var_name]]
    
    # ------------------------------------------------------------------
    # 4b. Compute MEAN via sparse matrix multiplication
    #     For each year column v: A %*% v gives sum of neighbor values
    #     We need to divide by the count of non-NA neighbors.
    # ------------------------------------------------------------------
    # Build a non-NA indicator matrix
    W <- matrix(0, nrow = n_cells, ncol = n_years)
    W[!is.na(V)] <- 1
    
    # Replace NAs with 0 for matrix multiplication (so they don't propagate)
    V0 <- V
    V0[is.na(V0)] <- 0
    
    # Neighbor sums and counts (sparse %*% dense is efficient in Matrix pkg)
    neighbor_sum   <- as.matrix(A %*% V0)        # n_cells x n_years
    neighbor_count <- as.matrix(A %*% W)          # n_cells x n_years
    
    # Mean (NA where count == 0)
    neighbor_mean <- neighbor_sum / neighbor_count
    neighbor_mean[neighbor_count == 0] <- NA_real_
    
    # ------------------------------------------------------------------
    # 4c. Compute MAX and MIN via Rcpp CSR row-wise pass (per year)
    # ------------------------------------------------------------------
    # A_csr@p: row pointers (0-based, length n_cells+1)
    # A_csr@j: column indices (0-based)
    # For each year, gather neighbor values and call C++
    
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    csr_p <- A_csr@p    # integer, 0-based row pointers
    csr_j <- A_csr@j    # integer, 0-based column indices
    
    for (t in seq_len(n_years)) {
      v_year <- V[, t]  # length n_cells, may contain NA
      
      # Build the values vector aligned with CSR structure:
      # For each nonzero entry (row i -> col j), the value is v_year[j+1]
      vals_for_csr <- v_year[csr_j + 1L]  # R is 1-based, csr_j is 0-based
      
      # Call Rcpp (pass 0-based pointers as-is; function expects them)
      mm <- sparse_row_maxmin(csr_p, csr_j, vals_for_csr, n_cells)
      neighbor_max[, t] <- mm[, 1]
      neighbor_min[, t] <- mm[, 2]
    }
    
    # ------------------------------------------------------------------
    # 4d. Write results back to dt using (cell_idx, year_idx) mapping
    # ------------------------------------------------------------------
    idx_mat <- cbind(dt$cell_idx, dt$year_idx)
    
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    set(dt, j = max_col,  value = neighbor_max[idx_mat])
    set(dt, j = min_col,  value = neighbor_min[idx_mat])
    set(dt, j = mean_col, value = neighbor_mean[idx_mat])
    
    cat(sprintf("    -> Added %s, %s, %s\n", max_col, min_col, mean_col))
  }
  
  return(dt)
}

cat("Computing neighbor features...\n")
system.time({
  dt <- compute_all_neighbor_features(
    dt, A, A_csr, id_to_row,
    years, n_cells, n_years,
    neighbor_source_vars
  )
})

# ==============================================================================
# Step 5: Write back to cell_data (preserving original row order & all columns)
# ==============================================================================
cat("Writing features back to cell_data...\n")

# Ensure original row ordering
setorder(dt, orig_row)

# Add the 15 new columns to cell_data
new_cols <- paste0("neighbor_", rep(c("max_", "min_", "mean_"), times = 5),
                   rep(neighbor_source_vars, each = 3))

for (col in new_cols) {
  cell_data[[col]] <- dt[[col]]
}

# Clean up helper columns (don't pass to model)
# dt[, c("cell_idx", "year_idx", "orig_row") := NULL]

cat("Done. Neighbor features added to cell_data.\n")

# ==============================================================================
# Step 6: Predict with the pre-trained Random Forest (unchanged)
# ==============================================================================
# The model object (e.g., `rf_model`) is already in memory.
# Column names match what the model expects.
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Graph representation** | 6.46M-element R list | 344K Ã— 344K sparse matrix (1.37M nnz) |
| **Neighbor lookup** | Rebuilt per cell-year via `paste`/`setNames` | Built once; reused across all years and variables |
| **Mean computation** | R `lapply` over 6.46M entries | Single sparse matrix Ã— dense matrix multiply |
| **Max/Min computation** | R `lapply` with per-element subsetting | Rcpp CSR row-wise pass (28 iterations, one per year) |
| **Total R-level iterations** | ~32.3M (`lapply` calls) | 140 (28 years Ã— 5 vars for max/min loop) |
| **Estimated runtime** | 86+ hours | **~2â€“5 minutes** |
| **Peak memory** | Heavy (6.46M-element list + copies) | ~1.5 GB (sparse matrix + one 344KÃ—28 dense matrix at a time) |
| **Numerical equivalence** | Baseline | âœ… Identical `max`, `min`, `mean` with same NA handling |
| **Random Forest model** | â€” | âœ… Untouched; same column names, same predict call |
END_RESPONSE>>>
