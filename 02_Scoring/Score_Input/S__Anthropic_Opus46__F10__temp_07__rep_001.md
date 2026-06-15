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
run_id: Anthropic_F10_T07_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three major bottlenecks:

1. **O(nÂ²)-scale neighbor lookup construction**: `build_neighbor_lookup` iterates over all ~6.46 million rows, and for each row performs string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookups in R are hash-based but still carry significant per-call overhead at this scale. The function builds ~6.46 million list elements, each requiring string concatenation, name matching, and NA filtering.

2. **Repeated R-level `lapply` over 6.46M rows in `compute_neighbor_stats`**: For each of the 5 variables, the code loops over every row in R, subsets a numeric vector by index, removes NAs, and computes max/min/mean. That's ~32.3 million R-level function calls total (6.46M Ã— 5), each with allocation overhead.

3. **The topology is conflated with the temporal dimension**: The rook-neighbor graph is purely spatial (344,208 nodes, ~1.37M directed edges), yet the lookup is rebuilt at the cell-year level (6.46M nodes), inflating the graph ~28Ã—. The spatial adjacency is identical across all years â€” this is pure redundancy.

**Memory**: The 6.46M-element list of integer vectors for `neighbor_lookup` alone consumes several GB due to R list/vector overhead, pushing the 16 GB laptop to its limits.

**Time estimate**: ~86+ hours is consistent with millions of R-level `lapply` iterations involving string operations and named lookups.

---

## Optimization Strategy

### Core Insight
Separate **spatial topology** (344K nodes, ~1.37M edges) from **temporal replication** (28 years). Build the sparse adjacency structure once over cells, then use vectorized/compiled operations to aggregate neighbor attributes year-by-year.

### Specific Techniques

1. **Build a sparse adjacency matrix once** from the `nb` object (344,208 Ã— 344,208 CSC matrix via `spdep::nb2listw` â†’ `Matrix::sparseMatrix`, or directly). This is a one-time cost.

2. **Reshape each variable into a 344,208 Ã— 28 matrix** (cells Ã— years). This allows column-wise (year-wise) sparse matrixâ€“vector operations.

3. **Compute neighbor stats via sparse matrix multiplication and sparse-max/min operations**:
   - **Mean**: `A %*% X / degree` (where `A` is the binary adjacency matrix, `X` is the cellÃ—year matrix, and `degree` is the row-sum vector).
   - **Max and Min**: Use a CSR representation and vectorized row-wise aggregation via compiled C++ code (`Rcpp`) or, without Rcpp, iterate over 344K cells (not 6.46M rows) using the sparse structure.

4. **Avoid string keys entirely**: Use integer cell indices and year indices throughout.

5. **Process year-by-year within each variable** to keep memory bounded.

This reduces the problem from 6.46M R-level iterations to either sparse matrix algebra or 344K iterations (18.7Ã— fewer), with each iteration doing simple numeric operations on ~4 neighbors on average.

---

## Optimized R Code

```r
###############################################################################
# OPTIMIZED SPATIAL NEIGHBOR AGGREGATION PIPELINE
# 
# Preserves numerical equivalence with the original compute_neighbor_stats:
#   - neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
# Preserves the trained Random Forest model (no retraining).
###############################################################################

library(Matrix)
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 0: Convert nb object to sparse adjacency matrix (one-time, ~1.37M edges)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_adjacency_matrix <- function(nb_obj) {
  # nb_obj: list of length n_cells, each element is integer vector of neighbor indices
  # (as produced by spdep::poly2nb or spdep::cell2nb)
  n <- length(nb_obj)
  
  # Build COO triplets
  from <- rep(seq_len(n), times = lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  
  # Remove 0-neighbor entries (spdep uses 0L to indicate no neighbors)
  valid <- to != 0L
  from  <- from[valid]
  to    <- to[valid]
  
  # Sparse binary adjacency matrix (rows = focal cell, cols = neighbor cell)
  A <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n, n),
    repr = "C"   # CSC format; will convert to CSR (dgRMatrix) for row ops
  )
  return(A)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build cell-index and year-index mappings
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

prepare_index_maps <- function(cell_data, id_order) {
  # cell_data must have columns: id, year
  # id_order: the canonical ordering of cell IDs matching the nb object
  
  dt <- as.data.table(cell_data)
  
  # Map cell id -> spatial index (1..n_cells) matching nb object ordering
  cell_map <- data.table(
    id        = id_order,
    cell_idx  = seq_along(id_order)
  )
  
  # Sorted unique years
  years_sorted <- sort(unique(dt$year))
  year_map <- data.table(
    year     = years_sorted,
    year_idx = seq_along(years_sorted)
  )
  
  list(cell_map = cell_map, year_map = year_map, 
       n_cells = length(id_order), n_years = length(years_sorted),
       years = years_sorted)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Reshape a variable into a cell Ã— year matrix
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

variable_to_matrix <- function(cell_data_dt, var_name, cell_map, year_map, 
                                n_cells, n_years) {
  # Extract needed columns
  sub <- cell_data_dt[, .(id, year, val = get(var_name))]
  
  # Merge indices
  sub <- cell_map[sub, on = "id", nomatch = 0L]
  sub <- year_map[sub, on = "year", nomatch = 0L]
  
  # Fill matrix (NA for missing cell-year combinations)
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(sub$cell_idx, sub$year_idx)] <- sub$val
  
  return(mat)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Compute neighbor max, min, mean using sparse adjacency
#          This is the performance-critical function.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_sparse <- function(A_csc, val_mat, n_cells, n_years) {
  # A_csc: dgCMatrix (CSC) adjacency matrix, n_cells x n_cells
  # val_mat: n_cells x n_years numeric matrix
  #
  # Returns list of three matrices (n_cells x n_years): max, min, mean
  # Numerically equivalent to original per-row neighbor aggregation.
  
  # Convert to dgRMatrix (CSR) for efficient row-wise access
  A_csr <- as(A_csc, "RsparseMatrix")
  
  # CSR components (0-indexed in Matrix package)
  row_ptr <- A_csr@p          # length n_cells + 1
  col_idx <- A_csr@j          # 0-indexed column indices
  # A_csr@x are all 1s (binary adjacency)
  
  # Pre-allocate output matrices
  max_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # --- Vectorized approach: process one year at a time ---
  # For each year-column, we need row-wise max, min, mean of A[i, ] applied to vals.
  # 
  # Mean is easy: (A %*% val_col) / degree, but must handle NAs properly
  # (original code drops NAs before computing stats).
  #
  # For full NA-aware max/min/mean matching the original:
  # We iterate over cells using the CSR structure.
  
  for (yr in seq_len(n_years)) {
    v <- val_mat[, yr]  # length n_cells
    
    for (i in seq_len(n_cells)) {
      start <- row_ptr[i] + 1L    # convert 0-indexed to 1-indexed
      end   <- row_ptr[i + 1L]    # 0-indexed end is exclusive, so this is correct
      
      if (end < start) {
        # No neighbors
        next  # already NA
      }
      
      nb_indices <- col_idx[start:end] + 1L  # convert 0-indexed to 1-indexed
      nb_vals    <- v[nb_indices]
      nb_vals    <- nb_vals[!is.na(nb_vals)]
      
      if (length(nb_vals) == 0L) next
      
      max_mat[i, yr]  <- max(nb_vals)
      min_mat[i, yr]  <- min(nb_vals)
      mean_mat[i, yr] <- mean(nb_vals)
    }
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3-ALT: Rcpp version for maximum speed (RECOMMENDED)
# If Rcpp is available, this replaces the R loop above.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

use_rcpp <- requireNamespace("Rcpp", quietly = TRUE) && 
            requireNamespace("RcppArmadillo", quietly = TRUE)

if (use_rcpp) {
  
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List neighbor_stats_csr(IntegerVector row_ptr,   // length n_cells+1, 0-indexed
                        IntegerVector col_idx,   // 0-indexed neighbor columns
                        NumericMatrix val_mat,    // n_cells x n_years
                        int n_cells, int n_years) {
  
  NumericMatrix max_mat(n_cells, n_years);
  NumericMatrix min_mat(n_cells, n_years);
  NumericMatrix mean_mat(n_cells, n_years);
  
  // Initialize to NA
  std::fill(max_mat.begin(),  max_mat.end(),  NA_REAL);
  std::fill(min_mat.begin(),  min_mat.end(),  NA_REAL);
  std::fill(mean_mat.begin(), mean_mat.end(), NA_REAL);
  
  for (int yr = 0; yr < n_years; yr++) {
    for (int i = 0; i < n_cells; i++) {
      int start = row_ptr[i];
      int end   = row_ptr[i + 1];
      
      if (start == end) continue;  // no neighbors
      
      double vmax = R_NegInf;
      double vmin = R_PosInf;
      double vsum = 0.0;
      int count = 0;
      
      for (int k = start; k < end; k++) {
        int j = col_idx[k];  // 0-indexed
        double val = val_mat(j, yr);
        if (R_IsNA(val)) continue;
        if (val > vmax) vmax = val;
        if (val < vmin) vmin = val;
        vsum += val;
        count++;
      }
      
      if (count > 0) {
        max_mat(i, yr)  = vmax;
        min_mat(i, yr)  = vmin;
        mean_mat(i, yr) = vsum / (double)count;
      }
    }
  }
  
  return List::create(Named("max")  = max_mat,
                      Named("min")  = min_mat,
                      Named("mean") = mean_mat);
}
')
  
  compute_neighbor_stats_sparse <- function(A_csc, val_mat, n_cells, n_years) {
    A_csr   <- as(A_csc, "RsparseMatrix")
    row_ptr <- A_csr@p        # 0-indexed, length n_cells + 1
    col_j   <- A_csr@j        # 0-indexed
    neighbor_stats_csr(row_ptr, col_j, val_mat, n_cells, n_years)
  }
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Write aggregated stats back to cell_data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

write_stats_to_data <- function(cell_data_dt, stats, var_name,
                                 cell_map, year_map, n_cells, n_years) {
  # stats: list with $max, $min, $mean, each n_cells x n_years matrix
  # Flatten back to the row order of cell_data_dt
  
  # Build row-index into matrices
  sub <- cell_data_dt[, .(row_id = .I, id, year)]
  sub <- cell_map[sub, on = "id", nomatch = 0L]
  sub <- year_map[sub, on = "year", nomatch = 0L]
  
  mat_idx <- cbind(sub$cell_idx, sub$year_idx)
  
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  set(cell_data_dt, j = max_col,  value = NA_real_)
  set(cell_data_dt, j = min_col,  value = NA_real_)
  set(cell_data_dt, j = mean_col, value = NA_real_)
  
  set(cell_data_dt, i = sub$row_id, j = max_col,  value = stats$max[mat_idx])
  set(cell_data_dt, i = sub$row_id, j = min_col,  value = stats$min[mat_idx])
  set(cell_data_dt, i = sub$row_id, j = mean_col, value = stats$mean[mat_idx])
  
  invisible(cell_data_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: MAIN PIPELINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                    rf_model) {
  
  cat("=== Optimized Neighbor Aggregation Pipeline ===\n")
  
  # 1. Build sparse adjacency matrix ONCE (344K x 344K, ~1.37M nonzeros)
  cat("[1/5] Building sparse adjacency matrix...\n")
  t0 <- proc.time()
  A <- build_adjacency_matrix(rook_neighbors_unique)
  cat("      Done:", round((proc.time() - t0)[3], 1), "sec\n")
  cat("      Dimensions:", nrow(A), "x", ncol(A), 
      " | Nonzeros:", nnzero(A), "\n")
  
  # 2. Prepare index mappings
  cat("[2/5] Preparing index maps...\n")
  t0 <- proc.time()
  cell_data_dt <- as.data.table(cell_data)
  maps <- prepare_index_maps(cell_data_dt, id_order)
  cat("      Done:", round((proc.time() - t0)[3], 1), "sec\n")
  cat("      Cells:", maps$n_cells, "| Years:", maps$n_years, 
      "| Rows:", nrow(cell_data_dt), "\n")
  
  # 3. For each variable: reshape â†’ aggregate â†’ write back
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  cat("[3/5] Computing neighbor statistics for", length(neighbor_source_vars), 
      "variables...\n")
  
  for (var_name in neighbor_source_vars) {
    cat("      Processing:", var_name, "... ")
    t0 <- proc.time()
    
    # Reshape to cell Ã— year matrix
    val_mat <- variable_to_matrix(
      cell_data_dt, var_name,
      maps$cell_map, maps$year_map,
      maps$n_cells, maps$n_years
    )
    
    # Compute neighbor max, min, mean via sparse CSR traversal
    stats <- compute_neighbor_stats_sparse(A, val_mat, maps$n_cells, maps$n_years)
    
    # Write results back to data.table
    write_stats_to_data(
      cell_data_dt, stats, var_name,
      maps$cell_map, maps$year_map,
      maps$n_cells, maps$n_years
    )
    
    # Free intermediate matrices
    rm(val_mat, stats)
    
    elapsed <- round((proc.time() - t0)[3], 1)
    cat(elapsed, "sec\n")
  }
  
  # 4. Predict using the pre-trained Random Forest (NO retraining)
  cat("[4/5] Generating predictions with pre-trained Random Forest...\n")
  t0 <- proc.time()
  cell_data_dt$prediction <- predict(rf_model, newdata = cell_data_dt)
  cat("      Done:", round((proc.time() - t0)[3], 1), "sec\n")
  
  # 5. Return result
  cat("[5/5] Converting back to data.frame...\n")
  result <- as.data.frame(cell_data_dt)
  
  cat("=== Pipeline complete ===\n")
  return(result)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# USAGE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 
# # Load pre-trained model and data
# rf_model              <- readRDS("trained_rf_model.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors_unique.rds")
# cell_data             <- readRDS("cell_data.rds")       # data.frame with ~6.46M rows
# id_order              <- readRDS("id_order.rds")         # canonical cell ID vector
# 
# # Run optimized pipeline
# result <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique, rf_model)


###############################################################################
# VERIFICATION: Confirm numerical equivalence with original implementation
###############################################################################

verify_equivalence <- function(cell_data_original, cell_data_optimized, 
                                neighbor_source_vars, tol = 1e-10) {
  cat("Verifying numerical equivalence...\n")
  all_pass <- TRUE
  
  for (var_name in neighbor_source_vars) {
    for (stat in c("max", "min", "mean")) {
      col <- paste0("neighbor_", stat, "_", var_name)
      
      orig <- cell_data_original[[col]]
      optm <- cell_data_optimized[[col]]
      
      # Both NA in same positions
      na_match <- all(is.na(orig) == is.na(optm))
      
      # Non-NA values match within tolerance
      valid <- !is.na(orig) & !is.na(optm)
      val_match <- all(abs(orig[valid] - optm[valid]) < tol)
      
      pass <- na_match && val_match
      if (!pass) all_pass <- FALSE
      
      cat(sprintf("  %-35s %s  (NA match: %s, val match: %s, n_valid: %d)\n",
                  col, ifelse(pass, "PASS", "FAIL"), na_match, val_match, sum(valid)))
    }
  }
  
  cat(ifelse(all_pass, "\nAll checks PASSED.\n", "\nSome checks FAILED!\n"))
  invisible(all_pass)
}
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Topology size** | 6.46M list elements (cell-year) | 344K Ã— 344K sparse matrix (~1.37M entries) |
| **Lookup construction** | String paste + named vector match over 6.46M rows | One-time `sparseMatrix()` call |
| **Aggregation loop** | 6.46M R-level iterations Ã— 5 vars = 32.3M calls | 344K cells Ã— 28 years Ã— 5 vars in compiled C++ (Rcpp), or 344K Ã— 28 Ã— 5 in R |
| **Memory for topology** | ~4â€“6 GB (R list overhead) | ~22 MB (sparse matrix) |
| **Estimated time (Rcpp)** | 86+ hours | **~5â€“15 minutes** |
| **Estimated time (pure R)** | 86+ hours | **~30â€“90 minutes** |
| **Numerical equivalence** | Baseline | Identical (same max/min/mean, same NA handling) |
| **RF model** | Unchanged | Unchanged â€” predict only |

The key speedup comes from (1) eliminating string operations entirely, (2) reducing the iteration space from 6.46M to 344K by factoring out the time dimension, (3) using compiled C++ via Rcpp for the inner loop, and (4) replacing R list-of-vectors topology (~6 GB) with a sparse matrix (~22 MB).
END_RESPONSE>>>
