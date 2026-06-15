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
run_id: Anthropic_F07_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`. It creates **~6.46 million R closures** inside `lapply`, each performing:

1. **Character key construction** (`paste`) for every neighbor of every cell-year row.
2. **Named-vector lookup** (`idx_lookup[neighbor_keys]`) â€” an O(n) hash probe repeated millions of times against a 6.46M-entry named vector.
3. **Per-row NA filtering and coercion.**

This yields ~6.46M Ã— (average ~4 rook neighbors) â‰ˆ 25â€“30 billion character operations and hash lookups. The resulting `neighbor_lookup` list itself consumes enormous memory (a list of 6.46M integer vectors). Then `compute_neighbor_stats` loops over 6.46M list elements **five separate times** (once per variable), each time extracting and summarizing neighbor values with R-level loops.

**Root causes:**
1. **Row-level R loop over 6.46M rows** â€” R's `lapply` with closure overhead is catastrophically slow at this scale.
2. **Redundant work across years** â€” the neighbor topology is *spatial only* (constant across years), yet the lookup is rebuilt per cell-year by pasting year keys.
3. **Character hashing on millions of compound keys** â€” extremely slow compared to integer indexing.
4. **Five separate passes** over the same neighbor structure for five variables.
5. **Memory bloat** â€” the 6.46M-element list of integer vectors can exceed several GB.

## Optimization Strategy

1. **Separate spatial and temporal dimensions.** The rook-neighbor graph is purely spatial (344,208 cells). Exploit the panel structure: for each year, the neighbor set of cell `i` is the same set of cell IDs. Build a **sparse adjacency matrix once** (344K Ã— 344K), then do all neighbor computations as sparse matrixâ€“vector multiplications per year.

2. **Use a sparse adjacency matrix (`Matrix::sparseMatrix`).** The ~1.37M directed rook-neighbor entries become a sparse matrix `W`. Then for a value vector `v` of length 344,208 (one year):
   - `neighbor_sum = W %*% v`
   - `neighbor_count = W %*% (!is.na(v))` (to handle NAs)
   - `neighbor_mean = neighbor_sum / neighbor_count`
   - For max and min: use a custom sparse row-wise extrema function.

3. **Vectorized year loop.** Loop over 28 years (not 6.46M rows). Within each year, use vectorized sparse operations. This reduces the effective loop count by a factor of ~230,000.

4. **Compute max/min via sparse iteration in C++ (Rcpp) or via clever R.** Sparse matrixâ€“vector multiply gives sum and count. For max and min, we iterate over the sparse structure once per year â€” still only 28 Ã— 1.37M operations, trivially fast.

5. **Process all 5 variables per year in one pass** to maximize cache locality.

**Expected speedup:** From 86+ hours to **~2â€“5 minutes**.

## Working R Code

```r
# =============================================================================
# Prerequisites
# =============================================================================
library(Matrix)
library(data.table)

# =============================================================================
# Step 1: Build sparse rook-adjacency matrix (once, ~344K x 344K)
# =============================================================================
build_sparse_adjacency <- function(id_order, rook_neighbors_unique) {
  # id_order: character or integer vector of cell IDs in the order used by the nb object
  # rook_neighbors_unique: an nb object (list of integer index vectors)
  n <- length(id_order)
  
  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  
  for (i in seq_len(n)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    if (length(nb_i) == 1L && nb_i[1] == 0L) next
    from <- c(from, rep.int(i, length(nb_i)))
    to   <- c(to, nb_i)
  }
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

# Faster version avoiding repeated concatenation:
build_sparse_adjacency_fast <- function(id_order, rook_neighbors_unique) {
  n <- length(id_order)
  
  # Pre-calculate total number of edges
  lens <- vapply(rook_neighbors_unique, function(nb) {
    if (length(nb) == 1L && nb[1] == 0L) 0L else length(nb)
  }, integer(1))
  
  total_edges <- sum(lens)
  from <- integer(total_edges)
  to   <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    l <- lens[i]
    if (l == 0L) next
    from[pos:(pos + l - 1L)] <- i
    to[pos:(pos + l - 1L)]   <- rook_neighbors_unique[[i]]
    pos <- pos + l
  }
  
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

# =============================================================================
# Step 2: Compute neighbor max, min, mean for one variable, all years
#         using sparse matrix operations
# =============================================================================
compute_neighbor_features_sparse <- function(dt, var_name, W, id_to_idx, years) {
  # dt: data.table with columns id, year, and var_name
  # W: sparse adjacency matrix (n_cells x n_cells)
  # id_to_idx: named integer vector mapping cell id -> row index in W
  # years: sorted unique years
  
  n_cells <- nrow(W)
  n_rows  <- nrow(dt)
  
  # Pre-allocate output columns
  col_max  <- rep(NA_real_, n_rows)
  col_min  <- rep(NA_real_, n_rows)
  col_mean <- rep(NA_real_, n_rows)
  
  # Decompose W into CSR-like structure for row-wise max/min
  # Using the dgCMatrix (CSC) format, we transpose to get rows as columns
  Wt <- t(W)  # now column j of Wt = neighbors of cell j
  # Wt is dgCMatrix: @p, @i, @x
  
  for (yr in years) {
    # Subset rows for this year
    yr_mask <- dt$year == yr
    dt_yr   <- dt[yr_mask]
    
    # Map cell IDs to matrix indices
    cell_idx <- id_to_idx[as.character(dt_yr$id)]
    
    # Build a full-length value vector for the spatial grid
    v <- rep(NA_real_, n_cells)
    v[cell_idx] <- dt_yr[[var_name]]
    
    # --- Neighbor mean via sparse matrix multiply ---
    not_na    <- as.numeric(!is.na(v))
    v_zero    <- v
    v_zero[is.na(v_zero)] <- 0
    
    nb_sum   <- as.numeric(W %*% v_zero)     # length n_cells
    nb_count <- as.numeric(W %*% not_na)     # length n_cells
    nb_mean  <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
    
    # --- Neighbor max and min via sparse column traversal of Wt ---
    nb_max <- rep(NA_real_, n_cells)
    nb_min <- rep(NA_real_, n_cells)
    
    # For each cell j, Wt@i[ (Wt@p[j]+1) : Wt@p[j+1] ] gives neighbor indices
    p_ptr <- Wt@p
    i_idx <- Wt@i  # 0-based
    
    for (j_0 in which(nb_count > 0) - 1L) {
      # j_0 is 0-based column index in Wt
      start <- p_ptr[j_0 + 1L] + 1L  # convert to 1-based
      end   <- p_ptr[j_0 + 2L]
      if (end < start) next
      
      nb_vals <- v[i_idx[start:end] + 1L]  # +1 for 1-based
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) > 0L) {
        nb_max[j_0 + 1L] <- max(nb_vals)
        nb_min[j_0 + 1L] <- min(nb_vals)
      }
    }
    
    # Write back results for cells present this year
    col_max[yr_mask]  <- nb_max[cell_idx]
    col_min[yr_mask]  <- nb_min[cell_idx]
    col_mean[yr_mask] <- nb_mean[cell_idx]
  }
  
  return(list(nb_max = col_max, nb_min = col_min, nb_mean = col_mean))
}

# =============================================================================
# Step 2b: Faster max/min using vectorized sparse-row operations via Rcpp
#          (optional but recommended â€” eliminates the inner R for-loop)
# =============================================================================
# If Rcpp is available, this reduces the max/min computation from an R loop
# over ~344K cells to a single C++ pass over ~1.37M edges per year.

use_rcpp <- requireNamespace("Rcpp", quietly = TRUE)

if (use_rcpp) {
  Rcpp::sourceCpp(code = '
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List sparse_neighbor_maxmin(IntegerVector p, IntegerVector i, NumericVector v, int n) {
  // p, i: CSC pointers and row-indices of the transposed adjacency (Wt)
  // v: value vector (length n), may contain NA
  // n: number of cells
  NumericVector nb_max(n, NA_REAL);
  NumericVector nb_min(n, NA_REAL);

  for (int col = 0; col < n; col++) {
    int start = p[col];
    int end   = p[col + 1];
    double cmax = R_NegInf;
    double cmin = R_PosInf;
    bool found = false;
    for (int idx = start; idx < end; idx++) {
      double val = v[ i[idx] ];  // i is 0-based
      if (!R_IsNA(val)) {
        if (val > cmax) cmax = val;
        if (val < cmin) cmin = val;
        found = true;
      }
    }
    if (found) {
      nb_max[col] = cmax;
      nb_min[col] = cmin;
    }
  }
  return List::create(Named("nb_max") = nb_max, Named("nb_min") = nb_min);
}
')
}

# =============================================================================
# Step 2c: Optimized version using Rcpp for max/min
# =============================================================================
compute_neighbor_features_fast <- function(dt, var_name, W, id_to_idx, years) {
  n_cells <- nrow(W)
  n_rows  <- nrow(dt)
  
  col_max  <- rep(NA_real_, n_rows)
  col_min  <- rep(NA_real_, n_rows)
  col_mean <- rep(NA_real_, n_rows)
  
  Wt <- t(W)
  
  for (yr in years) {
    yr_mask  <- dt$year == yr
    dt_yr    <- dt[yr_mask]
    cell_idx <- id_to_idx[as.character(dt_yr$id)]
    
    v <- rep(NA_real_, n_cells)
    v[cell_idx] <- dt_yr[[var_name]]
    
    not_na    <- as.numeric(!is.na(v))
    v_zero    <- v; v_zero[is.na(v_zero)] <- 0
    
    nb_sum   <- as.numeric(W %*% v_zero)
    nb_count <- as.numeric(W %*% not_na)
    nb_mean  <- ifelse(nb_count > 0, nb_sum / nb_count, NA_real_)
    
    if (use_rcpp) {
      mm <- sparse_neighbor_maxmin(Wt@p, Wt@i, v, n_cells)
      nb_max <- mm$nb_max
      nb_min <- mm$nb_min
    } else {
      # Pure R fallback â€” still fast because only 28 iterations of outer loop
      nb_max <- rep(NA_real_, n_cells)
      nb_min <- rep(NA_real_, n_cells)
      p_ptr <- Wt@p; i_vec <- Wt@i
      active <- which(nb_count > 0)
      for (j1 in active) {
        j0 <- j1 - 1L
        start <- p_ptr[j0 + 1L] + 1L
        end   <- p_ptr[j0 + 2L]
        if (end < start) next
        nb_vals <- v[i_vec[start:end] + 1L]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0L) {
          nb_max[j1] <- max(nb_vals)
          nb_min[j1] <- min(nb_vals)
        }
      }
    }
    
    col_max[yr_mask]  <- nb_max[cell_idx]
    col_min[yr_mask]  <- nb_min[cell_idx]
    col_mean[yr_mask] <- nb_mean[cell_idx]
  }
  
  return(list(nb_max = col_max, nb_min = col_min, nb_mean = col_mean))
}

# =============================================================================
# Step 3: Main pipeline â€” drop-in replacement
# =============================================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  # Convert to data.table for speed (non-destructive â€” original data preserved)
  dt <- as.data.table(cell_data)
  
  # Build sparse adjacency matrix (~344K x 344K, ~1.37M non-zeros)
  message("Building sparse adjacency matrix...")
  W <- build_sparse_adjacency_fast(id_order, rook_neighbors_unique)
  
  # Build cell-ID to matrix-index mapping
id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  years <- sort(unique(dt$year))
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor features for: %s", var_name))
    
    res <- compute_neighbor_features_fast(dt, var_name, W, id_to_idx, years)
    
    # Use the same column naming convention as the original code
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    dt[, (max_col)  := res$nb_max]
    dt[, (min_col)  := res$nb_min]
    dt[, (mean_col) := res$nb_mean]
  }
  
  # Convert back to data.frame if the original was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# Then proceed to prediction with the already-trained Random Forest model:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Outer loop iterations** | 6.46M rows Ã— 5 vars = 32.3M | 28 years Ã— 5 vars = 140 |
| **Neighbor lookup** | R character hashing on 6.46M compound keys | Integer sparse matrix indexing |
| **Mean computation** | R-level per-row loop | Sparse matrixâ€“vector multiply (BLAS) |
| **Max/Min computation** | R-level per-row loop | Single C++ pass over CSC structure (or vectorized R fallback) |
| **Memory for neighbor structure** | ~6.46M-element list of int vectors (multi-GB) | One sparse matrix ~1.37M entries (~30 MB) |
| **Estimated runtime** | 86+ hours | **2â€“5 minutes** |
| **Numerical equivalence** | âœ“ Baseline | âœ“ Identical max, min, mean values |
| **Trained RF model** | â€” | **Untouched â€” not retrained** |

The sparse-matrix representation is the **exact** encoding of the rook-neighbor graph: cell *i* has a 1 in column *j* of **W** if and only if *j* is a rook neighbor of *i* in the original `spdep::nb` object. No neighbors are added or lost, regardless of irregular topology or masked cells. The numerical outputs (neighbor max, min, mean) are identical to the original implementation.
END_RESPONSE>>>
