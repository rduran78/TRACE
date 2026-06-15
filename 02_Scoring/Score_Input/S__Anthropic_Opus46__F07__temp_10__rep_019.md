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
run_id: Anthropic_F07_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Neighbor Statistics Computation

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` uses `lapply` over ~6.46 million rows**, creating a list of 6.46M integer vectors. For each row, it performs string pasting, hash lookups, and NA filtering. This is the dominant bottleneck â€” the `paste`/`setNames`/`idx_lookup` pattern is executed per-row in pure R.

2. **`compute_neighbor_stats` iterates over 6.46M list elements**, subscripting a numeric vector and computing `max`/`min`/`mean` per element. While each operation is small, the overhead of 6.46M R function calls is enormous.

3. **Memory**: Storing 6.46M integer vectors in a list is wasteful. A flat sparse representation is far more efficient.

**Root cause**: The code treats a *cell-level* spatial problem (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M edges) as a *row-level* problem (6.46M rows), redundantly replicating the neighbor graph across every year.

## Optimization Strategy

1. **Separate the spatial topology from the panel expansion.** The neighbor graph has only 344K nodes and ~1.37M directed edges â€” this is tiny. Build a sparse adjacency matrix once over cells, then use matrix operations across all years simultaneously.

2. **Use a sparse adjacency matrix (`Matrix::sparseMatrix`)** to represent the rook-neighbor graph. Row-normalize it for means; use it directly for max/min.

3. **Reshape the variable into a cell Ã— year matrix**, then compute neighbor stats via sparse matrix multiplication (for mean) and row-wise sparse operations (for max/min). This replaces 6.46M R-level iterations with a handful of vectorized sparse matrix operations.

4. **For max and min**, iterate over 344K cells (not 6.46M rows) using the `nb` object directly â€” still fast because it's 50Ã— fewer iterations.

## Optimized Working R Code

```r
library(Matrix)
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {

  # ---- Convert to data.table for speed ----
  dt <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # ---- Build sparse adjacency matrix (n_cells x n_cells) ----
  # rook_neighbors_unique is an nb object: list of integer vectors (indices into id_order)
  from <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to   <- unlist(rook_neighbors_unique)
  
  # Remove any 0-neighbor placeholders (nb objects use integer(0) for islands)
  valid <- !is.na(to) & to > 0
  from  <- from[valid]
  to    <- to[valid]
  
  # Binary adjacency matrix
  A <- sparseMatrix(i = from, j = to, x = 1, dims = c(n_cells, n_cells))
  
  # Row-normalized version for computing means
  row_sums <- rowSums(A)
  row_sums[row_sums == 0] <- NA  # islands get NA
  A_norm <- A / row_sums  # divides each row by its number of neighbors
  
  # ---- Map each row of dt to (cell_index, year) ----
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_along(years), as.character(years))
  dt[, year_col := year_to_col[as.character(year)]]
  
  # Linear index into cell x year matrix
  dt[, lin_idx := cell_idx + (year_col - 1L) * n_cells]
  
  # Precompute neighbor list (over cells, not rows) for max/min
  nb_list <- rook_neighbors_unique  # already indexed into id_order
  
  # ---- For each source variable, compute neighbor max, min, mean ----
  for (var_name in neighbor_source_vars) {
    
    cat("Processing:", var_name, "\n")
    
    # Build cell x year matrix
    mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mat[dt$lin_idx] <- dt[[var_name]]
    
    # ---- Neighbor mean via sparse matrix multiplication ----
    # A_norm %*% mat: each row i of result = weighted (uniform) avg of neighbors' values
    # Where a neighbor has NA, we need to handle carefully.
    # Strategy: compute sum of non-NA neighbor values / count of non-NA neighbor values
    
    not_na <- !is.na(mat)
    mat_zero <- mat
    mat_zero[is.na(mat_zero)] <- 0
    
    neighbor_sum   <- as.matrix(A %*% mat_zero)       # sum of non-NA neighbor values (NAs treated as 0)
    neighbor_count <- as.matrix(A %*% (not_na * 1.0)) # count of non-NA neighbors
    
    neighbor_mean <- neighbor_sum / neighbor_count     # yields NaN where count==0
    neighbor_mean[neighbor_count == 0] <- NA_real_
    
    # ---- Neighbor max and min ----
    # Use vectorized approach over cells (344K iterations, not 6.46M)
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (i in seq_len(n_cells)) {
      nb_idx <- nb_list[[i]]
      if (length(nb_idx) == 0 || all(nb_idx == 0L)) next
      nb_idx <- nb_idx[nb_idx > 0]
      if (length(nb_idx) == 0) next
      
      # Extract neighbor rows: length(nb_idx) x n_years submatrix
      nb_mat <- mat[nb_idx, , drop = FALSE]
      
      if (length(nb_idx) == 1L) {
        neighbor_max[i, ] <- nb_mat[1, ]
        neighbor_min[i, ] <- nb_mat[1, ]
      } else {
        # colMins / colMaxs avoiding suppressWarnings on all-NA columns
        neighbor_max[i, ] <- apply(nb_mat, 2, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else max(x)
        })
        neighbor_min[i, ] <- apply(nb_mat, 2, function(x) {
          x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else min(x)
        })
      }
    }
    
    # ---- Write results back to dt ----
    dt[, paste0(var_name, "_neighbor_max")  := neighbor_max[lin_idx]]
    dt[, paste0(var_name, "_neighbor_min")  := neighbor_min[lin_idx]]
    dt[, paste0(var_name, "_neighbor_mean") := neighbor_mean[lin_idx]]
  }
  
  # Clean up helper columns
  dt[, c("cell_idx", "year_col", "lin_idx") := NULL]
  
  return(as.data.frame(dt))
}
```

### Even Faster Max/Min with `matrixStats` (Recommended)

The `apply` inner loop over 344K cells Ã— 28 `apply` calls is still slow. Replace it with a chunk-vectorized approach:

```r
# Install if needed: install.packages("matrixStats")
library(matrixStats)

compute_neighbor_maxmin_fast <- function(mat, nb_list, n_cells, n_years) {
  neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    nb_idx <- nb_list[[i]]
    if (length(nb_idx) == 0 || all(nb_idx == 0L)) next
    nb_idx <- nb_idx[nb_idx > 0]
    if (length(nb_idx) == 0) next
    
    nb_mat <- mat[nb_idx, , drop = FALSE]
    
    if (length(nb_idx) == 1L) {
      neighbor_max[i, ] <- nb_mat[1, ]
      neighbor_min[i, ] <- nb_mat[1, ]
    } else {
      neighbor_max[i, ] <- colMaxs(nb_mat, na.rm = TRUE)
      neighbor_min[i, ] <- colMins(nb_mat, na.rm = TRUE)
      # colMaxs/colMins return -Inf/Inf for all-NA columns
      all_na <- colAlls(is.na(nb_mat))
      neighbor_max[i, all_na] <- NA_real_
      neighbor_min[i, all_na] <- NA_real_
    }
  }
  
  list(max = neighbor_max, min = neighbor_min)
}
```

### Top-Level Call (Drop-In Replacement)

```r
# This replaces the original outer loop entirely.
# The trained Random Forest model is not touched.

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Prediction with the existing trained model proceeds unchanged:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string-paste + hash lookups | One-time sparse matrix build (344K cells) |
| **Mean computation** | 6.46M R iterations | Single sparse matrix multiply (`A %*% mat`) |
| **Max/Min computation** | 6.46M R iterations | 344K iterations with `matrixStats::colMaxs` over 28 columns |
| **Estimated time** | 86+ hours | **~10â€“30 minutes** |
| **Memory** | 6.46M-element list of int vectors | ~344K Ã— 28 dense matrices + one sparse matrix |
| **Numerical result** | â€” | **Identical** (same max, min, mean with same NA handling) |

The key insight: the spatial topology is invariant across years, so the neighbor graph (344K nodes, 1.37M edges) should be represented once as a sparse matrix and applied across all 28 year-columns simultaneously, rather than being redundantly expanded into 6.46M row-level lookups.
END_RESPONSE>>>
