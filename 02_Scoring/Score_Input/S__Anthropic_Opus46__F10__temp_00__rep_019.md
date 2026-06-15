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
run_id: Anthropic_F10_T00_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F10_Anthropic_temp_00_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each constructed via string-pasting and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per query against a vector of 6.46M names â€” this alone is catastrophically slow. The function does ~6.46M Ã— avg_neighbors string matches against a 6.46M-length named vector.

2. **`compute_neighbor_stats` iterates over 6.46M list elements in an `lapply`**, extracting subsets of a numeric vector. While each individual operation is small, the R-level loop overhead across 6.46M iterations, repeated for 5 variables, is substantial.

3. **The neighbor lookup is year-aware but redundant**: rook neighbors are a *spatial* relationship â€” the same cell has the same neighbors every year. The current code re-resolves neighbor row indices per cell-year, but the spatial topology is fixed. The only thing that changes is the year offset.

**Estimated cost breakdown:**
- `build_neighbor_lookup`: ~6.46M string-paste + named-vector lookups â†’ ~70+ hours
- `compute_neighbor_stats` (5 vars Ã— 6.46M rows): ~15+ hours
- Total: ~86+ hours

## Optimization Strategy

1. **Build the graph topology once at the cell level (344K nodes), not the cell-year level (6.46M rows).** The rook neighbor adjacency is year-invariant. We construct a sparse adjacency matrix once.

2. **Use sparse matrixâ€“dense matrix multiplication for aggregation.** For each variable, we reshape the values into a (cells Ã— years) matrix, then use the sparse adjacency matrix to compute neighbor sums and neighbor counts in one shot. From sum and count we get mean; for max and min we use row-wise sparse operations.

3. **For max and min**, we use `data.table` grouped operations with an edge list, which is vectorized and avoids per-row R loops.

4. **Avoid all string-pasting and named-vector lookups entirely.**

5. **Memory**: Sparse matrix of 344K Ã— 344K with ~1.37M nonzeros â‰ˆ 33 MB. Dense matrices of 344K Ã— 28 â‰ˆ 77 MB each. Total peak memory well under 16 GB.

**Expected speedup**: From 86+ hours to **~2â€“10 minutes**.

## Optimized R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Sparse graph neighborhood aggregation via matrix operations
# Preserves numerical equivalence with original compute_neighbor_stats
# =============================================================================

library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 0. Convert to data.table for speed (non-destructive)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  
  n_cells <- length(id_order)
  
  # Map cell IDs to integer indices 1..n_cells
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # -------------------------------------------------------------------------
  # 1. Build sparse adjacency matrix ONCE (344K x 344K, ~1.37M nonzeros)
  #    A[i,j] = 1 means cell j is a rook neighbor of cell i
  # -------------------------------------------------------------------------
  message("Building sparse adjacency matrix...")
  
  # Construct edge list from the nb object
  from_list <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  to_list   <- unlist(rook_neighbors_unique)
  
  # Remove 0-neighbor placeholders (spdep uses 0L for no-neighbor entries)
  valid <- to_list > 0L
  from_list <- from_list[valid]
  to_list   <- to_list[valid]
  
  # Sparse adjacency: row i has 1s in columns corresponding to neighbors of cell i
  A <- sparseMatrix(
    i = from_list,
    j = to_list,
    x = 1,
    dims = c(n_cells, n_cells)
  )
  
  # Neighbor count per cell (constant across years)
  neighbor_count <- as.numeric(rowSums(A))  # length n_cells
  
  rm(from_list, to_list, valid)
  
  # -------------------------------------------------------------------------
  # 2. Build row-index mapping: for each (cell_idx, year) -> row in dt
  #    We need this to scatter/gather between long format and matrix format
  # -------------------------------------------------------------------------
  message("Building cell-year index mapping...")
  
  # Map each row's cell ID to its cell index
  dt[, cell_idx := id_to_idx[as.character(id)]]
  
  # Get sorted unique years and map to column indices
  years_unique <- sort(unique(dt$year))
  n_years <- length(years_unique)
  year_to_col <- setNames(seq_along(years_unique), as.character(years_unique))
  dt[, year_col := year_to_col[as.character(year)]]
  
  # -------------------------------------------------------------------------
  # 3. Build edge list data.table for max/min (vectorized grouped ops)
  # -------------------------------------------------------------------------
  message("Building edge list for max/min computation...")
  
  # Extract edge list from sparse matrix
  A_T <- summary(A)  # gives (i, j, x) triplets
  edges_dt <- data.table(from = A_T$i, to = A_T$j)
  rm(A_T)
  
  # -------------------------------------------------------------------------
  # 4. For each variable, compute neighbor max, min, mean
  # -------------------------------------------------------------------------
  
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing variable: %s", var_name))
    
    vals <- dt[[var_name]]
    
    # --- 4a. Build (n_cells x n_years) matrix of values ---
    # Use NA for missing cell-year combinations
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(dt$cell_idx, dt$year_col)] <- vals
    
    # --- 4b. MEAN via sparse matrix multiplication ---
    # For non-NA handling: we need sum of non-NA neighbors and count of non-NA neighbors
    
    # Create a non-NA indicator matrix
    V_notna <- matrix(0, nrow = n_cells, ncol = n_years)
    V_notna[cbind(dt$cell_idx, dt$year_col)] <- as.numeric(!is.na(vals))
    
    # Replace NA with 0 for summation
    V_zero <- V
    V_zero[is.na(V_zero)] <- 0
    
    # Neighbor sum: A %*% V_zero  (n_cells x n_years)
    neighbor_sum <- as.matrix(A %*% V_zero)
    
    # Neighbor non-NA count: A %*% V_notna
    neighbor_nna_count <- as.matrix(A %*% V_notna)
    
    # Mean = sum / count (NA where count == 0)
    neighbor_mean_mat <- neighbor_sum / neighbor_nna_count
    neighbor_mean_mat[neighbor_nna_count == 0] <- NA_real_
    
    # Also set NA where cell has no neighbors at all
    neighbor_mean_mat[neighbor_count == 0, ] <- NA_real_
    
    rm(V_zero, neighbor_sum)
    
    # --- 4c. MAX and MIN via vectorized edge-list operations ---
    # For each year, look up neighbor values and compute grouped max/min
    
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    
    for (yc in seq_len(n_years)) {
      # Get values for this year column
      year_vals <- V[, yc]
      
      # Look up neighbor values via edge list
      nvals <- year_vals[edges_dt$to]
      
      # Build temporary DT for grouped aggregation
      tmp <- data.table(from = edges_dt$from, nval = nvals)
      
      # Remove edges where neighbor value is NA
      tmp <- tmp[!is.na(nval)]
      
      if (nrow(tmp) > 0) {
        agg <- tmp[, .(nmax = max(nval), nmin = min(nval)), by = from]
        neighbor_max_mat[agg$from, yc] <- agg$nmax
        neighbor_min_mat[agg$from, yc] <- agg$nmin
      }
    }
    
    rm(V, V_notna, neighbor_nna_count)
    
    # --- 4d. Scatter results back to long-format rows ---
    max_col <- paste0("neighbor_max_", var_name)
    min_col <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
    
    idx_mat <- cbind(dt$cell_idx, dt$year_col)
    
    set(dt, j = max_col,  value = neighbor_max_mat[idx_mat])
    set(dt, j = min_col,  value = neighbor_min_mat[idx_mat])
    set(dt, j = mean_col, value = neighbor_mean_mat[idx_mat])
    
    rm(neighbor_max_mat, neighbor_min_mat, neighbor_mean_mat, idx_mat)
    gc()
  }
  
  # -------------------------------------------------------------------------
  # 5. Clean up helper columns and return
  # -------------------------------------------------------------------------
  dt[, c("cell_idx", "year_col") := NULL]
  
  return(dt)
}


# =============================================================================
# USAGE
# =============================================================================

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data_dt <- optimize_neighbor_features(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = neighbor_source_vars
# )
#
# # Predict with the pre-trained Random Forest (no retraining)
# predictions <- predict(rf_model, newdata = cell_data_dt)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-paste + named-vector lookups | Sparse matrix built once from `nb` object |
| **Mean computation** | R-level loop over 6.46M list elements | Sparse matrix Ã— dense matrix multiply (`A %*% V`) |
| **Max/Min computation** | R-level loop over 6.46M list elements | Vectorized `data.table` grouped aggregation over edge list, 28 year iterations |
| **Topology reuse** | Rebuilt implicitly per row | Built once (344K Ã— 344K sparse), reused across all years and variables |
| **Numerical equivalence** | Baseline | **Preserved exactly**: same `max`, `min`, `mean` of non-NA rook-neighbor values per cell-year |
| **RF model** | Pre-trained | **Untouched** â€” only `predict()` is called |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** |
| **Peak memory** | Dominated by 6.46M-element list of index vectors | ~2â€“3 GB (sparse matrix + dense matrices + edge DT) |

### Why numerical equivalence is preserved

- **Mean**: `(A %*% V_zero)[i,] / (A %*% V_notna)[i,]` computes exactly `sum(non-NA neighbor values) / count(non-NA neighbor values)` â€” identical to `mean(neighbor_vals)` after removing NAs.
- **Max/Min**: The edge-list approach looks up exactly the same neighbor values (`V[neighbor_j, year]`) and applies `max`/`min` grouped by the focal cell â€” identical to the original per-row logic.
- **NA handling**: Cells with zero neighbors or all-NA neighbors produce `NA` for all three statistics, matching the original `if (length(idx) == 0) return(c(NA, NA, NA))` and `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` guards.
END_RESPONSE>>>
