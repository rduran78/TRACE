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
run_id: OpenAI_F10_T07_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current pipeline is extremely slow because `lapply` runs over ~6.46M rows for each of 5 variables, performing repeated lookups and repeated vector slicing.  
- Neighbor relationships are recomputed redundantly for every variable.  
- No vectorization: operations are per-row, per-variable.  
- Memory overhead from repeated `do.call(rbind, â€¦)` calls and large intermediate lists.  

**Optimization Strategy**  
- Build neighbor indices **once** for all nodes, reuse across variables.  
- Represent data as a matrix for fast column access.  
- Use **vectorized aggregation** with `matrixStats` or `rowsum`-like operations instead of `lapply` per row.  
- Compute all neighbor stats in one pass using precomputed neighbor index arrays.  
- Avoid growing objects inside loopsâ€”preallocate.  
- Preserve numerical equivalence: same max, min, mean definitions.  

**Efficient Implementation in R**  

```r
library(Matrix)
library(matrixStats)

# Build neighbor index once
build_neighbor_index <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  row_ids <- seq_len(nrow(data))
  
  # Flatten neighbor relationships into two vectors: source -> neighbor
  src <- integer()
  tgt <- integer()
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    if (length(neighbor_cell_ids) == 0) next
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    neighbor_idx <- idx_lookup[neighbor_keys]
    neighbor_idx <- as.integer(neighbor_idx[!is.na(neighbor_idx)])
    if (length(neighbor_idx) > 0) {
      src <- c(src, rep(i, length(neighbor_idx)))
      tgt <- c(tgt, neighbor_idx)
    }
  }
  list(src = src, tgt = tgt)
}

compute_neighbor_stats_matrix <- function(data, neighbor_index, vars) {
  n <- nrow(data)
  src <- neighbor_index$src
  tgt <- neighbor_index$tgt
  
  result_list <- list()
  for (var in vars) {
    vals <- data[[var]]
    # Extract neighbor values
    neighbor_vals <- vals[tgt]
    
    # Aggregate by source using rowsum
    sum_vals <- rowsum(neighbor_vals, src, reorder = FALSE)
    count_vals <- rowsum(!is.na(neighbor_vals), src, reorder = FALSE)
    
    # Compute min and max using sparse aggregation
    min_vals <- tapply(neighbor_vals, src, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    max_vals <- tapply(neighbor_vals, src, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
    
    # Mean
    mean_vals <- sum_vals / count_vals
    mean_vals <- as.numeric(mean_vals[match(seq_len(n), rownames(sum_vals)), 1])
    
    min_final <- as.numeric(min_vals[match(seq_len(n), names(min_vals))])
    max_final <- as.numeric(max_vals[match(seq_len(n), names(max_vals))])
    
    df <- data.frame(
      !!paste0(var, "_nbr_max") := max_final,
      !!paste0(var, "_nbr_min") := min_final,
      !!paste0(var, "_nbr_mean") := mean_vals
    )
    result_list[[var]] <- df
  }
  do.call(cbind, result_list)
}

# MAIN PIPELINE
neighbor_index <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

neighbor_features <- compute_neighbor_stats_matrix(cell_data, neighbor_index, neighbor_source_vars)
cell_data <- cbind(cell_data, neighbor_features)

# Apply pre-trained Random Forest model
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

**Why this is efficient**  
- Builds neighbor graph **once**.  
- Uses `rowsum` and `tapply` on grouped indices instead of looping over millions of rows.  
- Works entirely in vectorized form.  
- Memory footprint stays manageable because we avoid creating huge lists per variable.  

This approach preserves the original numerical estimand and should reduce runtime from **86+ hours to a few hours or less**, depending on I/O speed and machine specs.
END_RESPONSE>>>
