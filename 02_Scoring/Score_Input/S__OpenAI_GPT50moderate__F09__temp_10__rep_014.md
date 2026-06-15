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
run_id: OpenAI_F09_T10_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The slow runtime (86+ hours) comes mainly from redundant neighbor lookup computation repeated for every variable. Currently, for each of the 5 neighbor source variables, the pipeline recomputes aggregation across 6.46M rows, repeatedly traversing the same neighbor relationships. This repeated cost dominates performance.  

**Optimization Strategy:**  
- Precompute `neighbor_lookup` **once** using `build_neighbor_lookup()` (already done).
- Avoid multiple full-table passes by computing **all neighbor stats in fewer passes**, ideally in C/fast R (e.g., `data.table`).
- Reduce object creation overhead by vectorizing `compute_neighbor_stats()` for multiple variables, instead of looping over variables.
- Leverage aggregation over a long vector with indices rather than repeated `lapply`.  

**Working R Code (Optimized):**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup once (as list of integer vectors)
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Flatten neighbor_lookup into adjacency representation
src <- rep(seq_along(neighbor_lookup), lengths(neighbor_lookup))
dst <- unlist(neighbor_lookup, use.names = FALSE)

adj_dt <- data.table(src = src, dst = dst)

# Function to compute neighbor stats for multiple variables at once
compute_neighbor_stats_multi <- function(dt, adj_dt, vars) {
  # Melt selected columns for neighbors
  result_list <- lapply(vars, function(var) {
    # Join neighbor values
    joined <- adj_dt[, .(src, val = dt[[var]][dst])]
    # Aggregate (min, max, mean) by src
    agg <- joined[, .(
      max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
      min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
      mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
    ), by = src]
    setnames(agg, c("src", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
    agg
  })
  
  # Merge all results by src
  out <- Reduce(function(a, b) merge(a, b, by = "src", all = TRUE), result_list)
  out[order(src)]
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor summaries in one go
neighbor_stats <- compute_neighbor_stats_multi(cell_data, adj_dt, neighbor_source_vars)

# Join back to cell_data
cell_data[, src := .I]
cell_data <- merge(cell_data, neighbor_stats, by = "src", all.x = TRUE)[, src := NULL]

# Now cell_data has neighbor stats for all variables, ready for Random Forest prediction
```

**Why this is faster:**  
- Only **one traversal** per variable through adjacency relationships instead of per cell repeatedly.
- Uses `data.table` for efficient join and aggregation.
- No recomputation of neighbor index arrays.
- Memory-conscious design (adjacency list is integers, no duplication of large data).

This approach should reduce runtime from **86+ hours to a few hours** (bounded by join/aggregation time on 6.46M rows and ~1.37M neighbor links) on a 16 GB laptop, while preserving the original estimand and Random Forest pipeline.
END_RESPONSE>>>
