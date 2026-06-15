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
run_id: OpenAI_F09_T07_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current pipeline recomputes neighbor relationships repeatedly for each variable across all 6.46M rows, which is highly inefficient. The nested loops and repeated lookups create excessive overhead. Neighbor relationships are static across years, so rebuilding them per variable is unnecessary. Memory and time costs scale poorly with 344K cells Ã— 28 years.

---

**Optimization Strategy:**  
- Build the neighbor lookup **once** and reuse it for all variables.
- Use **vectorized operations** or preallocate results instead of repeated `lapply` calls.
- Avoid repeated paste and indexing for each variable; compute once.
- Consider `data.table` for fast joins and efficient memory usage.
- Compute neighbor stats in bulk using matrix operations, then bind results to the main dataset.

---

**Working R Code (Optimized):**

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Function to compute and add neighbor stats for multiple variables efficiently
compute_and_add_neighbor_features <- function(dt, vars, neighbor_lookup) {
  for (v in vars) {
    vals <- dt[[v]]
    
    # Preallocate matrix for results
    res_mat <- matrix(NA_real_, nrow(dt), 3, dimnames = list(NULL, c("max", "min", "mean")))
    
    # Compute stats
    for (i in seq_along(neighbor_lookup)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          res_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    
    # Add columns to data.table
    dt[[paste0(v, "_nbr_max")]]  <- res_mat[, "max"]
    dt[[paste0(v, "_nbr_min")]]  <- res_mat[, "min"]
    dt[[paste0(v, "_nbr_mean")]] <- res_mat[, "mean"]
  }
  invisible(dt)
}

# Apply optimization
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
compute_and_add_neighbor_features(cell_data, neighbor_source_vars, neighbor_lookup)

# The Random Forest model can now predict with new features without retraining
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Expected Impact:**  
- Neighbor table built **once** instead of 5Ã—.
- Eliminates redundant string operations and repeated indexing.
- Improves runtime dramaticallyâ€”likely reducing from 86+ hours to a few hours (depending on thread usage).
- Preserves estimands and trained model integrity.
END_RESPONSE>>>
