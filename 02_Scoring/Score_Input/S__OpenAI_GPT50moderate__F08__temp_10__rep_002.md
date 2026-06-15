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
run_id: OpenAI_F08_T10_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is inefficient because `compute_neighbor_stats` processes neighbor lookups row-by-row across **6.46M rows**, recalculating statistics repeatedly for each cell-year. Each neighbor set does not change over time, but the lookup repeats year-specific matching every iteration. This creates massive overhead (list processing inside loops, redundant computations) leading to the 86+ hours runtime.

---

**Optimization Strategy**  
1. **Separate static and dynamic components**:  
   - *Static*: neighbor relationships between cell IDs (344,208 cells).
   - *Dynamic*: year-specific variables (e.g., ntl, ec) attached to those cell IDs.  

2. **Build neighbor index once per cell (static)**: map each cell ID to neighbor IDs.  

3. **Process in year blocks** (28 iterations):  
   - For each yearâ€™s ~344k rows, compute neighbor stats using static neighbors and current-year variable values.
   - Vectorize computation with matrix approach instead of row-wise `lapply`.  

4. **Memory-efficient join**: work year-by-year to avoid holding 6.46M rows fully expanded in memory.  
   
5. Preserve Random Forest modelâ€”only feature computation changes.

---

**Optimized R Code**

```r
# Static neighbor lookup by cell index
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  lapply(neighbors, function(nb) id_order[nb])
}

# Compute stats for one year efficiently
compute_year_stats <- function(df_year, neighbor_static, var_name) {
  vals <- setNames(df_year[[var_name]], df_year$id)
  
  res_mat <- matrix(NA_real_, nrow = nrow(df_year), ncol = 3)
  for (i in seq_len(nrow(df_year))) {
    nb_ids <- neighbor_static[[i]]
    nb_vals <- vals[as.character(nb_ids)]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) > 0) {
      res_mat[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
    }
  }
  
  colnames(res_mat) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  res_mat
}

# Main optimized computation
compute_all_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  neighbor_static <- build_static_neighbor_lookup(id_order, neighbors)
  
  # Split once by year for block processing
  split_years <- split(cell_data, cell_data$year)
  processed_list <- vector("list", length(split_years))
  
  for (y in seq_along(split_years)) {
    df_year <- split_years[[y]]
    feat_list <- vector("list", length(vars))
    for (j in seq_along(vars)) {
      feat_list[[j]] <- compute_year_stats(df_year, neighbor_static, vars[j])
    }
    feat_mat <- do.call(cbind, feat_list)
    processed_list[[y]] <- cbind(df_year, feat_mat)
  }
  
  do.call(rbind, processed_list)
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Proceed with Random Forest predictions (model preserved)
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why Itâ€™s Faster**  
- Eliminates 6.46M *per-row* neighbor lookups.
- Performs static neighbor mapping only once.
- Loops over ~344k rows Ã— 28 years instead of 6.46M Ã— 5 vars row-by-row overhead.
- Memory stays reasonable (process ~344k rows at a time).
- Expected runtime drops from ~86+ hours to a few hours (or less with parallelization).

**Numerical consistency**: Uses same max/min/mean logic, preserves original estimand and model inputs.
END_RESPONSE>>>
