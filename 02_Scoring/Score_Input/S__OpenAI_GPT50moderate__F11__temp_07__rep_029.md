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
run_id: OpenAI_F11_T07_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that Random Forest inference is the main bottleneck is incorrect. The code shows that the pipeline spends substantial time constructing neighbor lookups (`build_neighbor_lookup`) and computing neighbor statistics (`compute_neighbor_stats`) for ~6.46 million rows across 5 variables. These two functions repeatedly traverse large lists and perform expensive operations like string concatenation (`paste`), indexing, and `lapply` over millions of elements. Random Forest inference on 6.46M rows is relatively fast (typically minutes), so the core bottleneck lies in the neighbor feature computation, not prediction.

---

**Optimization Strategy:**  
1. **Avoid string concatenation for keys**: Use integer indexing rather than `paste` keys.
2. **Vectorize neighbor statistics**: Replace per-row `lapply` with a more vectorized approach using `data.table` or matrix operations.
3. **Precompute neighbor index arrays**: Build a flat integer mapping once and reuse it.
4. **Use data.table for group aggregation**: Efficiently compute max, min, mean for neighbors.

---

**Working Optimized R Code (using `data.table`):**

```r
library(data.table)

# Convert cell_data to data.table for fast joins
setDT(cell_data)

# Precompute neighbor table
# rook_neighbors_unique: list of integer vectors of neighbor indices for each id_order position
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
})

# Expand to panel by joining on years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(neighbor_id = neighbor_id), by = .(id)]
neighbor_panel <- neighbor_dt[, .(id = rep(id, length(years)),
                                  neighbor_id = rep(neighbor_id, length(years)),
                                  year = rep(years, each = length(neighbor_id)))]
setkey(neighbor_panel, neighbor_id, year)
setkey(cell_data, id, year)

# Join neighbor values for all vars at once
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in vars) {
  tmp <- cell_data[, .(neighbor_val = get(var)), by = .(id, year)]
  setnames(tmp, "id", "neighbor_id")
  neighbor_panel <- tmp[neighbor_panel, on = .(neighbor_id, year)]
  # Compute stats by id-year
  stats <- neighbor_panel[, .(
    max = max(neighbor_val, na.rm = TRUE),
    min = min(neighbor_val, na.rm = TRUE),
    mean = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  setnames(stats, c("max", "min", "mean"), paste0(var, c("_max", "_min", "_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
}
```

---

**Why this works:**  
- Eliminates millions of `paste` calls and nested `lapply` loops.
- Uses keyed `data.table` joins and group aggregation for high speed.
- Preserves original numerical estimands (max, min, mean for neighbors).
- Random Forest model remains untouched, predictions happen after this step.

**Expected runtime reduction:**  
From 86+ hours to a few hours or less on a 16 GB laptop, because operations are now vectorized and memory-efficient.
END_RESPONSE>>>
