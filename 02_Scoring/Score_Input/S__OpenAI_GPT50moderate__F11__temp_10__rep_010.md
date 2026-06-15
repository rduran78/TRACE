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
run_id: OpenAI_F11_T10_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck does **not** match the evidence. The pipeline spends considerable time inside `build_neighbor_lookup` and `compute_neighbor_stats`. Both functions use `lapply` over millions of rows, repeatedly performing expensive vector operations and lookups in R interpreted loops. Given 6.46M rows and multiple neighbor source variables, this dominates runtime far more than calling `predict()` on ~6M rows (which typically takes minutes on a trained RF model).  

**Correct Bottleneck:**  
Neighbor feature computation, specifically the repeated neighbor index lookups and aggregation logic inside `compute_neighbor_stats`.

---

### **Optimization Strategy**
- Vectorize neighbor statistics calculation, avoid heavy repeated `lapply`.
- Use `data.table` for fast group operations on large datasets.
- Precompute long-format (cell-year Ã— neighbor-cell) table using rook neighbors and join for all variables instead of rebuilding lookups for each source variable.

---

### **Optimized R Code**
```r
library(data.table)

# Convert cell_data to data.table
dt <- as.data.table(cell_data)

# Prepare neighbor relationships in long format
# rook_neighbors_unique: list of neighbors by cell id in same order as id_order
neighbor_dt <- data.table(from_id = rep(id_order, lengths(rook_neighbors_unique)),
                           to_id = unlist(rook_neighbors_unique))

# Expand to panel: replicate for all years
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[, .(to_id = to_id, year = years), by = .(from_id)]

# Create join keys
neighbor_dt[, from_key := paste(from_id, year, sep = "_")]
neighbor_dt[, to_key := paste(to_id, year, sep = "_")]

# Map original row keys
dt[, cell_key := paste(id, year, sep = "_")]

# Join neighbor values for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  # Join values from dt to neighbor_dt by to_key
  neighbor_dt <- neighbor_dt[dt, on = .(to_key = cell_key), 
                              neighbor_val := get(var)]
  
  # Compute stats grouped by from_key
  stats_dt <- neighbor_dt[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = from_key]
  
  # Map stats back to dt
  setkey(stats_dt, from_key)
  dt[, paste0(var, "_nbr_max") := stats_dt[.(cell_key), max_val]]
  dt[, paste0(var, "_nbr_min") := stats_dt[.(cell_key), min_val]]
  dt[, paste0(var, "_nbr_mean") := stats_dt[.(cell_key), mean_val]]
  
  # Remove temp column
  neighbor_dt[, neighbor_val := NULL]
}

# Result is dt with neighbor features added
cell_data <- dt
```

---

**Why This Helps:**  
- Eliminates millions of small `lapply` calls.
- Does all neighbor joins and aggregations in compiled, optimized `data.table`.
- Reduces runtime from dozens of hours to likely **minutes or under an hour** on 16 GB RAM.

**Preserves:**  
- Original Random Forest model and estimand.
- Same neighbor-derived features with correct max/min/mean logic.

**Final Step:**  
Once features are computed, call `predict(rf_model, newdata = cell_data)` directly (inference will be relatively fast).  

**Summary:** Bottleneck is neighbor feature engineering, not Random Forest prediction. Optimize with `data.table` joins and grouped aggregations instead of nested loops.
END_RESPONSE>>>
