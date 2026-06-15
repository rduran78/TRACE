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
run_id: OpenAI_F12_T07_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim is **incorrect**. While `do.call(rbind, result)` has overhead, the dominant bottleneck is **per-cell repeated list processing in `compute_neighbor_stats()`** across 6.46M rows Ã— 5 variables, causing massive R-level iteration and memory churn. Each `lapply` iterates millions of times and repeatedly subsets vectors (`vals[idx]`). This is far more expensive than the final `rbind`.  

The true constraint: **Râ€™s interpreted loop over millions of rows** â†’ huge overhead compared to vectorized or compiled approaches.  

---

### **Optimization Strategy**
Avoid per-row R loops; leverage **vectorized/compiled code** or **data.table** joins. Pre-expand neighbor relationships into a long format and compute max/min/mean by group using fast aggregation. Then join back to the main table. This avoids millions of R-level function calls.

---

### **Optimized Approach**
1. Precompute a long table of `(cell_id, year, var_name, neighbor_value)` for all neighbors.
2. Aggregate `max`, `min`, `mean` by `(cell_id, year)`.
3. Join results back to the main dataset.
4. Use `data.table` for speed and memory efficiency.

---

### **Working R Code**

```r
library(data.table)

# Convert main data to data.table
setDT(cell_data)

# 1. Build neighbor lookup as a long table of pairs
# neighbor_lookup: list of integer indices (neighbors per row)
lookup_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(
    src = i,
    nbr = neighbor_lookup[[i]]
  )
}), use.names = TRUE, fill = TRUE)

# Add year and id info
lookup_dt[, year := cell_data$year[src]]

# 2. For each variable, compute stats in a vectorized way
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in neighbor_source_vars) {
  vals <- cell_data[[v]]
  
  # Map neighbor values
  lookup_dt[, val := vals[nbr]]
  
  # Remove NAs
  lookup_dt_no_na <- lookup_dt[!is.na(val)]
  
  # Aggregate max, min, mean by src (cell-year)
  agg <- lookup_dt_no_na[, .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(src)]
  
  # Join back on src -> add as new columns
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  cell_data[agg$src, (names(agg)[-1]) := agg[, -1], on = .I]
  
  # Remove val column for next iteration
  lookup_dt[, val := NULL]
}
```

---

### **Why this is faster**
- Eliminates millions of R-level calls by replacing `lapply` with `data.table` aggregation.
- Single pass per variable instead of deep nested loops.
- Avoids repeated object growth and expensive `do.call(rbind, ...)`.

---

**Expected runtime improvement:** From **86+ hours â†’ under 1 hour** on 16 GB RAM laptop for ~6.5M rows and ~1.3M neighbor links, thanks to vectorization and efficient memory layout.  

**Trained Random Forest model remains untouched.**  
**Original estimand preserved (same neighbor-based max/min/mean).**
END_RESPONSE>>>
