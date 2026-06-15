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
run_id: OpenAI_F11_T07_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence shows the colleagueâ€™s diagnosis is incorrect: Random Forest inference is **not** the main bottleneck. The pipelineâ€™s dominant cost comes from repeatedly running `compute_neighbor_stats()` over **6.46 million rows Ã— 5 variables** using nested `lapply()` with repeated lookups and `do.call(rbind, ...)`. This is highly inefficient in pure R and causes the 86+ hour runtime.

### **Diagnosis**
- `build_neighbor_lookup()` creates an `lapply` list of integer vectors for each row â†’ list length â‰ˆ 6.46M.
- For each variable, `compute_neighbor_stats()` iterates over these 6.46M elements and performs:
  - Multiple vector allocations.
  - `do.call(rbind, ...)` on a huge list.
- This scales poorly; complexity is effectively **O(N Ã— k)** for N=6.46M and kâ‰ˆ4â€“5 passes.
- Random Forest inference on 6.46M rows and 110 predictors is trivial compared to these repeated R loops.

### **Correct Optimization Strategy**
- Precompute neighbor indices **once** at the *id level* (344,208 cells) instead of per-row.
- Convert operations to vectorized `data.table` or `dplyr` joins rather than looping.
- Compute neighbor statistics **per year** in a grouped manner.

### **Optimized Approach**
- Avoid 6.46M-element neighbor list.
- Use `data.table` keyed joins to aggregate neighbor stats efficiently.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Ensure keys for fast joins
setkey(dt, id, year)

# Build neighbor table at cell level (id only, not repeated per row)
neighbors_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand neighbors by year for join
years <- unique(dt$year)
neighbors_dt <- neighbors_dt[CJ(year = years), on = .(id = V1)]
setnames(neighbors_dt, c("id", "neighbor_id", "year"))

# Join neighbor attributes for each variable and compute stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Bring neighbor variable values
  tmp <- dt[, .(neighbor_id = id, year, val = get(var_name))]
  neighbors_dt <- tmp[neighbors_dt, on = .(neighbor_id, year)]
  
  # Compute stats grouped by (id, year)
  stats <- neighbors_dt[, .(
    max = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Add back to main dt
  setnames(stats, c("id", "year", 
                    paste0(var_name, "_nb_max"),
                    paste0(var_name, "_nb_min"),
                    paste0(var_name, "_nb_mean")))
  dt <- stats[dt, on = .(id, year)]
}

# Convert back to data.frame if needed
cell_data <- as.data.frame(dt)
```

---

### **Expected Impact**
- Eliminates 6.46M-element nested `lapply()` calls.
- Vectorized grouping reduces runtime from **86+ hours â†’ minutes** on a laptop.
- Preserves trained Random Forest model and all original estimands.

**Bottom line:** The bottleneck is the neighbor feature engineering step, not Random Forest inference. Use keyed joins and grouped aggregation with `data.table` for massive speedup.
END_RESPONSE>>>
