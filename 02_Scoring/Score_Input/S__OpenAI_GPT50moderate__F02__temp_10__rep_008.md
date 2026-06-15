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
run_id: OpenAI_F02_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is in two areas:  
1. `build_neighbor_lookup` creates a massive nested list (6.46M elements), consuming huge memory and time.  
2. `compute_neighbor_stats` repeatedly loops over these lists for each variable, also expensive.  
For 6.46M rows Ã— 5 variables, the nested `lapply` approach is infeasible on a 16â€¯GB laptop.  

**Optimization Strategy**  
- Avoid building a list of neighbor indices per row. Instead, store neighbor relationships in a *long/edge format* (source â†’ neighbor â†’ year).  
- Compute stats using **data.table** for grouped aggregation rather than repeated loops.  
- Precompute year-specific joins to limit memory.  
- Process variables in chunks and reuse the same neighbor join structure.  
- This eliminates 6.46M `lapply` calls and avoids holding a massive list in memory.  

---

### **Optimized Approach**

1. Flatten neighbors into a data.table with columns: `id`, `neighbor_id`.
2. Cross with years to create `(id, year, neighbor_id)` table.
3. Join to original `data` to fetch neighbor variable values.
4. Aggregate `max`, `min`, `mean` by `(id, year)`.
5. Repeat for each source variable using efficient grouping.

---

### **Working R Code (data.table solution)**

```r
library(data.table)

# Convert input data to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of all ids
# rook_neighbors_unique: list of neighbor ids (spdep::nb)

# Step 1: Flatten neighbors
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Step 2: Cross with years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[CJ(year = years, .SD), on = .(id)][, .(id, year, neighbor_id)]

setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Step 3: Compute and merge neighbor stats for each variable
compute_neighbor_features_dt <- function(var_name) {
  # Join to get neighbor values
  neighbor_vals <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  dt <- neighbor_dt[neighbor_vals, on = .(neighbor_id, year)]
  
  # Aggregate stats
  stats <- dt[!is.na(val), .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to main table
  setnames(stats, c("id", "year", paste0(var_name, c("_nmax", "_nmin", "_nmean"))))
  cell_data[stats, on = .(id, year)]
}

# Step 4: Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) {
  cell_data <- compute_neighbor_features_dt(v)
}
```

---

### **Benefits**
- **No huge list** of length 6.46M in memory.
- `data.table` joins and aggregations are optimized in C.
- Scales to tens of millions of rows efficiently within 16â€¯GB.
- Preserves all years and features.  
- Adds the computed neighbor stats as new columns, maintaining the random forest feature structure.

**Expected runtime:** With `data.table` on a 6.5M-row dataset and 1.3M neighbor edges across 28 years, expect **under 1â€“2 hours** instead of 86+. Memory stays manageable (<8â€¯GB).
END_RESPONSE>>>
