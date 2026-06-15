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
run_id: OpenAI_F12_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim is **incorrect**. While `do.call(rbind, result)` has some overhead, the real bottleneck is the **nested `lapply`-based row-wise interpretation inside `compute_neighbor_stats()` combined with repeated expensive lookups and object allocation**, all performed 5 times (once per `var_name`) over ~6.46M rows and millions of neighbors. This results in tens of millions of R-level function calls and memory churn.  
`do.call(rbind, â€¦)` runs **once per variable**, contributing negligible time relative to per-row operations. The true performance issue is the **interpreted R loop inside `compute_neighbor_stats`** that computes neighbor statistics row-by-row.

---

### **Optimization Strategy**
- **Vectorize neighbor aggregation**: Instead of iterating over 6.46M rows with `lapply`, **precompute a long table of (cell-year, neighbor, value)** and use `data.table` or `collapse` for *grouped aggregation*.
- Do this once per variable in a tight, compiled backend instead of per-row R overhead.
- Benefit: Removes millions of R function calls and allows `C`-level grouping.

---

### **Optimized Approach**
1. Represent neighbor relationships as an edge list and repeat it across all years.
2. Join with data to get neighbor values and **aggregate with fast grouping**.
3. Compute max, min, mean in one grouped pass.

---

### **Working R Code**
```r
library(data.table)

# Assuming 'cell_data' has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# and 'id_order' matches neighbor structure

# 1. Create repeated edge list across all years
neighbors_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  data.table(id = id_order[i], neighbor = rook_neighbors_unique[[i]])
}), use.names = TRUE)

# Repeat across years
years <- unique(cell_data$year)
neighbor_pairs <- neighbors_dt[CJ(year = years, dummy = 1),
                                .(id, neighbor, year), on = .(dummy), allow.cartesian = TRUE][, dummy := NULL]

# Convert cell_data to data.table keyed
setDT(cell_data)
setkey(cell_data, id, year)

# For each var_name, compute stats
compute_neighbor_stats_fast <- function(var_name) {
  dt <- neighbor_pairs[
    cell_data, on = .(neighbor = id, year), nomatch = 0
  ][
    , .(neighbor_val = get(var_name)), by = .(id, year)
  ]
  dt[, .(
    paste0(var_name, "_ngb_max") = max(neighbor_val, na.rm = TRUE),
    paste0(var_name, "_ngb_min") = min(neighbor_val, na.rm = TRUE),
    paste0(var_name, "_ngb_mean") = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
}

# 2. Loop over 5 source variables and merge
result_list <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), result_list)

# 3. Merge back to cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- Eliminates the row-wise `lapply`, which caused ~86-hour runtime due to R interpreter overhead.
- Performs **one vectorized aggregation per variable** using `data.table`â€™s optimized `C` backend.
- Memory efficient: uses simple joins and grouping instead of constructing nested lists for 6.46M rows.

**Expected runtime improvement:** From 80+ hours to minutes on a 16GB laptop.  
**Model integrity preserved:** Same estimand, no model retraining required.
END_RESPONSE>>>
