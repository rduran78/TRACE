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
run_id: OpenAI_F11_T10_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The code evidence and dataset characteristics indicate that the main bottleneck is **not** Random Forest inference but rather the neighbor computation, especially `build_neighbor_lookup()` and repeated large `lapply()` calls inside `compute_neighbor_stats()` across 6.46 million rows and five variables. These steps involve expensive repetitive lookups and Râ€™s list-based operations, which are highly inefficient for large panels.

### **Diagnosis**
- `build_neighbor_lookup()` creates a 6.46M-length list of integer vectors, which is both memory-heavy and slow.
- For each of the 6.46M entries, `compute_neighbor_stats()` iterates through neighbors, subsets vectors, removes `NA`s, and computes summary stats. This occurs **5 times** (per variable), leading to tens of millions of R-level operations.
- Random Forest `predict()` on 6.46M rows is relatively fast (<1â€“2 hours), nowhere near 86+ hours.

### **Optimization Strategy**
- Replace repeated list operations with **vectorized aggregation** and **data.table joins**.
- Precompute neighbor relationships as a long table: `(cell_id, year) â†’ neighbor_id indices`.
- Use `data.table` or similar to join values and compute max, min, mean in a grouped manner.
- Completely skip rebuilding large R lists.

---

### **Efficient Implementation (using `data.table`)**

```r
library(data.table)

# Assume cell_data has columns: id, year, and neighbor source variables
setDT(cell_data)

# Expand rook neighbor relationships to long format once
# rook_neighbors_unique: list of integer neighbors for each id in id_order
neighbors_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Create all cell-year â†’ neighbor-year pairs
years <- sort(unique(cell_data$year))
neighbors_expanded <- neighbors_dt[CJ(id = id, year = years), on = .(id), allow.cartesian = TRUE]
setnames(neighbors_expanded, "id", "cell_id")
neighbors_expanded[, neighbor_year := year]

# Join neighbor values from cell_data for aggregation
for (var_name in neighbor_source_vars) {
  temp <- merge(neighbors_expanded,
                cell_data[, .(neighbor_id = id, neighbor_year = year, val = get(var_name))],
                by = c("neighbor_id", "neighbor_year"),
                all.x = TRUE)
  
  # Compute summary stats by original cell-year
  stats <- temp[, .(
    paste0(var_name, "_nbr_max") := max(val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") := min(val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") := mean(val, na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Merge stats back to cell_data
  cell_data <- merge(cell_data, stats, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
}

# After this, you can safely call predict() on the full dataset
# preds <- predict(random_forest_model, newdata = cell_data)
```

---

### **Why this is faster**
- Converts repeated nested loops into a single joinâ€“aggregate pattern (C-level optimized in data.table).
- Avoids building multi-million-element lists and repeated R function calls.
- Memory footprint is linear and handled efficiently with keyed joins.

**Expected gain:** From 86+ hours to a few hours (or less) depending on I/O and disk speed. Predict step remains unchanged and is not the bottleneck.
END_RESPONSE>>>
