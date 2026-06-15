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
run_id: OpenAI_F11_T07_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence shows the main bottleneck is **neighbor feature computation**, not Random Forest inference. With 6.46M rows and nested `lapply` calls over millions of elements, `build_neighbor_lookup` and `compute_neighbor_stats` dominate runtime. Random Forest `predict()` on 6.46M rows with 110 variables is large but would take minutes to hours, not 86+ hours.

### **Diagnosis**
- `build_neighbor_lookup` creates ~6.46M lists of integer vectors via `lapply`, which is memory- and time-heavy.
- `compute_neighbor_stats` loops over these lists for each of 5 variables, doing repeated filtering and aggregation in R.
- This is pure R interpreted code operating on millions of elements â†’ huge overhead.
- Random Forest inference is not the bottleneck.

---

### **Optimization Strategy**
- Precompute neighbor indices **once** as an integer matrix.
- Use **vectorized operations** or **data.table** joins to compute neighbor stats.
- Avoid repeated `lapply` and `paste` inside the main loop.
- Compute all variables in a single pass if possible.
- Keep Random Forest model intact; only optimize feature engineering.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(cell_data)

# Create unique key for fast joins
dt[, key_id := paste(id, year, sep = "_")]
setkey(dt, key_id)

# Precompute neighbor lookup as a long table instead of list
id_to_ref <- setNames(seq_along(id_order), id_order)
lookup_list <- vector("list", length(id_order))

for (ref_idx in seq_along(id_order)) {
  neighbors <- rook_neighbors_unique[[ref_idx]]
  if (length(neighbors) > 0) {
    ref_id <- id_order[ref_idx]
    neighbor_ids <- id_order[neighbors]
    lookup_list[[ref_idx]] <- data.table(
      ref_id = ref_id,
      neighbor_id = neighbor_ids
    )
  }
}

neighbor_pairs <- rbindlist(lookup_list)
rm(lookup_list)

# Expand neighbor pairs across all years
years <- unique(dt$year)
neighbor_pairs <- neighbor_pairs[, .(id = ref_id, neighbor_id, year = years), by = .(ref_id, neighbor_id)]
neighbor_pairs[, ref_key := paste(id, year, sep = "_")]
neighbor_pairs[, neighbor_key := paste(neighbor_id, year, sep = "_")]

# Join neighbor values once for all variables
setkey(neighbor_pairs, neighbor_key)
neighbor_pairs <- neighbor_pairs[dt, on = .(neighbor_key = key_id)]

# Compute stats in vectorized form for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

results_list <- lapply(neighbor_source_vars, function(var) {
  tmp <- neighbor_pairs[, .(
    max_val = max(get(var), na.rm = TRUE),
    min_val = min(get(var), na.rm = TRUE),
    mean_val = mean(get(var), na.rm = TRUE)
  ), by = ref_key]
  
  setnames(tmp, c("ref_key", paste0(var, "_max"), paste0(var, "_min"), paste0(var, "_mean")))
  tmp
})

# Merge all stats back into main table
all_stats <- Reduce(function(x, y) merge(x, y, by = "ref_key"), results_list)
dt <- merge(dt, all_stats, by.x = "key_id", by.y = "ref_key", all.x = TRUE)

# Drop helper columns
dt[, key_id := NULL]

# Now run Random Forest prediction as before
preds <- predict(trained_rf_model, newdata = as.data.frame(dt))
```

---

### **Expected Performance Gain**
- Eliminates millions of nested `lapply` calls.
- Uses `data.table` grouping and aggregation in C, drastically reducing runtime from 86+ hours to a few hours or less (depending on I/O).
- Preserves original estimand and trained model.

**Conclusion:** Reject colleagueâ€™s claim. The bottleneck is neighbor feature computation in pure R loops. Optimize by vectorizing and using `data.table` for joins and aggregation.
END_RESPONSE>>>
