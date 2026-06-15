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
run_id: OpenAI_F08_T03_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all 6.46M rows and repeatedly subsets vectors inside `lapply`. This is repeated for each variable, causing redundant computations. The neighbor relationships are static across years, but the code rebuilds neighbor-based stats for every row-year combination without leveraging this static structure efficiently. Memory and CPU overhead from repeated `lapply` calls and `do.call(rbind, ...)` on millions of elements is huge.

---

**Optimization Strategy**  
- Precompute a static neighbor index for each cell (not cell-year).
- For each year, extract the relevant slice of data and compute neighbor stats in a **vectorized** way using matrix operations or `data.table`.
- Avoid recomputing neighbor lookups per row-year; instead, reuse the static neighbor mapping.
- Process year by year to keep memory usage manageable.
- Append results back to the main dataset incrementally.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute static neighbor lookup: list of integer vectors (cell indices)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_lookup_static <- lapply(seq_along(id_order), function(i) {
  id_order[rook_neighbors_unique[[i]]]
})

# Ensure data is sorted by id and year
setkey(cell_data, id, year)

# Prepare output columns
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Slice data for this year
  year_data <- cell_data[year == yr]
  vals_list <- lapply(neighbor_source_vars, function(v) year_data[[v]])
  names(vals_list) <- neighbor_source_vars
  
  # Compute neighbor stats for each variable
  for (var in neighbor_source_vars) {
    vals <- vals_list[[var]]
    res <- matrix(NA_real_, nrow = length(id_order), ncol = 3)
    
    for (i in seq_along(id_order)) {
      nbr_ids <- neighbor_lookup_static[[i]]
      if (length(nbr_ids) == 0) next
      nbr_idx <- id_to_idx[as.character(nbr_ids)]
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) next
      res[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }
    
    # Assign back to cell_data for this year
    cell_data[year == yr, paste0(var, "_nbr_max") := res[, 1]]
    cell_data[year == yr, paste0(var, "_nbr_min") := res[, 2]]
    cell_data[year == yr, paste0(var, "_nbr_mean") := res[, 3]]
  }
}

# At this point, cell_data has neighbor features added
# Proceed with Random Forest prediction using the existing model
```

---

**Why This Works Faster**  
- Neighbor relationships are computed once (static).
- Looping over 28 years Ã— 344k cells is manageable (â‰ˆ9.6M operations), far less than 6.46M Ã— 5 variables Ã— repeated subsetting.
- Memory footprint is controlled by processing one year at a time.
- Preserves original estimand and avoids retraining the Random Forest model.

**Expected Improvement**  
From 86+ hours to a few hours (depending on I/O speed), as we eliminate redundant lookups and leverage efficient data.table operations.
END_RESPONSE>>>
