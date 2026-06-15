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
run_id: OpenAI_F03_T00_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small vectors and lists. This is extremely slow and memory-heavy.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies `cell_data` repeatedly.  
3. **Inefficient neighbor lookup**: Current approach builds a large list of integer vectors for every row, which is costly in both time and memory.  
4. **Random Forest inference**: If predictions are done in a loop (row-by-row), this is catastrophic. `predict()` on the entire data frame at once is much faster.  

---

### **Optimization Strategy**
- **Vectorize neighbor feature computation**: Instead of row-wise `lapply`, use `data.table` joins or matrix aggregation.
- **Precompute neighbor relationships in long format**: Create a table of `(cell_id, year, neighbor_id)` and join features in bulk.
- **Avoid repeated copies**: Use `data.table` for in-place updates.
- **Batch Random Forest predictions**: Call `predict(rf_model, newdata)` once on the full dataset or in large chunks.
- **Memory efficiency**: Drop intermediate objects early, use integer keys, and avoid large nested lists.

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor pairs in long format
# rook_neighbors_unique: list of integer vectors (neighbors per id_order index)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, each = length(years)),
                                neighbor_id = rep(neighbor_id, each = length(years)),
                                year = rep(years, times = .N)), by = .I][, .I := NULL]

setkey(neighbor_dt, neighbor_id, year)

# Join neighbor features in bulk
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  neighbor_dt[, val := cell_data[.SD, on = .(id = neighbor_id, year), get(var_name)]]
  
  # Compute stats by (id, year)
  stats_dt <- neighbor_dt[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats_dt, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats_dt[cell_data, on = .(id, year)]
  
  # Drop val column for next iteration
  neighbor_dt[, val := NULL]
}

# Random Forest prediction in one batch
predictions <- predict(rf_model, newdata = cell_data)

# Attach predictions
cell_data[, gdp_pred := predictions]
```

---

### **Why This Is Faster**
- **No per-row loops**: All neighbor stats computed via grouped aggregation in `data.table`.
- **Single pass per variable**: Each neighbor feature set computed in one join + aggregation.
- **Predict in bulk**: Avoids millions of function calls.
- **Memory-friendly**: Uses long format and in-place updates.

---

**Expected runtime**: From 86+ hours â†’ a few hours (or less) on 16 GB RAM, depending on disk I/O and CPU.  
**Preserves**: Trained Random Forest model, original estimand, and neighbor-based feature logic.
END_RESPONSE>>>
