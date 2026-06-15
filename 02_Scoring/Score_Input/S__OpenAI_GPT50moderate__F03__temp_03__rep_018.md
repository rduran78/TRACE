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
run_id: OpenAI_F03_T03_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` and object copying in `build_neighbor_lookup` and `compute_neighbor_stats`. These functions create large lists and repeatedly subset vectors, which is inefficient for millions of rows. Additionally:  
- `neighbor_lookup` is a list of length equal to rows (â‰ˆ6.46M), consuming huge memory.  
- `compute_neighbor_stats` repeatedly allocates and binds results, causing overhead.  
- Random Forest inference is typically fast if the model is in memory, but prediction on millions of rows is slowed by inefficient feature preparation.  

**Optimization Strategy**  
1. **Avoid per-row `lapply`**: Use vectorized operations or precompute neighbor indices in a compact structure (e.g., data.table or matrix).  
2. **Use data.table for joins and aggregations**: Compute neighbor stats by merging on `(id, year)` instead of looping.  
3. **Reduce memory footprint**: Store neighbor relationships in long format and aggregate with `max`, `min`, `mean` via fast grouping.  
4. **Keep Random Forest loaded once**: Use `predict(model, newdata, ...)` on a prepared data.table.  
5. **Parallelize aggregation**: Use `data.table` or `future` for parallel grouping.  

---

### **Working R Code (Optimized)**

```r
library(data.table)
library(randomForest)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order and rook_neighbors_unique are given
# rf_model is the trained Random Forest model already loaded

# Convert cell_data to data.table for efficiency
setDT(cell_data)

# Build neighbor relationships in long format
# rook_neighbors_unique: list of neighbors per id_order index
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id = rep(id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .(id, neighbor_id)]
neighbor_dt <- neighbor_dt[, .(id, year, neighbor_id)]

# Merge neighbor values
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  vals <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  merged <- merge(neighbor_dt, vals, by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Compute stats per (id, year)
  stats <- merged[, .(
    paste0(var_name, "_nbr_max") = max(val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") = min(val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# Prepare predictors (ensure columns match rf_model)
predictors <- setdiff(names(cell_data), c("id", "year", "target_var")) # adjust target_var name

# Random Forest prediction
predictions <- predict(rf_model, newdata = cell_data[, ..predictors])

# Attach predictions
cell_data[, gdp_pred := predictions]
```

---

**Why this is faster**  
- Eliminates 6.46M-row `lapply` loops.  
- Uses `data.table` merges and group operations (highly optimized in C).  
- Neighbor stats computed in bulk rather than per-row.  
- Preserves original estimand and Random Forest model.  

**Expected improvement**  
From 86+ hours to a few hours (or less) on a 16 GB machine, depending on disk I/O and parallelization.
END_RESPONSE>>>
