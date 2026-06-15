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
run_id: OpenAI_F11_T03_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F11_OpenAI_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the main bottleneck is incorrect. The provided code shows that the pipeline spends significant time in `build_neighbor_lookup()` and `compute_neighbor_stats()`. These functions use deeply nested `lapply()` calls over 6.46 million rows and repeatedly manipulate large vectors and lists in R, which is highly inefficient. Random Forest prediction on 6.46M rows with 110 predictors is relatively fast compared to these repeated R-level loops.

**Correct Bottleneck:**  
Neighbor feature computation dominates runtime due to:
- Inefficient repeated list lookups and string concatenations.
- Lack of vectorization; heavy use of `lapply()` over millions of elements.

**Optimization Strategy:**  
- Precompute neighbor indices as integer vectors without repeated string operations.
- Use `data.table` for fast joins and aggregations instead of per-row `lapply()`.
- Compute all neighbor stats in a single grouped operation rather than looping over variables.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create a unique key for each cell-year
cell_data[, cell_year_key := .I]  # row index as key

# Expand neighbor relationships across years
# rook_neighbors_unique: list of integer vectors (neighbors per cell)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Join with years to create full neighbor pairs for all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Map to cell-year rows
cell_data[, id_year := paste(id, year, sep = "_")]
neighbor_dt[, id_year := paste(id, year, sep = "_")]
neighbor_dt[, neighbor_id_year := paste(neighbor_id, year, sep = "_")]

# Lookup row indices
idx_map <- cell_data[, .(id_year, cell_year_key)]
neighbor_dt <- merge(neighbor_dt, idx_map, by = "id_year", all.x = TRUE)
setnames(neighbor_dt, "cell_year_key", "row_idx")
neighbor_dt <- merge(neighbor_dt, idx_map, by.x = "neighbor_id_year", by.y = "id_year", all.x = TRUE)
setnames(neighbor_dt, "cell_year_key", "neighbor_idx")

# Drop incomplete
neighbor_dt <- neighbor_dt[!is.na(row_idx) & !is.na(neighbor_idx)]

# Compute neighbor stats for all variables efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, value := vals[neighbor_idx]]
  
  stats <- neighbor_dt[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = row_idx]
  
  # Merge back into cell_data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by.x = "cell_year_key", by.y = "row_idx", all.x = TRUE)
}

# Drop helper columns
cell_data[, c("cell_year_key", "id_year") := NULL]
```

---

### **Why This Works**
- Eliminates per-row `lapply()` and string concatenation inside loops.
- Uses `data.table` joins and grouped aggregations for speed.
- Computes neighbor stats in bulk rather than row-wise.
- Memory-efficient: avoids creating millions of small lists.

**Expected Impact:**  
This approach reduces runtime from **86+ hours to a few hours or less** on a standard laptop, without changing the Random Forest model or the numerical estimand.
END_RESPONSE>>>
