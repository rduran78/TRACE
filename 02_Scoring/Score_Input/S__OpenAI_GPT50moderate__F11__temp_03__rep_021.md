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
run_id: OpenAI_F11_T03_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F11_OpenAI_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect. The main bottleneck is **neighbor feature computation**, not Random Forest inference. Evidence:  
- `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over **6.46M rows Ã— 5 variables**, causing massive R-level loops and object allocations.  
- Random Forest inference on 6.46M rows with 110 predictors is heavy but typically minutes to a few hours, not 86+ hours.  
- The nested `lapply` calls dominate runtime due to repeated indexing and vector operations in pure R.

---

### **Diagnosis**
The bottleneck is the repeated per-row neighbor aggregation in `compute_neighbor_stats`. Each call iterates over millions of rows and performs lookups in R lists, which is extremely slow and memory-inefficient.

---

### **Optimization Strategy**
- Precompute neighbor indices once (already done).
- Replace R loops with **vectorized or compiled operations** (e.g., `data.table` or `Rcpp`).
- Compute all neighbor stats in a single pass using efficient joins instead of per-row `lapply`.

---

### **Optimized Approach**
Use `data.table` to melt neighbor relationships into long format and aggregate in C-optimized code.

---

### **Working R Code**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Create neighbor pairs (cell_id-year -> neighbor_id-year)
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
neighbors_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(ref_idx = i, neighbor_idx = rook_neighbors_unique[[i]])
}))
neighbors_dt <- merge(neighbors_dt, id_order_dt, by = "ref_idx")
setnames(neighbors_dt, "id", "cell_id")
neighbors_dt <- merge(neighbors_dt, id_order_dt, by.x = "neighbor_idx", by.y = "ref_idx")
setnames(neighbors_dt, "id", "neighbor_id")
neighbors_dt[, c("ref_idx", "neighbor_idx") := NULL]

# Expand to panel: join by year
years <- unique(cell_data$year)
neighbors_panel <- neighbors_dt[, .(cell_id, neighbor_id)][, .(year = years), by = .(cell_id, neighbor_id)]
setkey(neighbors_panel, neighbor_id, year)

# Join neighbor values for all variables
long_data <- melt(cell_data, id.vars = c("id", "year"), measure.vars = neighbor_source_vars)
setnames(long_data, c("id", "year", "variable", "value"), c("neighbor_id", "year", "variable", "value"))

neighbors_panel <- neighbors_panel[long_data, on = .(neighbor_id, year)]
neighbors_panel <- neighbors_panel[!is.na(value)]

# Compute stats by (cell_id, year, variable)
agg_stats <- neighbors_panel[, .(
  max_val = max(value, na.rm = TRUE),
  min_val = min(value, na.rm = TRUE),
  mean_val = mean(value, na.rm = TRUE)
), by = .(cell_id, year, variable)]

# Reshape back to wide and merge
agg_wide <- dcast(agg_stats, cell_id + year ~ variable, value.var = c("max_val", "min_val", "mean_val"))
setnames(agg_wide, old = c("cell_id"), new = c("id"))
cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Fixes the Bottleneck**
- Eliminates per-row `lapply` loops.
- Uses `data.table` joins and aggregations in C for speed.
- Processes all variables in one pass.
- Scales to millions of rows efficiently.

---

**Expected runtime:** From 86+ hours â†’ **minutes to under an hour** on a 16 GB laptop.  
Random Forest inference remains unchanged and will now dominate runtime, but still far less than original neighbor computation.
END_RESPONSE>>>
