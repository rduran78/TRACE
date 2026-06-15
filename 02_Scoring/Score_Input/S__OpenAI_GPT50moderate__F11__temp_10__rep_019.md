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
run_id: OpenAI_F11_T10_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is **not** Random Forest inference. Loading a pre-trained RF model and calling `predict()` on ~6.46M rows with 110 predictors is significant but generally feasible within minutes on a modern laptop. The true bottleneck is the **neighbor statistics computation**:

- `build_neighbor_lookup()` returns a list of length equal to cell-year rows (~6.46M).  
- `compute_neighbor_stats()` iterates over this list for each of 5 variables.  
- Each `lapply` call repeatedly performs vector indexing and aggregation inside R loopsâ€”effectively *5 Ã— 6.46M small operations in pure R*.  

This explains the **86+ hours runtime**: itâ€™s dominated by repeated interpreted-R loops, not prediction.

---

### **Optimization Strategy**

- Construct neighbor stats per **cellâ€“year** row in a **vectorized data.table join** or collapse computation to **matrix operations** rather than R loops.
- Precompute lookup indices once and reuse efficiently.
- Avoid applying five separate `lapply` passesâ€”compute all neighbor summaries in a single grouped aggregation.
- Use `data.table` for speed and memory efficiency.

---

### **Optimized Approach**

1. Convert data to `data.table`.
2. Create long-form table of (cell_id, year, neighbor_id).
3. Join neighbor values from the main table for all five variables.
4. Aggregate by (cell_id, year) to compute max, min, mean in one pass.
5. Merge back to original data.

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of neighbors indexed by id position in id_order

# 1. Expand neighbor relationships by year
id_order_dt <- data.table(id = id_order, pos = seq_along(id_order))
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(from = id_order[i], to = id_order[rook_neighbors_unique[[i]]])
}))
# Cartesian join on years
years <- unique(cell_data$year)
edges_expanded <- edges[, .(year = years), by = .(from, to)]

# 2. Join neighbor values
cell_data_keyed <- copy(cell_data)
setkey(cell_data_keyed, id, year)
setkey(edges_expanded, to, year)

neighbor_values <- cell_data_keyed[edges_expanded, nomatch = 0L]

# 3. Aggregate neighbor stats for all variables together
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_list <- c(
  lapply(vars, function(v) as.call(list(as.name("max"), as.name(v), na.rm = TRUE))),
  lapply(vars, function(v) as.call(list(as.name("min"), as.name(v), na.rm = TRUE))),
  lapply(vars, function(v) as.call(list(as.name("mean"), as.name(v), na.rm = TRUE)))
)
names(agg_list) <- c(paste0(vars, "_nbr_max"),
                     paste0(vars, "_nbr_min"),
                     paste0(vars, "_nbr_mean"))

neighbor_stats <- neighbor_values[, eval(as.call(c(quote(.SD), agg_list))), by = .(id = from, year)]

# 4. Merge back to main data
cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)
```

---

### **Expected Impact**

- Replaces **5 Ã— 6.46M interpreted loops** with one highly optimized `data.table` group aggregation.
- Expected runtime: minutes to under an hour instead of 86+ hours.
- Random Forest prediction stays as is:  
  ```r
  preds <- predict(trained_rf, newdata = cell_data)
  ```

**Summary:** Reject the claim about RF inference. The bottleneck is neighbor feature computation via `lapply`. **Solution:** vectorize using `data.table` joins and grouped aggregation.  
This preserves numeric fidelity, the original RF model, and the estimand while cutting runtime by orders of magnitude.
END_RESPONSE>>>
