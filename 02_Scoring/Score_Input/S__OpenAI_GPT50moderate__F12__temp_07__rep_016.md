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
run_id: OpenAI_F12_T07_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is incorrect. The dominant bottleneck is not `do.call(rbind, result)` or repeated list-bindingâ€”itâ€™s the **inner loop logic repeatedly scanning `vals` and computing statistics for ~6.46M rows Ã— 5 variables**, which is computationally expensive in R due to millions of `lapply` calls and repeated subsetting. The `rbind` step is negligible compared to those lookups.

### **Diagnosis**
- For each row (6.46M), `compute_neighbor_stats()`:
  - Extracts `idx` (neighbor indices).
  - Subsets `vals` (vector length = 6.46M).
  - Filters `NA`, then computes `max`, `min`, and `mean`.
- This is done **5 times** for the 5 variables â†’ ~32M expensive R-level operations.
- `do.call(rbind, result)` happens once per variable and is trivial in comparison.
- Root cause: *pure R loops over millions of elements with repeated indexing*. Vectorization or compiled backends would drastically outperform this.

---

### **Optimization Strategy**
- Precompute neighbor index list once (already done).
- Replace row-wise R loops with **vectorized aggregation** using `data.table` or matrix-based computation.
- Compute all 5 variables in a **single grouped operation** by exploding neighbor relationships into a long format table and summarizing with fast aggregators.
- Preserve the trained model and estimand by ensuring numerical equivalence of neighbor stats.

---

### **Optimized Approach**
- Convert neighbor relationships into a long `data.table`.
- Join on `data` to bring in variable values.
- Aggregate `max`, `min`, and `mean` per focal cell-year for each variable.
- Reshape back to wide format and merge into original `cell_data`.

---

### **Working R Code**

```r
library(data.table)

# Convert input data to data.table
setDT(cell_data)

# Build long neighbor table: each row = (focal_row, neighbor_row)
row_keys <- paste(cell_data$id, cell_data$year, sep = "_")
idx_lookup <- setNames(seq_len(nrow(cell_data)), row_keys)

neighbor_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(focal = i, neighbor = neighbor_lookup[[i]])
}))

# Bring in neighbor values for all source vars
neighbor_vals <- melt(
  cell_data[, c("id", "year", neighbor_source_vars), with = FALSE][
    , row_id := .I],
  id.vars = c("row_id", "id", "year"),
  measure.vars = neighbor_source_vars,
  variable.name = "var_name",
  value.name = "val"
)

# Join neighbor_dt with neighbor_vals
neighbor_long <- merge(
  neighbor_dt, neighbor_vals,
  by.x = "neighbor", by.y = "row_id",
  allow.cartesian = TRUE
)

# Compute stats per focal and var_name
stats <- neighbor_long[
  !is.na(val),
  .(max_val = max(val), min_val = min(val), mean_val = mean(val)),
  by = .(focal, var_name)
]

# Reshape back to wide format
stats_wide <- dcast(
  stats, focal ~ var_name,
  value.var = c("max_val", "min_val", "mean_val")
)

# Merge back into cell_data
cell_data <- cbind(cell_data, stats_wide[match(seq_len(nrow(cell_data)), focal), -"focal"])
```

---

### **Expected Performance Gains**
- Eliminates millions of R-level function calls.
- Uses highly optimized C-backed `data.table` aggregation.
- Reduces runtime from **86+ hours â†’ under 1 hour** on 16 GB RAM laptop (empirically observed in similar pipelines).
- Preserves all original estimands and avoids retraining the Random Forest.

**Summary:** Reject the claim about `do.call(rbind, result)`. The real bottleneck is repeated row-wise neighbor stat computation in R. Solve by vectorized neighbor aggregation using `data.table`.
END_RESPONSE>>>
