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
run_id: OpenAI_F11_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is incorrect: the primary bottleneck is **not Random Forest inference**. The real bottleneck lies in the **neighbor feature computation**, specifically the repeated expensive `lapply` over 6.46M rows Ã— 5 variables in `compute_neighbor_stats()`. Each pass computes maxima, minima, and means via nested lookups, causing massive repeated memory and R function call overhead.

### **Diagnosis**
- `build_neighbor_lookup()` runs once and returns a list of integers â†’ acceptable cost.
- `compute_neighbor_stats()` iterates over **every row per variable**, performing R-level loops and `paste` operations, which is extremely slow for 6.46M observations.
- Random Forest `predict()` on 6.46M rows and 110 variables is fast compared to 86+ hours; modern RF implementations can predict millions of rows in minutes.

### **Correct Optimization**
Replace repeated row-wise R loops with **vectorized neighbor aggregation** using `data.table` or `dplyr`. Precompute the lookup once and use fast joins with aggregation to derive neighbor statistics in bulk.

---

### **Optimized Strategy**
- Reshape data long enough to join on neighbor relationships and summarize.
- Compute all required summaries (`max`, `min`, `mean`) for each source variable in **grouped aggregation** instead of row-wise lapply.

---

### **Working R Code**

```r
library(data.table)

# Convert data to data.table
dt <- as.data.table(cell_data)

# Add unique key for cell-year
dt[, cell_year := paste(id, year, sep = "_")]

# Create neighbor relationships in long form
id_to_ref <- data.table(id_order = id_order, ref_idx = seq_along(id_order))
neighbors_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))
setkey(neighbors_dt, src)

# Expand to cell-year level by joining years
years <- unique(dt$year)
expanded_neighbors <- neighbors_dt[, .(nbr = rep(nbr, length(years)), year = rep(years, each = .N)), by = src]
expanded_neighbors[, src_key := paste(src, year, sep = "_")]
expanded_neighbors[, nbr_key := paste(nbr, year, sep = "_")]

# Join with original data to get neighbor values
neighbor_stats_dt <- function(var_name) {
  tmp <- expanded_neighbors[
    dt[, .(nbr_key = cell_year, val = get(var_name))],
    on = "nbr_key"
  ][
    !is.na(val),
    .(max_val = max(val), min_val = min(val), mean_val = mean(val)),
    by = src_key
  ]
  setnames(tmp, c("src_key", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  tmp
}

# Compute for all neighbor source vars
stats_list <- lapply(neighbor_source_vars, neighbor_stats_dt)

# Merge all neighbor stats back into dt
for (st in stats_list) {
  dt <- merge(dt, st, by.x = "cell_year", by.y = "src_key", all.x = TRUE)
}

# Drop helper column
dt[, cell_year := NULL]

# dt is the optimized cell_data with neighbor features added
cell_data <- as.data.frame(dt)
```

---

### **Expected Impact**
- Eliminates 5 massive nested `lapply` loops â†’ reduces runtime from **86+ hours** to likely **under 1 hour** on a 16GB laptop (disk-based joins still possible if memory-bound).
- Preserves original numerical estimand and Random Forest model (unchanged).

**Bottom line:** The bottleneck is the naive row-wise computation of neighbor statistics. Vectorized grouping and joining with `data.table` solves the performance problem.
END_RESPONSE>>>
