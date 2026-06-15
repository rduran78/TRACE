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
run_id: OpenAI_F11_T07_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that Random Forest inference is the main bottleneck is incorrect. The code shows that before prediction, the pipeline spends significant time computing neighbor-based features for ~6.46 million rows over 28 years. Functions `build_neighbor_lookup` and `compute_neighbor_stats` execute large `lapply` loops involving repeated string concatenation, lookups, and subsetting, which are highly inefficient in R and scale poorly at this size. This preprocessing stepâ€”not model inferenceâ€”is the dominant time sink.

---

**Optimization Strategy:**  
- Avoid repeated string operations and `lapply` calls; use **vectorized joins** or **data.table merges**.
- Precompute neighbor relationships as a long table (cell-year â†’ neighbor-year).
- Compute max/min/mean with `data.table` aggregation instead of nested loops.
- Preserve the original estimand by computing the same statistics (max, min, mean) for each neighbor variable.

---

**Optimized R Code using `data.table`:**

```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Build long neighbor table (cell_id, year, neighbor_id)
id_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
nb_list <- rook_neighbors_unique

# Expand neighbors into a long format
neighbor_dt <- rbindlist(lapply(seq_along(nb_list), function(i) {
  if (length(nb_list[[i]]) == 0) return(NULL)
  data.table(ref_idx = i, neighbor_idx = nb_list[[i]])
}))
neighbor_dt[, id := id_order[ref_idx]]
neighbor_dt[, neighbor_id := id_order[neighbor_idx]]

# Create all cell-year â†’ neighbor-year pairs
years <- unique(cell_data$year)
expanded_neighbors <- neighbor_dt[, .(id, neighbor_id)][, .(year = years), by = .(id, neighbor_id)]

# Join to get neighbor values
expanded_neighbors <- merge(expanded_neighbors, cell_data[, .(neighbor_id = id, year, ntl, ec, pop_density, def, usd_est_n2)],
                            by = c("neighbor_id", "year"), all.x = TRUE)

# Aggregate neighbor stats for each variable
agg_stats <- expanded_neighbors[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Merge back to cell_data
cell_data <- merge(cell_data, agg_stats, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works:**  
- Eliminates nested `lapply` loops over millions of rows.
- Performs neighbor lookups and aggregations in a single vectorized operation.
- Uses efficient `data.table` joins and group-by, reducing complexity from hours to minutes.
- Preserves original estimands (max, min, mean for the 5 variables).

**Expected Outcome:**  
Preprocessing time drops from 86+ hours to under an hour on a standard laptop. Random Forest inference (already trained) remains a minor cost compared to feature engineering.
END_RESPONSE>>>
