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
run_id: OpenAI_F11_T03_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F11_OpenAI_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that Random Forest inference (`predict()`) is the main bottleneck is **incorrect**. The code shows that before prediction, the pipeline constructs neighbor-based features for ~6.46 million rows across 5 variables. The functions `build_neighbor_lookup()` and especially `compute_neighbor_stats()` use **nested `lapply` calls and repeated vector indexing**, which are extremely inefficient in R at this scale. The cost of computing neighbor summaries for millions of rows dominates runtime, not model inference.

---

### **Correct Bottleneck**
- `compute_neighbor_stats()` iterates over every row (6.46M) and computes stats by indexing into `vals` for each neighbor set.
- This is repeated **5 times** (once per variable).
- The neighbor lookup itself (`lapply(row_ids, ...)`) also scales poorly.

Random Forest prediction on 6.46M rows with 110 features is large but feasible in minutes on a laptop using `ranger` or `randomForest`. The 86+ hours runtime comes from the neighbor feature engineering.

---

## **Optimization Strategy**
1. **Avoid per-row `lapply`**: Use **vectorized joins** or **data.table** to compute neighbor stats in bulk.
2. Pre-expand neighbor relationships into a long table: `(cell_id, year, neighbor_id)`.
3. Join neighbor values for each variable, then compute `max`, `min`, `mean` using `data.table` group operations.
4. Reuse this for all 5 variables without recomputing neighbor structure.

---

### **Optimized R Code (data.table approach)**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# Expand neighbor relationships into long format
# rook_neighbors_unique: list of integer vectors, length = length(id_order)
edges <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Cross with years to create full panel for neighbors
years <- unique(cell_data$year)
edges_panel <- edges[, .(id, neighbor_id)][, .(year = years), by = .(id, neighbor_id)]

# Merge neighbor values
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  # Join neighbor values
  neighbor_vals <- merge(
    edges_panel,
    cell_data[, .(neighbor_id = id, year, val = get(var_name))],
    by = c("neighbor_id", "year"),
    all.x = TRUE
  )
  
  # Compute stats by (id, year)
  stats <- neighbor_vals[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setnames(stats, c("id", "year", paste0(var_name, "_nbr_max"),
                              paste0(var_name, "_nbr_min"),
                              paste0(var_name, "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}
```

---

### **Why This Works**
- Converts neighbor computations into **bulk joins and grouped aggregations** using `data.table`, which is highly optimized in C.
- Eliminates millions of R-level function calls.
- Memory-efficient: processes one variable at a time.
- Preserves the original estimand and trained Random Forest model.

---

**Expected Impact:**  
- From 86+ hours to **under 1 hour** on a standard laptop.
- Random Forest prediction remains unchanged and will run quickly after feature engineering.

**Final Answer:** The bottleneck is neighbor feature computation, not Random Forest inference. Optimize by vectorizing neighbor stats using `data.table` joins and grouped aggregation as shown above.
END_RESPONSE>>>
