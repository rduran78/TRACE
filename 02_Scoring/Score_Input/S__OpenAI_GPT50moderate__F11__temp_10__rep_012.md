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
run_id: OpenAI_F11_T10_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that **Random Forest inference** is the main bottleneck is incorrect. The evidence shows that most time will be spent in building and using the neighbor lookup for ~6.46 million rows across multiple variables. Specifically:

- `build_neighbor_lookup` iterates over **`344,208` cells**, creating lists of index vectors.
- `compute_neighbor_stats` later applies expensive `lapply` across **6.46 million rows Ã— 5 variables**, each involving subsetting and aggregation.
- These loops are pure R and not vectorized, making them extremely slow for this dataset size.
  
Random Forest prediction on 6.46M rows is heavy but usually **hours, not 86+ hours**, especially since the model is already trained and loaded once. The dominant cost is neighbor feature engineering in R lists and repeated loops.

---

### **Optimization Strategy**
- Compute neighbor statistics in a **vectorized or data.table manner**, not row-wise in R loops.
- Precompute neighbor relationships and join instead of calculating per row dynamically.
- Avoid repeated `lapply` over millions of rows; pivot to `data.table` joins or matrix operations.

---

### **Optimized R Code (using `data.table`)**

```r
library(data.table)

# Convert to data.table
cell_data_dt <- as.data.table(cell_data)

# Precompute neighbor pairs for all years
expand_neighbors <- function(ids, neighbors, years) {
  # ids: vector of cell ids
  # neighbors: list of neighbor indices
  nbl <- rbindlist(lapply(seq_along(neighbors), function(i) {
    if (length(neighbors[[i]]) == 0) return(NULL)
    data.table(
      id    = ids[i],
      neigh = ids[neighbors[[i]]]
    )
  }))
  nbl <- nbl[CJ(id = id, year = years, unique = TRUE), on = .(id), allow.cartesian = TRUE]
  setnames(nbl, c("id", "year", "neigh_year", "neigh"), c("id", "year", "id2_year", "id2"))
  return(nbl[, .(id, year, id2)])  # id-year with neighbor id2
}

id_order_vec <- id_order
years_vec <- sort(unique(cell_data$year))

neighbor_pairs <- expand_neighbors(id_order_vec, rook_neighbors_unique, years_vec)

# Join neighbor values for all variables and aggregate
compute_neighbor_features_dt <- function(dt, pairs, var) {
  dt_sub <- dt[, .(id, year, value = get(var))]
  setnames(dt_sub, "value", var)
  
  dt_pairs <- merge(pairs, dt_sub, by.x = c("id2", "year"), by.y = c("id", "year"), all.x = TRUE)
  agg <- dt_pairs[, .(
    max_val = max(get(var), na.rm = TRUE),
    min_val = min(get(var), na.rm = TRUE),
    mean_val = mean(get(var), na.rm = TRUE)
  ), by = .(id, year)]
  
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste(var, c("nb_max", "nb_min", "nb_mean"), sep = "_"))
  
  merge(dt, agg, by = c("id", "year"), all.x = TRUE)
}

# Apply for all neighbor source variables
for (v in neighbor_source_vars) {
  cell_data_dt <- compute_neighbor_features_dt(cell_data_dt, neighbor_pairs, v)
}

# cell_data_dt now augmented with neighbor stats
```

---

### **Why This Works**
- Moves from **6.46M row-wise iterative lookups** to **fast keyed joins with `data.table`**.
- Avoids repeatedly parsing neighbor lists inside loops.
- Aggregation is vectorized using `data.table`â€™s `by` groups.
- Memory efficient since operations occur in place.

---

**Expected Impact:** Reduction from **86+ hours to a few hours or less**, as we eliminate the Python-like R loops and use join + aggregate instead of millions of function calls.

**Model inference stays untouched** (still calls `predict()` on final dataset).
END_RESPONSE>>>
