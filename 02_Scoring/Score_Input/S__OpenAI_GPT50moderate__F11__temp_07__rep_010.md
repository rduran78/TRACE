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
run_id: OpenAI_F11_T07_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The evidence points to the real bottleneck being **neighbor feature computation**, not Random Forest inference. The `build_neighbor_lookup` and `compute_neighbor_stats` functions use nested `lapply` calls on millions of rows, repeatedly performing expensive vector lookups and string operations. With ~6.46M rows and 5 variables, this dominates the runtime. Random Forest prediction on 6.46M rows with 110 predictors is relatively fast compared to 86+ hours, so the colleagueâ€™s diagnosis is incorrect.

---

### **Diagnosis**
- `build_neighbor_lookup` constructs a list of neighbor indices for **every row** (â‰ˆ6.46M), using string concatenation and hashing for lookups.
- `compute_neighbor_stats` iterates through the same list multiple times (once per variable), repeatedly scanning neighbors and computing summary stats.
- This is highly inefficient for large panel data and explains the 86+ hour runtime.

---

### **Optimization Strategy**
- Avoid recomputing neighbor stats per row with repeated `lapply`.
- Reshape data into a **wide format by year**, join neighbor relationships once, and compute aggregations using **vectorized operations**.
- Use **data.table** for fast joins and grouped summaries.
- Reuse the same neighbor relationships across variables without rebuilding lookup structures.
- Leave Random Forest model and predictions unchanged.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Ensure keys for fast joins
setkey(dt, id, year)

# Load neighbor relationships (rook) into a data.table
# rook_neighbors_unique: list of integer vectors, index by cell id
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Cross with years to create full neighbor mapping for all years
years <- unique(dt$year)
edges_year <- CJ(year = years, id = edges$id)[edges, on = "id", allow.cartesian = TRUE]
setnames(edges_year, "neighbor_id", "neighbor")

# Reshape and merge neighbor values for each variable
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_vars) {
  edges_year[, (var) := dt[.SD, on = .(id = neighbor, year), get(var)]]
  
  # Compute stats by id-year
  stats <- edges_year[, .(
    max_val = max(get(var), na.rm = TRUE),
    min_val = min(get(var), na.rm = TRUE),
    mean_val = mean(get(var), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Replace NaN with NA
  for (col in names(stats)[3:5]) {
    set(stats, which(!is.finite(stats[[col]])), col, NA)
  }
  
  # Merge back into main dt
  setnames(stats, c("id", "year", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  dt <- stats[dt, on = .(id, year)]
}

# dt now contains original features + neighbor features
# Predict with pre-trained Random Forest
preds <- predict(rf_model, newdata = dt)
```

---

### **Why This Works**
- Eliminates per-row `lapply` and string concatenation.
- Aggregates neighbor stats in **chunked vectorized operations** using `data.table`.
- Handles all years and variables in a **single pass per variable**, reducing complexity from O(N Ã— neighbors Ã— variables) with heavy R loops to O(N) with optimized joins and group-bys.
- Preserves the trained Random Forest model and numerical estimand.

---

**Expected runtime improvement:** From 86+ hours to **under an hour**, largely limited by disk I/O and memory bandwidth, assuming 16 GB RAM and efficient `data.table` operations.
END_RESPONSE>>>
