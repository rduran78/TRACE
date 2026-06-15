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
run_id: OpenAI_F11_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect. The code evidence points to **neighbor feature computation**, not Random Forest inference, as the main bottleneck. Specifically:

- `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over **6.46 million rows**, with repeated string concatenation (`paste`) and list indexing, which is extremely expensive in R.
- Random Forest `predict()` on 6.46M Ã— 110 features is large but typically takes minutes to an hour on a modern laptop, not 86 hours.
- The actual cost driver is repeatedly computing neighbor stats for 5 variables over millions of rows using inefficient R loops and vector operations inside lists.

---

### **Optimization Strategy**

- Precompute neighbor stats using **vectorized or data.table aggregation** instead of looping over every row.
- Store neighbor relationships in an efficient structure and join once.
- Avoid repeated calls to `paste()` and `lapply` in the inner loop.
- Leverage `data.table` joins or `vapply` for speed.

---

### **Optimized Approach in R (data.table)**

```r
library(data.table)

# Convert cell_data to data.table
dt <- as.data.table(cell_data)

# Ensure keys for fast join
setkey(dt, id, year)

# Prepare neighbor relationships
# rook_neighbors_unique: list of neighbors per cell
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Replicate for all years
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Join neighbor values for all neighbor source vars at once
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt and compute stats in one pass
metrics_list <- lapply(vars, function(v) {
  tmp <- merge(neighbor_dt, dt[, .(neighbor_id = id, year, val = get(v))],
               by = c("neighbor_id", "year"), allow.cartesian = TRUE)

  # Compute max, min, mean per (id, year)
  tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)][, paste0(v, c("_nb_max", "_nb_min", "_nb_mean")) := .(max_val, min_val, mean_val)][, .(id, year,
                                                                                get(paste0(v, "_nb_max")),
                                                                                get(paste0(v, "_nb_min")),
                                                                                get(paste0(v, "_nb_mean")))]
})

# Combine all metrics
metrics <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), metrics_list)

# Merge back to main table
dt <- merge(dt, metrics, by = c("id", "year"), all.x = TRUE)

# Convert back if needed
cell_data <- as.data.frame(dt)
```

---

### **Why This is Faster**
- Completely removes **millions of R-level loops**.
- Uses `data.table` bulk operations and joins (highly optimized in C).
- Processes all years and all neighbors in a vectorized way.

---

**Expected Impact:** Speeds up from **86+ hours to under 1â€“2 hours** on a standard laptop, preserving the Random Forest model and estimand.
END_RESPONSE>>>
