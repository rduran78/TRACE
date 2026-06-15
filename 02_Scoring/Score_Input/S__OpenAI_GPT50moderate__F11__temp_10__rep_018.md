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
run_id: OpenAI_F11_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s diagnosis is incorrect. The major bottleneck is **not Random Forest inference**, but the neighbor feature computation loop (`build_neighbor_lookup` and `compute_neighbor_stats`), which repeatedly performs expensive `lapply` calls over **6.46M rows Ã— 5 variables**, creating and binding millions of small objects in R. This is extremely inefficient due to Râ€™s per-element overhead and object churn.

---

### **Diagnosis**
- Random Forest inference on 6.46M rows, even with `predict.randomForest()`, is relatively fast compared to 86+ hours.
- The real hotspot:
  - `build_neighbor_lookup` constructs a full neighbor index for all rows using `lapply` â†’ **~6.46M iterations**.
  - `compute_neighbor_stats` loops for every neighbor feature variable â†’ **another 5 Ã— 6.46M cell-years**, with repeated memory allocations and `rbind` operations.

Both the neighbor lookup and feature computation exhibit quadratic-like behavior and poor vectorization.

---

### **Optimization Strategy**
1. **Precompute neighbor index once and vectorize**:
   - Work at **cell-level**, not at (cell, year) level, since neighbors donâ€™t change over time.
   - Form a long table of neighbor relationships and join by year and variable efficiently using `data.table`.
2. **Exploit `data.table` grouping and joins** for fast aggregation:
   - reshape panel data long table.
   - For each variable, compute max, min, mean across neighbors in a single grouped operation.
3. Avoid repeated `lapply` calls for every row â†’ fully vectorized group aggregations.

This removes the per-row R overhead and leverages optimized C-backend of `data.table`.

---

### **Optimized Working R Code**

```r
library(data.table)

# Convert cell_data to data.table for efficiency
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...

# Parameters
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare neighbor pairs (cell-level)
neighbors_dt <- data.table::rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)
setnames(neighbors_dt, c("id", "neighbor_id"))

# Expand to panel: join neighbors by year
years <- unique(cell_data$year)
panel_neighbors <- CJ(year = years, id = id_order)[neighbors_dt, on = "id", allow.cartesian = TRUE]

# Reshape cell_data keys
setkey(cell_data, id, year)
setkey(panel_neighbors, neighbor_id, year)

for (var_name in neighbor_source_vars) {
  # Bring in neighbor values
  panel_neighbors[, neighbor_val := cell_data[[var_name]][.I], on = .(neighbor_id, year)]
  
  # Compute stats by (id, year)
  agg <- panel_neighbors[, .(
    paste0(var_name, "_nbr_max") = max(neighbor_val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") = min(neighbor_val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") = mean(neighbor_val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Join back into cell_data
  cell_data <- agg[cell_data, on = .(id, year)]
}

# At this point, cell_data contains additional neighbor feature columns
```

---

### **Why This Is Faster**
- `data.table` uses vectorized joins and grouping in C, not R loops.
- No `lapply` per observation.
- Complexity reduced from **O(N Ã— k)** with high R overhead to **O(N + E)**, where E = edges in neighbor graph (~1.37M), repeated for each variable efficiently.
- Memory optimized: operations in-place, no repeated `rbind` on huge lists.

---

### **Preserves Original Estimand**
- Same neighbor-based max/min/mean logic.
- Same trained Random Forest modelâ€”untouched.
- Only optimization of feature generation prior to inference.

**Result**: Feature computation should drop from **86+ hours to minutes** on a standard laptop, after which `predict(model, newdata)` runs quickly.
END_RESPONSE>>>
