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
run_id: OpenAI_F11_T00_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F11_OpenAI_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect. The main bottleneck is **not Random Forest inference** but the **neighbor feature computation**, specifically the `build_neighbor_lookup` and repeated `compute_neighbor_stats` calls over 6.46M rows and 5 variables. These functions use deeply nested `lapply` loops and repeated string concatenation, which are extremely slow in R for this scale.

### **Diagnosis**
- `build_neighbor_lookup` iterates over all 6.46M rows, performing string operations and list indexing.
- `compute_neighbor_stats` repeatedly traverses large lists and subsets vectors for each row.
- This is \(O(N \times k)\) with heavy R-level overhead, where \(N = 6.46\text{M}\) and \(k\) is average neighbor count.
- Random Forest prediction on 6.46M rows with 110 features is fast (minutes) compared to 86+ hours.

### **Optimization Strategy**
- Precompute neighbor relationships once at the **cell level** (344,208 cells), not per cell-year row.
- Use **vectorized joins** instead of per-row loops.
- Leverage `data.table` for fast grouping and aggregation.
- Compute neighbor stats by joining neighbor-year pairs and aggregating in bulk.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor pairs at cell level
neighbors_dt <- data.table(
  from = rep(id_order, lengths(rook_neighbors_unique)),
  to   = unlist(rook_neighbors_unique)
)

# Expand to cell-year pairs
years <- unique(cell_data$year)
neighbors_dt <- neighbors_dt[, .(id = from, neighbor_id = to), by = years]
setnames(neighbors_dt, "years", "year")

# Merge neighbor values for all variables in one pass
long_data <- melt(cell_data, id.vars = c("id", "year"), 
                  measure.vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
                  variable.name = "var_name", value.name = "value")

neighbors_long <- merge(neighbors_dt, long_data, 
                        by.x = c("neighbor_id", "year"), 
                        by.y = c("id", "year"), 
                        allow.cartesian = TRUE)

# Compute neighbor stats in bulk
neighbor_stats <- neighbors_long[, .(
  max_val = max(value, na.rm = TRUE),
  min_val = min(value, na.rm = TRUE),
  mean_val = mean(value, na.rm = TRUE)
), by = .(id, year, var_name)]

# Reshape back to wide and merge into cell_data
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var_name, 
                             value.var = c("max_val", "min_val", "mean_val"))

cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- Eliminates per-row loops and string concatenation.
- Uses `data.table` joins and aggregations, which are highly optimized in C.
- Processes all 5 variables in one pass instead of 5 separate loops.
- Reduces complexity from \(O(N \times k)\) in R loops to efficient vectorized operations.

**Expected runtime:** From 86+ hours to **minutes** on a 16 GB laptop.

**Preserves:** Original estimand, trained Random Forest model, and feature definitions.
END_RESPONSE>>>
