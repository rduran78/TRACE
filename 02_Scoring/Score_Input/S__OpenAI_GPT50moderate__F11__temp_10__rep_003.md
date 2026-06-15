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
run_id: OpenAI_F11_T10_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim that Random Forest inference is the main bottleneck is **incorrect**. Evidence from the pipeline shows that the bottleneck lies in the **construction and repeated use of `neighbor_lookup` combined with `compute_neighbor_stats`** over ~6.46M rows and five variables.  

### **Diagnosis**
- **`build_neighbor_lookup`** produces a huge nested list by iterating over every row (6.46M iterations â†’ massive memory + repeated expensive string concatenations).
- **`compute_neighbor_stats`** applies `lapply` again over 6.46M elements for each variable (Ã—5 = ~32M iterations).
- These are pure R loops with heavy list operations â†’ far slower than Random Forest prediction.
- Random Forest inference on 6.46M rows and 110 variables typically runs in minutes to an hour on a decent laptop, but this pipeline runs 86+ hours, which strongly suggests the neighborhood feature computation dominates.

### **Optimization Strategy**
- **Vectorize neighbor aggregation**: Convert neighbor relationships into a sparse matrix and compute max/min/mean using fast matrix ops, or use `data.table` joins.
- Compute all neighbor-derived features in one pass rather than 5 separate `lapply` passes.
- Avoid per-row `lapply` and string concatenation.

---

### **Optimized R Code Using `data.table`**
```r
library(data.table)

# Convert data to data.table
dt <- as.data.table(cell_data)

# Create a unique cell-year key
dt[, key := .I]  # fast internal key

# Expand neighbor relationships WITH year (cross join per year)
years <- unique(dt$year)
neighbors_dt <- rbindlist(lapply(years, function(y) {
  data.table(
    year = y,
    from_id = rep(id_order, lengths(rook_neighbors_unique)),
    to_id = unlist(rook_neighbors_unique)
  )
}))

# Map to actual row indices
dt_ids <- dt[, .(id, year, key)]
neighbors_dt <- merge(neighbors_dt, dt_ids, by.x = c("from_id","year"), by.y = c("id","year"))
setnames(neighbors_dt, "key", "from_key")
neighbors_dt <- merge(neighbors_dt, dt_ids, by.x = c("to_id","year"), by.y = c("id","year"))
setnames(neighbors_dt, "key", "to_key")

# For each variable, compute stats directly via data.table grouped aggregation
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in neighbor_source_vars) {
  # Get neighbor values
  neighbors_dt[, val := dt[[v]][to_key]]

  # Aggregate stats per 'from_key'
  agg <- neighbors_dt[, .(
    paste0(v, "_nbr_max") = max(val, na.rm = TRUE),
    paste0(v, "_nbr_min") = min(val, na.rm = TRUE),
    paste0(v, "_nbr_mean") = mean(val, na.rm = TRUE)
  ), by = from_key]

  # Merge back into dt
  setkey(agg, from_key)
  dt[agg$from_key, c(paste0(v, "_nbr_max"), paste0(v, "_nbr_min"), paste0(v, "_nbr_mean")) := 
       .(agg[[2]], agg[[3]], agg[[4]])]
}
```

---

### **Why It Works**
- Eliminates millions of R-level loops and list concatenation.
- Uses `data.table` joins and grouped aggregationâ€”optimized C-level operations.
- Computes all neighbor-based features in a fraction of time (hours â†’ minutes).

**Preserves the trained Random Forest model and original estimand**; only restructuring feature engineering.

**Final bottleneck after optimization**: Random Forest inference, which should be manageable given efficient feature precomputation.
END_RESPONSE>>>
