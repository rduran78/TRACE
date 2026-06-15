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
run_id: OpenAI_F11_T07_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect. The main bottleneck is **not Random Forest inference**; itâ€™s the neighbor feature computation implemented with deeply nested `lapply()` calls over ~6.46M rows and repeated list indexing, which is extremely inefficient in R.

### **Diagnosis**
- `build_neighbor_lookup()` constructs a list of neighbor indices for every row (6.46M iterations), using repeated string pasting and hash lookups.
- `compute_neighbor_stats()` iterates again over the same large list for every neighbor variable (5 times), performing repeated subsetting and aggregation.
- Complexity: O(N Ã— avg_neighbors Ã— num_vars) across millions of rows â†’ dominates runtime.
- Random Forest prediction on 6.46M rows with 110 predictors is large but typically minutes to an hour on a laptopâ€”not 86+ hours.

### **Optimization Strategy**
- Avoid per-row `lapply()` and string manipulation.
- Precompute neighbor relationships as a **long table** (row â†’ neighbor) and use **vectorized joins and aggregations** via `data.table` or `dplyr`.
- Compute all neighbor statistics in one pass using fast group operations.
- Preserve the Random Forest model; only refactor feature engineering.

---

### **Optimized R Code (using `data.table`)**

```r
library(data.table)

# Convert to data.table
dt <- as.data.table(cell_data)

# Assume id and year columns exist
# Expand rook neighbor list into long format
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Join with years to create full panel edges
years <- unique(dt$year)
edges_panel <- CJ(year = years, id = edges$id)[
  , neighbor_id := edges$neighbor_id[match(id, edges$id)]
][!is.na(neighbor_id)]

# Merge neighbor values
setkey(dt, id, year)
setkey(edges_panel, neighbor_id, year)
edges_panel <- edges_panel[dt, on = .(neighbor_id = id, year)]
# edges_panel now has: id, year, neighbor_id, var values

# Compute stats for each source variable in one pass
neighbor_source_vars <- c("ntl","ec","pop_density","def","usd_est_n2")

stats <- edges_panel[,
  c(
    lapply(.SD, max, na.rm = TRUE),
    lapply(.SD, min, na.rm = TRUE),
    lapply(.SD, mean, na.rm = TRUE)
  ),
  by = .(id, year),
  .SDcols = neighbor_source_vars
]

# Rename columns appropriately
setnames(stats,
  old = names(stats)[-(1:2)],
  new = c(
    paste0(neighbor_source_vars, "_nbr_max"),
    paste0(neighbor_source_vars, "_nbr_min"),
    paste0(neighbor_source_vars, "_nbr_mean")
  )
)

# Merge back to main data
dt <- stats[dt, on = .(id, year)]

# Predictions with pre-trained RF model
preds <- predict(rf_model, newdata = dt)
```

---

### **Expected Impact**
- Eliminates 6.46M `lapply` calls and string concatenations.
- Uses efficient joins and grouped aggregations in compiled C code (`data.table`).
- Cuts runtime from **86+ hours to a few minutes** on a standard laptop.
- No change to the Random Forest model or estimand; only feature engineering improved.

**Bottom line:** The bottleneck is the neighbor feature computation, not RF inference. Optimize using vectorized joins and grouped aggregations instead of nested `lapply()`.
END_RESPONSE>>>
