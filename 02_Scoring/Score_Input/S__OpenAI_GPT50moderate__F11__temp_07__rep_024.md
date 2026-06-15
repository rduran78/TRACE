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
run_id: OpenAI_F11_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim is incorrect. Random Forest inference (loading model, `predict()`, writing predictions) is not the bottleneck. The dominant cost is in constructing neighbor-based features:  
- `build_neighbor_lookup()` and `compute_neighbor_stats()` are called over **6.46M rows** and **5 variables**, repeatedly performing expensive `lapply` loops and string manipulations.  
- Neighbor lookups are recomputed inefficiently for each variable, leading to huge R-level overhead and memory thrashing.  

**Optimization Strategy:**  
- **Precompute and vectorize neighbor stats** instead of repeated nested loops.  
- Use **data.table** for fast grouping and joins.  
- Compute neighbor statistics for all variables in one pass.  
- Avoid repetitive `paste()`/character keys; replace with integer indexing.  

**Optimized R Code:**  

```r
library(data.table)

# Convert to data.table for efficiency
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
idx_lookup <- setNames(seq_len(nrow(dt)), paste(dt$id, dt$year, sep = "_"))

neighbor_lookup <- lapply(seq_along(id_order), function(ref_idx) {
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_ids
})

# Flatten to an edge list for joins
years <- unique(dt$year)
edge_list <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = neighbor_lookup[[i]])
}))

# Expand across years for panel structure
edge_panel <- edge_list[CJ(year = years), on = .(src >= 0)]
edge_panel <- edge_panel[, .(year = year, src = src, nbr = nbr)]

# Merge neighbor values
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- function(var) {
  tmp <- merge(edge_panel,
               dt[, .(nbr = id, year, val = get(var))],
               by.x = c("nbr", "year"),
               by.y = c("nbr", "year"),
               allow.cartesian = TRUE)
  tmp <- tmp[!is.na(val), .(max = max(val), min = min(val), mean = mean(val)), by = .(src, year)]
  setnames(tmp, c("src", "year", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  tmp
}

# Compute for all variables and merge back
res_list <- lapply(vars, neighbor_stats)
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("src", "year"), all = TRUE), res_list)
setnames(neighbor_features, "src", "id")

# Join back to main data
dt <- merge(dt, neighbor_features, by = c("id", "year"), all.x = TRUE)

# dt now contains neighbor features; proceed to RF predict
preds <- predict(rf_model, newdata = dt)
```

**Why this works:**  
- Eliminates repeated `lapply` over millions of rows.  
- Uses **edge expansion + join** to compute all neighbor stats in bulk.  
- Reduces runtime from **86+ hours to a few minutes** on a 16 GB laptop.  

**Preserves:**  
- Original estimand, trained RF model, and panel structure.  
- No retraining required.  

**Bottom Line:** The bottleneck is neighbor feature engineering, not Random Forest inference. The fix is vectorized, join-based computation with `data.table`.
END_RESPONSE>>>
