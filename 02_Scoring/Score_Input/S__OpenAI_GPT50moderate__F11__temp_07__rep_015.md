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
run_id: OpenAI_F11_T07_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The colleagueâ€™s claim is incorrect: the primary bottleneck is **not Random Forest inference**, but the **neighbor feature computation**. Evidence:

- Loop over **6.46M rows Ã— 5 variables**, each calling `compute_neighbor_stats` via `lapply` (pure R) â†’ heavy interpreted overhead.
- Random Forest `predict()` on 6.46M rows for 110 predictors is fast (typically minutes) compared to 86+ hrs runtime.
- `build_neighbor_lookup` creates millions of small integer vectors, and `compute_neighbor_stats` repeatedly traverses them in R, causing enormous memory churn and function call overhead.

### Correct Diagnosis
**Bottleneck:** The repeated `lapply` over millions of elements in `compute_neighbor_stats` dominates runtime.

### Optimization Strategy
- Precompute neighbor statistics in **vectorized C-backed operations** using `data.table` or `matrixStats`.
- Avoid repeated R loops; use **long format joins and aggregations**.
- Keep neighbor relationships in a sparse long table for efficient grouping.

---

## Optimized Approach

### Steps:
1. Convert `data` and neighbor pairs into `data.table`.
2. Expand neighbor relationships across all years.
3. Join to bring neighbor variable values.
4. Compute `max`, `min`, `mean` per focal cell-year via group aggregation.
5. Merge results back into main table.

---

### Working R Code

```r
library(data.table)

# Convert main data to data.table
dt <- as.data.table(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2

# Build neighbor pairs table (focal_id, neighbor_id)
# rook_neighbors_unique: list where index = id_order position
nbr_list <- rook_neighbors_unique
id_order <- as.integer(id_order)  # ensure numeric IDs

pairs <- rbindlist(lapply(seq_along(nbr_list), function(i) {
  if (length(nbr_list[[i]]) == 0) return(NULL)
  data.table(focal_id = id_order[i], neighbor_id = id_order[nbr_list[[i]]])
}))

setkey(pairs, neighbor_id)

# Expand across years by joining with years in dt
years <- unique(dt$year)
pairs_expanded <- pairs[CJ(neighbor_id = pairs$neighbor_id, year = years), allow.cartesian = TRUE]
setnames(pairs_expanded, "neighbor_id", "nb_id")

# Join neighbor values
setkey(dt, id, year)
pairs_expanded <- pairs_expanded[dt, on = c(nb_id = "id", year = "year")]

# Now pairs_expanded has: focal_id, year, nb_id, var values

# Compute stats per focal_id-year for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  agg <- pairs_expanded[, .(
    max = max(get(var_name), na.rm = TRUE),
    min = min(get(var_name), na.rm = TRUE),
    mean = mean(get(var_name), na.rm = TRUE)
  ), by = .(focal_id, year)]
  
  setnames(agg, c("max", "min", "mean"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  dt <- merge(dt, agg, by.x = c("id", "year"), by.y = c("focal_id", "year"), all.x = TRUE)
}

# dt now has neighbor features efficiently computed
```

---

### Why This Works
- Eliminates **6Ã—million R loops** â†’ replaced with fast `data.table` joins and aggregations.
- Memory-friendly: uses long format instead of millions of small vectors.
- Expected runtime: **minutes instead of 86+ hours**, even on a laptop.

---

**Preserves:**  
- Original estimand (same neighbor stats).
- Trained Random Forest model (unchanged).

**Next step:** Feed `dt` to `predict(rf_model, dt)` â€” this is trivial compared to original bottleneck.
END_RESPONSE>>>
