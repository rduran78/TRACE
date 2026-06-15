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
run_id: OpenAI_F11_T10_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F11_OpenAI_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The primary bottleneck is **not Random Forest inference**, but the **neighbor feature computation implemented with deeply nested R loops (`lapply` over ~6.46M rows for each of 5 variables)**. Each `compute_neighbor_stats` call traverses a large neighbor list and performs vector operations repeatedly, resulting in *O(N Ã— neighbors Ã— vars)* complexity and massive R-level overhead.

### âœ… Diagnosis
- Neighbor feature construction dominates runtime:
  - 6.46M rows Ã— 5 variables Ã— repeated list indexing and `lapply` calls = *tens of hours* in pure R loops.
- Random Forest inference on 6.46M rows is fast in comparison (typically minutes to <1 hour with `ranger`).
- Therefore, the colleague's claim is incorrect.

---

### âœ… Correct Optimization Strategy
- **Vectorize and/or use data.table to compute neighbor stats in bulk** rather than row-by-row loops.
- Pre-expand neighbor relationships into a long table, join values, and compute summaries grouped by `id-year`.
- Leverage `data.table` for efficient grouped aggregations.
- Reuse `neighbor_lookup` in long form instead of creating 6.46M lists repeatedly.

---

### âœ… Optimized Working R Code

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Build neighbor long table once (id-year -> neighbor id-year)
# 'cell_data' has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
neighbors_long <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
  ref_id <- id_order[ref_idx]
  neigh_ids <- rook_neighbors_unique[[ref_idx]]
  if (length(neigh_ids) == 0) return(NULL)
  data.table(
    id = ref_id,
    neighbor_id = id_order[neigh_ids]
  )
}))

# Expand across years
years <- unique(cell_data$year)
neighbors_long <- neighbors_long[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbors_long, "year", "year")  # keep year column
neighbors_long <- neighbors_long[, .(id, year, neighbor_id)]

# Join neighbor values for all variables in one pass
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  # Join to bring neighbor variable values
  tmp <- merge(
    neighbors_long,
    cell_data[, .(neighbor_id = id, year, val = get(var_name))],
    by = c("neighbor_id", "year"),
    all.x = TRUE
  )
  
  # Remove NAs
  tmp <- tmp[!is.na(val)]
  
  # Compute stats grouped by (id, year)
  stats <- tmp[, .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(id, year)]
  
  # Merge back into main table
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# At this point, cell_data contains all neighbor features efficiently.
```

---

### âœ… Why This is Faster
- Eliminates deep nested `lapply` calls on millions of rows.
- Groups and summarizes at C-level with `data.table` (highly optimized).
- One full neighbor expansion and batch processing for all years.

---

**Expected improvement:** From 86+ hours â†’ **under 1 hour** for feature engineering on a 16GB laptop, as the computation is dominated by efficient joins and aggregations rather than R-level loops. Random Forest inference remains as is.

**Conclusion:** *Neighbor feature computation is the primary bottleneck; optimize it with vectorized, data.table-based aggregation.*
END_RESPONSE>>>
