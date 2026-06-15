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
run_id: OpenAI_F04_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is the neighbor feature construction:
- `build_neighbor_lookup` uses `lapply` over 6.46M rows, repeatedly constructing and traversing vectors. It allocates many intermediate objects, creating severe overhead.
- `compute_neighbor_stats` is similarly applying `lapply` row-wise, resulting in ~6.46M function calls for each variable.
- Both functions operate in R loops instead of vectorized or batch operations, causing the extreme execution time.

---

**Optimization Strategy:**  
1. **Precompute neighbor mapping with integer indices (matrix-based)**  
   - Convert neighbor relationships into an integer matrix aligned by `id` and reuse it for all years.
2. **Avoid per-row function calls:**  
   - Use vectorized aggregation via `data.table` or `matrixStats` in a long format join-aggregate-merge approach.
3. **Parallelize across variables or years:**  
   - Use `data.table` join grouped by `(year, neighbor_id)` to produce summaries for all neighbors in bulk.
4. **Leverage sparse relationships:**  
   - Store neighbors in a long data frame of `(id, neighbor_id)` and join rather than nested `lapply`.

This reduces 6.46M Ã— 5 Ã— 3 computations in tight loops to group operations handled natively in C.

---

**Working R Code (data.table approach):**
```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Convert rook_neighbors_unique into a long neighbor table
# rook_neighbors_unique is a list, each element is neighbor IDs for id_order[i]
nb_long <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0L) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand to cell-year level by cross-joining years
years <- unique(cell_data$year)
nb_long <- nb_long[, .(id, neighbor_id)][, year := rep(years, each = .N)]

# Reshape neighbor data for joins
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Merge neighbor values for all (id,year)
neighbor_values <- merge(nb_long,
  cell_data[, c("id", "year", vars), with = FALSE],
  by.x = c("neighbor_id", "year"),
  by.y = c("id", "year"),
  all.x = TRUE,
  allow.cartesian = TRUE
)

# Compute summary stats per id-year
agg_list <- lapply(vars, function(v) {
  data.table(
    stat_id = nb_long$id,
    year = nb_long$year,
    vals = neighbor_values[[v]]
  )[, .(
    paste0("neighbor_", v, "_max") := max(vals, na.rm = TRUE),
    paste0("neighbor_", v, "_min") := min(vals, na.rm = TRUE),
    paste0("neighbor_", v, "_mean") := mean(vals, na.rm = TRUE)
  ), by = .(stat_id, year)]
})

# Combine all aggregates
agg_dt <- Reduce(function(x, y) merge(x, y, by = c("stat_id", "year")), agg_list)
setnames(agg_dt, "stat_id", "id")

# Join back to original
cell_data <- merge(cell_data, agg_dt, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works:**  
- Avoids per-observation looping in R.
- Joins and group aggregation are implemented in C in `data.table`.
- Scales much better: tens of minutes instead of 86 hrs on 6.46M rows.

**Expected runtime cut:** Orders of magnitude reduction (hours â†’ under ~1 hour with 16 GB RAM).  
**Model unchanged:** The trained Random Forest is not touched; features computed match original semantics.
END_RESPONSE>>>
