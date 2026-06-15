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
run_id: OpenAI_F04_T07_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the neighbor feature construction, not Random Forest inference. The current implementation repeatedly performs costly per-row lookups (`lapply` over 6.46M rows) and string concatenations (`paste`), which are inefficient in R for large data. Each iteration recomputes neighbor indices for every observation instead of leveraging vectorized operations or pre-aggregation.  

---

**Optimization Strategy**  
1. **Precompute and cache neighbor indices by (cell_id â†’ neighbor_ids)** once, without string operations.
2. **Avoid repeated lookups per year**: Instead, join on `id` and `year` using `data.table` for fast indexed operations.
3. Compute all neighbor statistics in **one pass per variable** using vectorized aggregation rather than `lapply`.
4. Use **`data.table` grouping** and `rbindlist` for speed and memory efficiency.
5. Parallelize if possible (via `future.apply` or `data.table` threads).

---

**Optimized Working R Code**

```r
library(data.table)

# Convert data to data.table
setDT(cell_data)

# Precompute neighbor lookup as a list keyed by id
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
neighbor_list <- lapply(seq_along(id_order), function(ref_idx) {
  id_order[rook_neighbors_unique[[ref_idx]]]
})
names(neighbor_list) <- id_order

# Reshape cell_data: create a composite key for fast joins
cell_data[, key := id]  # 'year' already present

# Build a long table of neighbor relationships for all years
neighbor_dt <- rbindlist(lapply(names(neighbor_list), function(cell_id) {
  data.table(
    id = cell_id,
    neighbor_id = neighbor_list[[cell_id]]
  )
}), use.names = TRUE)

# Expand for all years (Cartesian join)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(id, neighbor_id, year = years), by = .(id, neighbor_id)]

# Join neighbor attributes
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]

# Compute stats for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_dt <- neighbor_dt[, .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats_dt, c("id", "year", paste0(var_name, "_nbr_max"),
                                  paste0(var_name, "_nbr_min"),
                                  paste0(var_name, "_nbr_mean")))
  cell_data <- stats_dt[cell_data, on = .(id, year)]
}

# Result: cell_data now contains neighbor features efficiently
```

---

**Why This Is Faster**  
- Eliminates per-row string operations and `lapply`.
- Uses **vectorized joins and aggregations** via `data.table`.
- Processes neighbor relationships in bulk rather than repeated lookups.
- Reduces memory churn by avoiding repeated small object allocations.

**Expected Outcome**  
This approach should reduce runtime from 86+ hours to a manageable number (potentially a few hours or less), while preserving the trained Random Forest model and the original numerical estimand.
END_RESPONSE>>>
