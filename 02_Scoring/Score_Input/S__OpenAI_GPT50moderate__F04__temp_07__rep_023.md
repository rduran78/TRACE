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
run_id: OpenAI_F04_T07_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is `build_neighbor_lookup` and repeated `lapply` calls scanning millions of rows. Each time we compute neighbor stats, we iterate over 6.46M rows and perform expensive list indexing. This is highly inefficient because:
- `neighbor_lookup` itself is a large list of length 6.46M.
- For each variable, `compute_neighbor_stats` loops again over all rows.
- The workflow repeats 5 times for different variables â†’ ~32M iterations.
- Râ€™s list-based lookups and repeated `paste` string operations are slow at this scale.

**Optimization Strategy:**  
1. **Precompute a compact neighbor index table** instead of a giant list:
   - Convert the spatial neighbor relationships into a two-column `data.table`: `(row_id, neighbor_id)`, expanded across years.
   - Join this with the main data to get neighbor variable values using `data.table` joins (fast, vectorized).
2. **Compute all neighbor stats in one grouped aggregation** rather than looping over every row for every variable.
3. Use **`data.table` for memory efficiency and speed**.
4. Avoid repeated string concatenation and `lapply` by leveraging vectorized joins and `by` grouping.

**Optimized R Code:**

```r
library(data.table)

# Convert main data to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...

# Map cell id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Expand rook neighbors into a long table
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Cross with years to create full panel neighbor relationships
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[, .(src, nbr)][, .(year = years), by = .(src, nbr)]

# Map to row indices in cell_data
cell_data[, key := paste(id, year, sep = "_")]
neighbor_dt[, src_key := paste(src, year, sep = "_")]
neighbor_dt[, nbr_key := paste(nbr, year, sep = "_")]

# Create lookup for row indices
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$key)
neighbor_dt[, src_idx := idx_lookup[src_key]]
neighbor_dt[, nbr_idx := idx_lookup[nbr_key]]

# Drop keys
neighbor_dt <- neighbor_dt[!is.na(src_idx) & !is.na(nbr_idx), .(src_idx, nbr_idx)]

# Compute neighbor stats for all variables in one go
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in vars) {
  neighbor_vals <- data.table(src_idx = neighbor_dt$src_idx,
                               val = cell_data[[var]][neighbor_dt$nbr_idx])
  
  agg <- neighbor_vals[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = src_idx]
  
  setnames(agg, c("src_idx", paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")))
  cell_data[agg$src_idx, c(paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean")) := agg[, -1]]
}
```

**Why itâ€™s faster:**  
- `neighbor_dt` stores ~1.37M Ã— 28 â‰ˆ 38.4M rows, but operations are vectorized in `data.table`, avoiding millions of R loops.
- All neighbor computations per variable are done in one grouped aggregation rather than per-row `lapply`.
- Memory usage remains manageable because IDs and indices are integers, and only necessary columns are kept.

**Expected performance improvement:**  
From 86+ hours down to a few hours (often under 2 hours on a 16 GB laptop), since `data.table` joins and grouped aggregations are highly optimized in C.

**Preserves:**  
- Original Random Forest model (no retraining).
- Original numeric estimand.
- Exact neighbor definitions and values.

**Summary:** Replace per-row `lapply` loops with vectorized `data.table` joins and grouped aggregation for massive speedup.
END_RESPONSE>>>
