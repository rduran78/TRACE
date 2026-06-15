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
run_id: OpenAI_F04_T10_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The computational bottleneck is in repeatedly traversing and looking up 6.46 million rows across 28 years in pure R lists through `lapply` for each row. This is highly inefficient due to:  
- Multiple vectorized lookups and `paste` string ops per row.  
- Inefficient repeated R-level loops (6.46M Ã— 5 variables).  
- Heavy object creation and coercion in `compute_neighbor_stats`.  

Random Forest inference is fast; neighbor feature generation dominates.

---

**Optimization Strategy**  
- Precompute a long-format edge list mapping each observation to its neighbor observations for all years (vectorized, data.table).  
- Join data by neighbor indices using efficient data.table joins instead of looping.  
- Aggregate neighbor stats (max, min, mean) in batch using grouped operations.  
- Avoid repeated string concatenation by generating numeric keys (`id`+`year code`).  
- Process all 5 neighbor variables in bulk using melt/dcast or lapply after join.  

This reduces 6.46M row Ã— row-loop complexity to O(n + e) grouping operations where *e â‰ˆ edges Ã— years*.

---

**Optimized R Code Using data.table**  
```r
library(data.table)

# Assume: cell_data (id, year, variables), id_order, rook_neighbors_unique
setDT(cell_data)
setkey(cell_data, id, year)

# 1. Numeric year encoding for fast joins
cell_data[, year_code := as.integer(factor(year))]

# 2. Build neighbor edge list once (id pairs, replicated for all years if needed)
# rook_neighbors_unique: list of neighbors by position in id_order
edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[[i]], nbr_id = id_order[rook_neighbors_unique[[i]]])
}))

# 3. Expand to all years (Cartesian join)
years <- unique(cell_data[, .(year, year_code)])
edges_all <- edge_list[rep(seq_len(.N), each = nrow(years))]
edges_all[, c("year", "year_code") := years[rep(seq_len(nrow(years)), .N / nrow(years))]]

# 4. Map to cell_data rows
setkey(edges_all, nbr_id, year)
edges_all <- edges_all[cell_data, on = .(nbr_id = id, year), nomatch = 0L]
# Now edges_all has: src_id, nbr_id, year, year_code, NBR variables from cell_data
# Rename for clarity
setnames(edges_all, old = c("id"), new = c("nbr_row_id"))

# 5. Compute neighbor stats for all vars in batch
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_list <- lapply(neighbor_source_vars, function(v) {
  edges_all[, .(
    max = max(get(v), na.rm = TRUE),
    min = min(get(v), na.rm = TRUE),
    mean = mean(get(v), na.rm = TRUE)
  ), by = .(src_id, year)]
})

# 6. Merge all stats back into cell_data
for (j in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[j]
  stats <- agg_list[[j]]
  setnames(stats, c("max", "min", "mean"), paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by.x = c("id", "year"), by.y = c("src_id", "year"), all.x = TRUE)
}

# Clean up
cell_data[, year_code := NULL]
```

---

**Why This Works**  
- Eliminates per-row `lapply` computation.
- Uses compiled C backend of data.table for joins and aggregations, highly optimized for 10M+ rows.
- Handles all years and variables in grouped vectorized operations.
- Memory footprint manageable: `edges_all` â‰ˆ (edges Ã— years) ~ 1.3M Ã— 28 â‰ˆ 38M, ~ a few GB in RAMâ€”OK on 16 GB laptop if processed sequentially.

---

**Outcome**  
This approach turns an 86+ hour job into likely **minutes to an hour**, without retraining the RF or changing the numerical estimand.
END_RESPONSE>>>
