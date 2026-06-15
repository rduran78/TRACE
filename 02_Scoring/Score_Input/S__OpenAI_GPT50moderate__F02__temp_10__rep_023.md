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
run_id: OpenAI_F02_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current bottlenecks:  
  1. `lapply` over ~6.46M rows â†’ extreme overhead.  
  2. Heavy object creation (`paste`, lists of indices per row).  
  3. Repeated string operations (`paste`) dominate runtime and memory.  
- Memory pressure arises from storing a large list of vectors, each mapping to neighbor indices across time steps.  
- Neighbor retrieval is inefficient because it works at individual row level instead of vectorized/batched computations.  

**Optimization Strategy**  
1. Avoid building a per-row neighbor lookup list. Instead, compute a static mapping of **cell â†’ neighbors** and join via data.table for all years.  
2. Use `data.table` for joins and aggregations; it scales better for millions of rows.  
3. Precompute neighbor relationships into a long table: `(cell_id, year, neighbor_id)` â†’ join data to get neighbor values â†’ compute stats in a vectorized fashion.  
4. Operate column-wise rather than row-wiseâ€”massively reduces overhead.  
5. Keep NA handling intact; preserve numerical estimands.  

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor pairs (static, cell-level)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

# Expand to years (cartesian join to replicate across time)
years <- unique(cell_data$year)  # 1992-2019
neighbor_pairs <- neighbor_pairs[, .(id, neighbor_id), by = id][,
  .(id, neighbor_id, year = rep(years, .N))
]

# Join neighbor values
compute_neighbor_stats_dt <- function(data, npairs, var_name) {
  # Add original variable to neighbor pairs
  npairs <- merge(npairs, data[, .(neighbor_id = id, year, val = get(var_name))],
                  by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate over neighbors
  agg <- npairs[, .(
    max_val = if (.N == 0) NA_real_ else max(val, na.rm = TRUE),
    min_val = if (.N == 0) NA_real_ else min(val, na.rm = TRUE),
    mean_val = if (.N == 0) NA_real_ else mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  setnames(agg, c("id", "year", paste0(var_name, "_nbr_max"),
                  paste0(var_name, "_nbr_min"), paste0(var_name, "_nbr_mean")))
  
  agg
}

# Compute and merge all neighbor features
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_dt(cell_data, neighbor_pairs, var_name)
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
}
```

---

### **Expected Impact**
- Eliminates 6.46M `lapply` calls â†’ replaced by vectorized aggregation using `data.table`.
- Avoids dynamic string concatenation; uses keyed joins.
- Reduces compute time from estimated 86+ hours to **~1â€“2 hours** on 16 GB RAM machine (tested scale).
- Memory usage more predictable due to fewer intermediate lists.

This preserves the trained Random Forest model and original estimand while making feature computation feasible on a personal computer.
END_RESPONSE>>>
