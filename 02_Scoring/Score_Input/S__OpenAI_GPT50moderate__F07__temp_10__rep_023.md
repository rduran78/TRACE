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
run_id: OpenAI_F07_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- Current method uses nested `lapply` over ~6.46M rows with multiple variables â†’ O(n Ã— k) lookup, large object serialization overhead, repeated string construction (`paste` calls), and list-of-lists indexing.  
- `neighbor_lookup` is computed once but indexing occurs repeatedly, costing extra time.  
- Memory stress comes from storing huge lists and repeated conversions to integer vectors.  
- Main bottlenecks:  
  1. String-based keys (`paste`) for every observation.  
  2. Multiple passes through neighbor lists for 5 variables.  
  3. Non-vectorized approach (looping through millions of rows in R).  

**Optimization Strategy**  
1. Eliminate expensive key matching: precompute integer mapping of `(id, year)` â†’ row index using vectorized join (e.g., `data.table` keyed operations).  
2. Represent neighbor relationships once as an integer matrix (mapping row â†’ neighbor rows) for all years via fast joins.  
3. Compute stats using matrix/vector operations rather than `lapply`.  
4. Use `data.table` groups to compute max/min/mean in a single grouped step.  
5. Avoid recalculating neighbor lookups for each variable.  

**Optimized Working R Code**  
```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Precompute all (id, year) combos
id_order_dt <- data.table(id = id_order)

# Expand rook neighbors to a long table
rook_pairs <- data.table(
  from_id = rep(id_order, lengths(rook_neighbors_unique)),
  to_id   = unlist(rook_neighbors_unique)
)

# Expand to panel years (Cartesian join)
years <- unique(dt$year)
rook_pairs <- rook_pairs[, .(to_id, year = years), by = from_id]

# Map to row indices in dt
rook_pairs[, from_idx := dt[rook_pairs, .I, on = .(id = from_id, year)]]
rook_pairs[, to_idx   := dt[rook_pairs, .I, on = .(id = to_id, year)]]
rook_pairs <- rook_pairs[!is.na(from_idx) & !is.na(to_idx)]

# For each variable, compute neighbor stats in a single grouped step
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in vars) {
  # Pull val for neighbors
  vals <- dt[[v]]
  rook_pairs[, val := vals[to_idx]]
  
  # Aggregate max, min, mean per from_idx
  stats <- rook_pairs[
    !is.na(val), .(
      max_val = max(val, na.rm = TRUE),
      min_val = min(val, na.rm = TRUE),
      mean_val = mean(val, na.rm = TRUE)
    ), by = from_idx
  ]
  
  # Join back to dt
  dt[stats$from_idx, paste0(v, "_nb_max") := stats$max_val]
  dt[stats$from_idx, paste0(v, "_nb_min") := stats$min_val]
  dt[stats$from_idx, paste0(v, "_nb_mean") := stats$mean_val]
}

cell_data <- as.data.frame(dt)
```

**Why Faster?**  
- Avoids repeated list traversal: single join operation builds neighbor relationships for all years.  
- Single large aggregation per variable using highly optimized `data.table`.  
- Eliminates repeated `paste` calls and complex lists â†’ integer indexing only.  
- Scales to millions of rows efficiently, likely dropping runtime from **86+ hours to a few hours** (or less on SSD-equipped laptop).  

**Preserves:**  
- Original rook topology.  
- Exact numerical estimands (max, min, mean).  
- Model compatibility (no retraining).
END_RESPONSE>>>
