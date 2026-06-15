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
run_id: OpenAI_F04_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is `build_neighbor_lookup`: it uses an expensive `lapply` with repeated character concatenation and lookups for all 6.46M rows across 28 years. This is highly inefficient (86+ hours). Each iteration rebuilds neighbor keys and does multiple hash lookups. The `compute_neighbor_stats` step is vectorized, but memory allocation for lists and repeated filtering still incurs overhead. Random Forest inference is not the issue.

---

### **Optimization Strategy**
- **Precompute static structures**:
  - Neighbor IDs are spatially fixed (do not change by year) â†’ avoid recomputing for every row.
- **Use vectorized joins/data.table instead of nested lapply**:
  - Build a *long neighbor map* (row-year joined with neighbor-year).
- **Compute all neighbor statistics in a single grouped operation using data.table** rather than calling `compute_neighbor_stats` repeatedly.
- **Memory efficiency:** Avoid expanding full 6.46M Ã— #neighbors matrix in memory. Compute only needed aggregates.

---

### **Optimized Implementation (data.table)**
```r
library(data.table)

# Convert input to data.table
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setkey(cell_data, id, year)

# Precompute neighbor pairs once (static graph)
# rook_neighbors_unique: list of integer vectors, same length as id_order
pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

setkey(pairs, neighbor_id)  # to join by neighbor_id later

# Ensure key on cell_data for fast join
setkey(cell_data, id)

# Join neighbors for all years using Cartesian join on year
expand_pairs <- cell_data[, .(id, year)][pairs, on = .(id = neighbor_id), allow.cartesian = TRUE]
# rename for clarity: expand_pairs$id is the neighbor cell, join back to source as `cell_id`
setnames(expand_pairs, "id", "neighbor_id")
# expand_pairs now: neighbor_id, cell_id(from pairs originally), year

# Merge neighbor values
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  vals <- cell_data[, .(neighbor_id = id, year, v = get(var_name))]
  expand_pairs <- merge(expand_pairs, vals, by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Compute stats by group (cell_id, year)
  stats <- expand_pairs[, .(
    max_v = max(v, na.rm = TRUE),
    min_v = min(v, na.rm = TRUE),
    mean_v = mean(v, na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Handle cases where all neighbors are NA: replace with NA
  stats[!is.finite(max_v), `:=`(max_v = NA, min_v = NA, mean_v = NA)]
  
  # Add back to cell_data
  setkey(stats, cell_id, year)
  setkey(cell_data, id, year)
  cell_data[stats, paste0(var_name, "_nbr_max") := max_v]
  cell_data[stats, paste0(var_name, "_nbr_min") := min_v]
  cell_data[stats, paste0(var_name, "_nbr_mean") := mean_v]
  
  # Drop 'v' column for next iteration
  expand_pairs[, v := NULL]
}

```

---

### **Why is this fast?**
- Uses `data.table` joins and group aggregation (highly optimized in C).
- The neighbor graph (~1.37M edges) is expanded only once for the (cell, year) combinations.
- Loops only over 5 variables, no deeply nested `lapply`.
- Avoids repeated string concatenations and hash lookups.
- Runs in **minutes to a couple of hours** on 16 GB RAM instead of 86+ hours.

**Preserves:**
- Trained Random Forest model remains untouched.
- Neighbor feature definitions (max, min, mean) unchanged.

**Expected runtime improvement:** 86h â†’ ~1â€“2h (depending on I/O).  
Memory overhead: manageable because operations are streamlined and aggregated immediately.  
This directly addresses the bottleneck: inefficient spatial neighbor feature construction.
END_RESPONSE>>>
