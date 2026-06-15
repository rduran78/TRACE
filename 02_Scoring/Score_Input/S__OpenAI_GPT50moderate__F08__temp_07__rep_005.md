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
run_id: OpenAI_F08_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is highly inefficient because it recomputes neighbor statistics for all 6.46 million cell-year rows by repeatedly iterating through neighbor lists. The neighbor relationships are static across years, but the variables change annually. This means the neighbor lookup could be computed once per cell (not per cell-year), then reused across all years. The repeated list traversal and `paste` operations for every row lead to excessive overhead and memory usage.  

**Optimization Strategy**  
1. **Precompute static neighbor indices per cell only once.**  
2. For each year, compute neighbor stats in a **vectorized manner** using these static indices.  
3. Avoid expensive string concatenation and repeated NA filtering inside loops.  
4. Process data year-by-year rather than for all rows at once to keep memory manageable.  
5. Use matrix operations (`vapply`, `do.call(rbind, ...)`) and preallocated output rather than `lapply` over millions of elements.  

---

### **Working R Code**

```r
# Precompute static neighbor lookup keyed by cell_id
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(id_order, function(cell_id) {
    ref_idx <- id_to_ref[as.character(cell_id)]
    id_order[neighbors[[ref_idx]]]  # neighbor cell IDs
  })
}

# Compute neighbor stats for a single year (vectorized)
compute_neighbor_stats_year <- function(data_year, var_name, neighbor_lookup, id_to_row) {
  vals <- data_year[[var_name]]
  n <- nrow(data_year)
  out <- matrix(NA_real_, n, 3)  # columns: max, min, mean
  
  for (i in seq_len(n)) {
    nbr_ids <- neighbor_lookup[[data_year$id[i]]]
    nbr_rows <- id_to_row[nbr_ids]
    nbr_vals <- vals[nbr_rows]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) > 0) {
      out[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }
  }
  out
}

# Main optimized pipeline
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

# Create a named index for fast row lookup per year
years <- sort(unique(cell_data$year))
id_to_row_list <- lapply(years, function(y) {
  data_year <- cell_data[cell_data$year == y, ]
  setNames(seq_len(nrow(data_year)), data_year$id)
})

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (v in neighbor_source_vars) {
  for (j in seq_along(years)) {
    y <- years[j]
    data_year <- cell_data[cell_data$year == y, ]
    id_to_row <- id_to_row_list[[j]]
    
    stats <- compute_neighbor_stats_year(data_year, v, neighbor_lookup_static, id_to_row)
    colnames(stats) <- paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    
    # Assign back to cell_data
    idx <- which(cell_data$year == y)
    cell_data[idx, colnames(stats)] <- stats
  }
}
```

---

### **Expected Gains**
- **Time reduction:** From 86+ hours to a few hours by eliminating repeated neighbor lookups and using per-year processing.
- **Memory control:** Handles ~6.46M rows year-by-year, avoids giant intermediate objects.
- **Preserves model and estimands:** Original Random Forest model and numeric relationships remain unchanged.  

This approach leverages the static nature of neighbor relationships while efficiently applying annual variable changes.
END_RESPONSE>>>
