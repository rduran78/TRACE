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
run_id: OpenAI_F08_T10_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current design computes neighbor stats in a highly redundant way:  
- `neighbor_lookup` is built at the cell-year level (â‰ˆ6.46M rows), although neighbor relationships are *static* across years (only depend on `id`).  
- For each row and each variable, it repeatedly looks up neighbor indices for that year, creating large lists.  
- This results in ~6.46M Ã— 5 variables Ã— 3 stats computations, leading to massive overhead in memory and time (86+ hours).  

**Optimization Strategy**  
1. **Exploit static structure**: Neighbor relationships depend only on cell `id`, so compute a single `id`-based mapping (not row-based).  
2. **Compute year-wise in blocks**: For each year, slice the data, compute neighbor stats using the same `id` neighbors into arrays, then rbind.  
3. **Vectorize**: Use `vapply`/matrix ops for efficiency.  
4. **Avoid repeated list processing** by precomputing neighbor id lists once and referencing them yearly.  

This reduces complexity from O(#rows Ã— neighbors) to O(#years Ã— cells Ã— neighbors), which is about *344k Ã— 28* operations (significant speed gain).  

---

### **Optimized Implementation**

```r
# Precompute static neighbor index lists by cell id
build_neighbor_id_lookup <- function(id_order, neighbors) {
  # neighbors: spdep nb object
  lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
}

compute_year_neighbor_stats <- function(year_data, id_order, neighbor_id_lookup, var_name) {
  vals <- setNames(year_data[[var_name]], year_data$id)  # vector indexed by id
  
  # Preallocate matrix: rows = cells in same order as id_order
  out <- matrix(NA_real_, nrow = length(id_order), ncol = 3)
  
  for (i in seq_along(id_order)) {
    n_ids <- neighbor_id_lookup[[i]]
    neighbor_vals <- vals[as.character(n_ids)]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) > 0) {
      out[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
  }
  
  colnames(out) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  as.data.frame(out)
}

# Main pipeline
id_order <- unique(cell_data$id)  # assumes sorted same as id_order input
neighbor_id_lookup <- build_neighbor_id_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Preallocate a list
result_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  year_data <- subset(cell_data, year == yr)
  year_data <- year_data[match(id_order, year_data$id), ]  # align with id_order
  
  stats_list <- lapply(neighbor_source_vars, function(var_name) {
    compute_year_neighbor_stats(year_data, id_order, neighbor_id_lookup, var_name)
  })
  
  neighbor_stats <- do.call(cbind, stats_list)
  result_list[[y]] <- cbind(year = yr, id = id_order, neighbor_stats)
}

# Combine back
neighbor_features <- do.call(rbind, result_list)

# Merge with original data efficiently (id and year keys)
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

**Why This Works**  
- Only 344k rows processed per year (instead of 6.46M list ops).
- Neighbor relationships computed once, reused for all years.
- Preserves target estimands since stats computed per cell-year using true neighbors.
- Memory footprint stays manageable (processing one year at a time).
- Compatible with the pre-trained Random Forestâ€”only feature engineering changes.

**Expected Gain:** From 86+ hours to a few hours on a 16 GB laptop.  

**Next Step:** Run your existing Random Forest predict phase with the enriched `cell_data`.
END_RESPONSE>>>
