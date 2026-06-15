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
run_id: OpenAI_F08_T10_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is that `compute_neighbor_stats()` operates row-wise across ~6.46 million rows, repeatedly scanning neighbor indices for each cell-year observation. This recomputes the same neighbor structure for every year, despite neighbors being static. It results in extreme overhead in both memory and time (86+ hours estimate).

Key inefficiency:
- `neighbor_lookup` ties neighbors per cell-year row, causing large repeated lookups.
- Neighbor relationships are constant; only variable values change by year.
- Computation is not vectorized and invoked 5 times (one per variable).

**Optimization Strategy**  
1. Keep a static neighbor lookup **at cell level**, not cell-year level.
2. Split data by year and compute neighbor max/min/mean for each variable **once per year** by leveraging the static neighbor structure.
3. Use `vapply`/`data.table`/`matrix` operations for fast aggregation rather than looping over millions of rows.
4. Bind results back to the full dataset after computing per-year neighbor stats.
5. Memory-safe: process one year at a time (28 passes instead of 6.46M row-level passes).

---

### **Optimized R Code**

```r
library(data.table)

compute_neighbor_stats_by_year <- function(dt, neighbors, id_order, var_name) {
  # dt: data.table with columns id, year, <var_name>
  setkey(dt, id)  # fast join by id
  
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  
  # Preallocate result matrix
  n <- nrow(dt)
  result <- matrix(NA_real_, n, 3)
  
  # Process by year
  years <- unique(dt$year)
  for (yr in years) {
    # Subset for current year
    idx_year <- which(dt$year == yr)
    vals <- dt[[var_name]][idx_year]
    ids <- dt$id[idx_year]
    
    # Map each cell to stats from its neighbors for this year
    stats_year <- vapply(seq_along(ids), function(i) {
      cell_id <- ids[i]
      ref_idx <- id_to_idx[as.character(cell_id)]
      neigh_ids <- id_order[neighbors[[ref_idx]]]
      
      # Pull neighbor values for this year
      neigh_vals <- vals[ match(neigh_ids, ids, nomatch = 0L) ]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
      c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
    }, numeric(3))
    
    result[idx_year, ] <- t(stats_year)
  }
  
  result
}

compute_and_add_neighbor_features <- function(dt, neighbors, id_order, vars) {
  for (var_name in vars) {
    stats <- compute_neighbor_stats_by_year(dt, neighbors, id_order, var_name)
    dt[[paste0(var_name, "_nbr_max")]] <- stats[, 1]
    dt[[paste0(var_name, "_nbr_min")]] <- stats[, 2]
    dt[[paste0(var_name, "_nbr_mean")]] <- stats[, 3]
  }
  dt
}

# ---- Usage ----
# Convert to data.table for efficiency
cell_data <- as.data.table(cell_data)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_and_add_neighbor_features(cell_data, rook_neighbors_unique, id_order, neighbor_source_vars)
```

---

### **Expected Improvement**
- Complexity reduced from `O(nrow * avg_neighbors)` over 6.46M rows to `O(years * cells * avg_neighbors)` with batching.
- For 28 years Ã— 344K cells, this is orders faster and memory-friendly.
- Parallelization (`future.apply` or `parLapply`) per year can accelerate further.

This preserves:
- Original numerical estimand.
- The trained Random Forest pipeline (we only add neighbor features upstream).
- The static neighbor topology while recomputing variable-dependent features efficiently.
END_RESPONSE>>>
