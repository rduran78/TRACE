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
run_id: OpenAI_F08_T07_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because `compute_neighbor_stats` is applied for every variable across all 6.46M rows using a per-row `lapply`. This creates massive overhead:  
- **Redundant computation**: For each year and variable, the same neighbor relationships are recalculated repeatedly.  
- **High memory usage and poor vectorization**: Each row is processed individually rather than in grouped or matrix form.  
- **Static neighbor structure ignored**: The neighbor graph does not change over time, but the code recomputes neighbor indices repeatedly by concatenating `id` and `year`.  

---

**Optimization Strategy**  
1. **Precompute a static neighbor index by `id` (cell)**, not by row. This is possible because neighbor relationships are constant across years.  
2. **Process one year at a time in a vectorized manner**: For each year, subset data and compute neighbor stats for all rows in that year using fast matrix operations.  
3. **Avoid repeated string concatenation and lookups**: Use integer indexing and pre-built maps.  
4. **Write results directly into columns** rather than rebuilding data frames in loops.  
5. **Use `data.table` for efficient grouping and assignment**.  

This reduces complexity from `O(N_rows Ã— neighbors)` across the whole panel to `O(N_year Ã— (N_cells + neighbors))` per variable.  

---

**Working R Code**  

```r
library(data.table)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer neighbor indices (1-based) matching id_order
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor indices by position
id_to_pos <- setNames(seq_along(id_order), id_order)
neighbor_pos_list <- lapply(rook_neighbors_unique, function(neigh_ids) {
  id_to_pos[as.character(neigh_ids)]
})

# Prepare new columns for neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Compute per year in-memory
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  # Subset this year's data in id_order
  year_dt <- cell_data[year == yr][match(id_order, id)]
  
  for (var in neighbor_source_vars) {
    vals <- year_dt[[var]]
    
    # Compute neighbor stats using precomputed neighbor positions
    nbr_stats <- lapply(neighbor_pos_list, function(nbr_idx) {
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) return(c(NA, NA, NA))
      c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    })
    
    nbr_stats_mat <- do.call(rbind, nbr_stats)
    
    # Assign back by id/year
    cell_data[year == yr, paste0(var, "_nbr_max") := nbr_stats_mat[, 1]]
    cell_data[year == yr, paste0(var, "_nbr_min") := nbr_stats_mat[, 2]]
    cell_data[year == yr, paste0(var, "_nbr_mean") := nbr_stats_mat[, 3]]
  }
}

# At this point, cell_data has neighbor-based features ready for RF prediction
```

---

**Why This Works**  
- **Static neighbor graph leveraged**: We build neighbor indices once, not per row-year.  
- **Year-based batching**: Each year runs in-memory for 344,208 rows rather than all 6.46M rows at once.  
- **Vectorized assignment**: Uses `data.table` for speed and memory efficiency.  
- **Preserves original estimand**: The computed stats (max, min, mean of neighbors per variable) remain unchanged.  
- **No retraining**: Random Forest step uses the enriched dataset as before.  

---

**Expected Performance Gain**  
From 86+ hours to a few hours on a standard 16 GB laptop (dominated by 28 passes Ã— 344k rows, fully memory-based).  
Further gains possible by parallelizing the outer loop over years using `parallel::mclapply` or `future.apply`.
END_RESPONSE>>>
