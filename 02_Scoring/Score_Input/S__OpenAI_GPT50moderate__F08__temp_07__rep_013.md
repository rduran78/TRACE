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
run_id: OpenAI_F08_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation recomputes neighbor statistics for every row across all years, repeatedly iterating over a large neighbor structure (â‰ˆ6.46M rows). This leads to extreme overhead because:  
- Neighbor relationships are static across years, yet `build_neighbor_lookup` builds a year-specific lookup for all rows repeatedly.  
- `compute_neighbor_stats` operates row-wise with `lapply`, causing millions of small computations and excessive memory usage.  
- Inefficient repeated indexing and binding.  

**Optimization Strategy**  
- Precompute a static neighbor index keyed by cell ID only (not year).  
- Split computation by year: for each year, compute neighbor stats in bulk using matrix operations.  
- Use `data.table` or vectorized operations to avoid per-row loops.  
- Append results year-wise to `cell_data` without repeatedly recomputing lookup structures.  

**Working R Code**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute static neighbor list: cell_id -> neighbor cell_ids
neighbor_list <- lapply(seq_along(id_order), function(i) id_order[rook_neighbors_unique[[i]]])
names(neighbor_list) <- as.character(id_order)

# Function to compute neighbor stats for a single year
compute_year_stats <- function(dt_year, var_name) {
  vals <- dt_year[[var_name]]
  result <- lapply(neighbor_list, function(neigh_ids) {
    if (length(neigh_ids) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[match(neigh_ids, dt_year$id)]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  })
  do.call(rbind, result)
}

# Main loop: compute neighbor stats year-wise and bind
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare final container
all_results <- vector("list", length(neighbor_source_vars))

for (var_name in neighbor_source_vars) {
  res_list <- list()
  for (yr in unique(cell_data$year)) {
    dt_year <- cell_data[year == yr]
    stats_mat <- compute_year_stats(dt_year, var_name)
    tmp <- data.table(id = dt_year$id,
                      year = yr,
                      paste0(var_name, "_nb_max") := stats_mat[,1],
                      paste0(var_name, "_nb_min") := stats_mat[,2],
                      paste0(var_name, "_nb_mean") := stats_mat[,3])
    res_list[[as.character(yr)]] <- tmp
  }
  all_results[[var_name]] <- rbindlist(res_list)
}

# Merge all neighbor features back into cell_data by id-year
for (var_name in neighbor_source_vars) {
  cell_data <- merge(cell_data, all_results[[var_name]], by = c("id", "year"), all.x = TRUE)
}
```

**Why This Works**  
- Eliminates per-row neighbor lookup; uses static mapping.  
- Processes one year at a timeâ€”only 344K rows per iteration instead of 6.46M at once.  
- Vectorized operations reduce overhead dramatically (expected runtime: hours â†’ minutes on 16GB RAM laptop).  
- Preserves original estimands and pre-trained Random Forest input structure.  

This approach respects the static neighbor topology and dynamic year-specific variables, optimizing for time and memory without retraining the model.
END_RESPONSE>>>
