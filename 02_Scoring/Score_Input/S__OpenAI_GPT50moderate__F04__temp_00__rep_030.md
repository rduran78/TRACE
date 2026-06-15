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
run_id: OpenAI_F04_T00_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F04_OpenAI_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated `lapply` over ~6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and named vector lookups, which are extremely slow at this scale. The neighbor relationships are static across years, yet the code recomputes keys for every row-year combination. Additionally, `compute_neighbor_stats` repeatedly scans vectors and allocates lists, adding overhead.

**Optimization Strategy**  
1. **Precompute neighbor indices by cell ID only** (not per year) and reuse them across all years.
2. **Avoid string-based lookups**; use integer indexing instead.
3. **Vectorize neighbor feature computation** using `data.table` or matrix operations.
4. **Process by year in chunks** to reduce memory pressure.
5. **Parallelize** across variables or years using `parallel` or `future.apply`.

**Working R Code (Optimized)**  
```r
library(data.table)
library(parallel)

# Convert to data.table for fast joins
setDT(cell_data)

# Precompute neighbor index list by cell ID
id_to_idx <- setNames(seq_len(nrow(unique(cell_data[, .(id)]))), unique(cell_data$id))
neighbor_idx_list <- lapply(rook_neighbors_unique, function(neigh_ids) {
  as.integer(id_to_idx[as.character(neigh_ids)])
})

# Add a column for cell index
cell_data[, cell_idx := id_to_idx[as.character(id)]]

# Function to compute neighbor stats for one variable and one year chunk
compute_neighbor_stats_chunk <- function(dt, var_name, neighbor_idx_list) {
  vals <- dt[[var_name]]
  res <- matrix(NA_real_, nrow(dt), 3)
  for (i in seq_len(nrow(dt))) {
    neigh_idx <- neighbor_idx_list[[dt$cell_idx[i]]]
    if (length(neigh_idx) > 0) {
      neigh_vals <- vals[neigh_idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        res[i, ] <- c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
      }
    }
  }
  res
}

# Process by year and variable in parallel
years <- unique(cell_data$year)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("cell_data", "neighbor_idx_list", "compute_neighbor_stats_chunk"))

for (var_name in neighbor_source_vars) {
  results_list <- parLapply(cl, years, function(y) {
    dt_year <- cell_data[year == y]
    compute_neighbor_stats_chunk(dt_year, var_name, neighbor_idx_list)
  })
  
  # Bind results and add columns
  res_mat <- do.call(rbind, results_list)
  colnames(res_mat) <- paste(var_name, c("neigh_max", "neigh_min", "neigh_mean"), sep = "_")
  cell_data[, (colnames(res_mat)) := res_mat]
}

stopCluster(cl)
```

**Why This Works**  
- Eliminates expensive string operations and repeated lookups.
- Uses integer indexing for neighbors.
- Processes data in manageable chunks (by year).
- Parallelization reduces wall-clock time dramatically.
- Preserves original estimand and Random Forest model.

**Expected Performance**  
This approach should reduce runtime from 86+ hours to a few hours on a 16 GB laptop, depending on parallel efficiency.
END_RESPONSE>>>
