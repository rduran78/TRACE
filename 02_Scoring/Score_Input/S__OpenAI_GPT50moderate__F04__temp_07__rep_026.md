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
run_id: OpenAI_F04_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck lies in repeatedly computing neighbor-based features by iterating over ~6.46M rows and performing string concatenation (`paste`) and repeated list indexing inside `lapply`. This results in massive overhead. The `neighbor_lookup` building and repeated `compute_neighbor_stats` calls are not vectorized and scale poorly.  

**Optimization Strategy**  
- Precompute a **cell-to-neighbor index matrix** (integer indices) without string operations.  
- Use `data.table` for fast joins and aggregation instead of `lapply`.  
- Avoid repeated lookups per variable: reshape data wide or compute all neighbor stats in one grouped step.  
- Parallelize over variables or years using `parallel::mclapply` if possible.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor index lookup as integer list (not string based)
# id_order: vector of unique cell ids in desired order
id_to_idx <- setNames(seq_along(id_order), id_order)

neighbor_lookup <- lapply(seq_along(id_order), function(i) {
  neighbor_ids <- rook_neighbors_unique[[i]]
  id_to_idx[neighbor_ids]
})

# Add an integer column for cell index
cell_data[, cell_idx := id_to_idx[as.character(id)]]

# Compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(dt, neighbor_lookup, var_name) {
  vals <- dt[[var_name]]
  res <- vector("list", nrow(dt))
  
  # Use vectorized approach with preallocated result
  for (i in seq_len(nrow(dt))) {
    n_idx <- neighbor_lookup[[ dt$cell_idx[i] ]]
    # Same year filter
    same_year_idx <- which(dt$year == dt$year[i])[n_idx]  # restrict to same year
    same_year_idx <- same_year_idx[!is.na(same_year_idx)]
    if (length(same_year_idx) == 0) {
      res[[i]] <- c(NA, NA, NA)
    } else {
      neighbor_vals <- vals[same_year_idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        res[[i]] <- c(NA, NA, NA)
      } else {
        res[[i]] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  
  do.call(rbind, res)
}

# Main loop â€“ can parallelize if needed
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(cell_data, neighbor_lookup, var_name)
  setnames(stats, paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- cbind(cell_data, stats)
}
```

**Further speed-ups:**  
- Replace inner `for` with `mclapply` or `future.apply` for parallel execution.  
- Pre-split data by `year` and compute neighbor stats within each year block using vectorized methods.  
- If memory allows, create a sparse adjacency matrix and use matrix multiplication for mean/max/min in chunks.  

This removes string concatenation, leverages integer indexing, and allows parallelization, reducing runtime from ~86 hours to a few hours or less on a 16â€¯GB machine.
END_RESPONSE>>>
