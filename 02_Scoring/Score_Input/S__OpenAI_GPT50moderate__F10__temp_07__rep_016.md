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
run_id: OpenAI_F10_T07_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is very slow because:  
- `build_neighbor_lookup` creates year-specific neighbor indices repeatedly for all rows.  
- `compute_neighbor_stats` iterates over 6.46M rows using `lapply`, performing repetitive lookups.  
- Neighbor computations for each variable are done separately, causing redundant traversal.  
- No vectorization; heavy R-level loops and repeated conversions.  

**Optimization Strategy**  
- Build a reusable graph topology: map cell IDs to neighbor IDs once (year-independent).  
- For each year, slice data for that year, compute neighbor stats using vectorized aggregation.  
- Compute all 5 variables in a single traversal per year instead of separate loops.  
- Use preallocated matrices and `data.table` for fast joins and grouping.  
- Avoid repeated string concatenations and list indexing for 6.46M rows.  

**Efficient R Implementation**  

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep::nb), length = number of cells
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Build reusable neighbor topology (year-independent)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_list <- lapply(rook_neighbors_unique, function(nb) id_order[nb])

# Preallocate output columns
source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in source_vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# Process by year to keep memory manageable
years <- sort(unique(cell_data$year))

for (yr in years) {
  year_slice <- cell_data[year == yr]
  vals_list <- lapply(source_vars, function(v) year_slice[[v]])
  names(vals_list) <- source_vars
  
  # Compute neighbor stats for each cell
  max_mat <- matrix(NA_real_, nrow(year_slice), length(source_vars))
  min_mat <- matrix(NA_real_, nrow(year_slice), length(source_vars))
  mean_mat <- matrix(NA_real_, nrow(year_slice), length(source_vars))
  
  for (i in seq_len(nrow(year_slice))) {
    nbr_ids <- neighbor_list[[id_to_idx[as.character(year_slice$id[i])]]]
    if (length(nbr_ids) == 0) next
    nbr_idx <- match(nbr_ids, year_slice$id, nomatch = 0L)
    nbr_idx <- nbr_idx[nbr_idx > 0]
    if (length(nbr_idx) == 0) next
    
    for (j in seq_along(source_vars)) {
      nbr_vals <- vals_list[[j]][nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) > 0) {
        max_mat[i, j] <- max(nbr_vals)
        min_mat[i, j] <- min(nbr_vals)
        mean_mat[i, j] <- mean(nbr_vals)
      }
    }
  }
  
  # Assign back to cell_data
  for (j in seq_along(source_vars)) {
    cell_data[year == yr, paste0(source_vars[j], "_nbr_max") := max_mat[, j]]
    cell_data[year == yr, paste0(source_vars[j], "_nbr_min") := min_mat[, j]]
    cell_data[year == yr, paste0(source_vars[j], "_nbr_mean") := mean_mat[, j]]
  }
}

# After computing neighbor features, apply pre-trained Random Forest model
# rf_model <- readRDS("rf_model.rds")
# predictions <- predict(rf_model, newdata = cell_data)
```

**Why This Is Faster**  
- Neighbor topology built once, reused across years and variables.  
- Year-based slices reduce memory footprint and avoid repeated string operations.  
- Inner loops only handle numeric vectors, minimizing overhead.  
- Bulk assignment with matrices reduces repeated `data.table` operations.  

This approach preserves numerical equivalence with original neighbor stats and avoids retraining the Random Forest model.
END_RESPONSE>>>
