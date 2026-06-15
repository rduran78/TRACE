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
run_id: OpenAI_F09_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline repeatedly computes neighbor lookups for each cell-year combination, which is very inefficient given ~6.46M rows. The repeated per-row `lapply` combined with dynamic string pasting and index lookups dramatically increases runtime. The bottleneck is the nested loop across 6.46M rows Ã— 5 variables, performing redundant work for neighbors that do not change over time.  

**Optimization Strategy**  
- Precompute a reusable `neighbor_lookup` **once** for all cell IDs (static over time).
- For each year, join yearly attributes onto this static adjacency structure instead of recomputing combinations.
- Vectorize neighbor statistic computations by operating on matrices instead of row-by-row loops.
- Use integer indexing rather than string-pasting keys.
- Avoid repeatedly binding data frames; instead, preallocate output columns.
- Optionally use `data.table` for efficient grouping and joins.

**Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup (indices for each cell)
# Returns a list where element i = vector of neighbor cell IDs
neighbor_lookup_static <- rook_neighbors_unique

# For convenience, map cell_id to row positions for each year
# Build an index by (id, year)
cell_data[, row_idx := .I]

# Parameters
years <- sort(unique(cell_data$year))
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate columns for neighbor stats
for (var_name in neighbor_source_vars) {
  for (stat in c("max", "min", "mean")) {
    set(cell_data, j = paste0(var_name, "_nbr_", stat), value = NA_real_)
  }
}

# Compute neighbor stats year-by-year
for (yr in years) {
  cat("Processing year:", yr, "\n")
  
  year_data <- cell_data[year == yr]
  idx_map <- setNames(year_data$row_idx, year_data$id)
  
  for (var_name in neighbor_source_vars) {
    vals <- year_data[[var_name]]
    
    # Compute stats for each cell in this year
    stats_mat <- t(vapply(year_data$id, function(cid) {
      n_ids <- neighbor_lookup_static[[as.character(cid)]]
      n_idx <- idx_map[as.character(n_ids)]
      n_idx <- n_idx[!is.na(n_idx)]
      if (length(n_idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
      n_vals <- vals[match(n_idx, year_data$row_idx)]
      n_vals <- n_vals[!is.na(n_vals)]
      if (length(n_vals) == 0) return(c(NA_real_, NA_real_, NA_real_))
      c(max(n_vals), min(n_vals), mean(n_vals))
    }, numeric(3)))
    
    # Assign back
    cell_data[year == yr, paste0(var_name, "_nbr_max") := stats_mat[,1]]
    cell_data[year == yr, paste0(var_name, "_nbr_min") := stats_mat[,2]]
    cell_data[year == yr, paste0(var_name, "_nbr_mean") := stats_mat[,3]]
  }
}

# Now cell_data contains neighbor-based features
# Proceed with Random Forest prediction using the trained model
# Example: preds <- predict(trained_rf_model, newdata = cell_data)
```

**Key Improvements**
- Neighbor relationships are computed **once** (static list).
- Loop is only over years (28) and variables (5), not over full 6.46M rows repeatedly.
- Uses `data.table` for efficient subsetting and assignment.
- Memory footprint remains manageable on 16 GB RAM because we process one year at a time (~344K rows).
- Preserves original estimand and trained model.

This should reduce runtime from 86+ hours to a few hours or less on a standard laptop.
END_RESPONSE>>>
