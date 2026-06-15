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
run_id: OpenAI_F04_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The computational bottleneck is not Random Forest inference but the repeated execution of `lapply` loops over ~6.46 million rows and dynamic string-based lookups in `build_neighbor_lookup` and `compute_neighbor_stats`. The current design performs expensive per-row operations and redundant work for each neighbor feature, causing massive overhead.

---

**Optimization Strategy**  
1. **Avoid repeated string concatenations and lookups**: Replace string-based keys (`paste(id, year)`) with direct integer indexing.
2. **Vectorize neighbor stats computation**: Precompute `neighbor_lookup` once as integer indices and avoid applying `lapply` multiple times for each variable.
3. **Use `data.table` for fast joins and grouping**: Reshape the data so neighbor relationships are expanded once, then aggregate with fast group operations.
4. **Parallelize computation**: Use `parallel` or `future.apply` for multi-core execution.
5. **Memory-aware batching**: Process variables in blocks if RAM is tight.
6. **Preserve model and estimand**: Do not retrain Random Forest; only change feature engineering performance.

---

**Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert to data.table
setDT(cell_data)

# Precompute mapping of (id, year) -> row index
cell_data[, row_idx := .I]

# Expand neighbor relationships across all years
# rook_neighbors_unique: list of neighbor IDs for each id in id_order
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[i], nbr_id = id_order[rook_neighbors_unique[[i]]])
}))

# Join neighbors for every year
years <- unique(cell_data$year)
expanded_neighbors <- CJ(year = years, src_id = id_order)[
  neighbor_dt, on = .(src_id), allow.cartesian = TRUE
]

# Map to row indices
expanded_neighbors[cell_data, on = .(src_id = id, year), src_idx := i.row_idx]
expanded_neighbors[cell_data, on = .(nbr_id = id, year), nbr_idx := i.row_idx]
expanded_neighbors <- expanded_neighbors[!is.na(src_idx) & !is.na(nbr_idx)]

# Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  vals <- cell_data[[var_name]]
  tmp <- expanded_neighbors[, .(src_idx, nbr_val = vals[nbr_idx])]
  tmp <- tmp[!is.na(nbr_val)]
  agg <- tmp[, .(
    paste0(var_name, "_nbr_max") = max(nbr_val),
    paste0(var_name, "_nbr_min") = min(nbr_val),
    paste0(var_name, "_nbr_mean") = mean(nbr_val)
  ), by = src_idx]
  agg
}

# Process all neighbor source variables in parallel
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("expanded_neighbors", "cell_data", "compute_neighbor_stats_dt"))
clusterEvalQ(cl, library(data.table))

agg_list <- parLapply(cl, neighbor_source_vars, compute_neighbor_stats_dt)
stopCluster(cl)

# Merge all results back
result_dt <- Reduce(function(x, y) merge(x, y, by = "src_idx", all = TRUE), agg_list)
cell_data <- merge(cell_data, result_dt, by.x = "row_idx", by.y = "src_idx", all.x = TRUE)

# Clean up helper column
cell_data[, row_idx := NULL]
```

---

**Expected Performance Gains**  
- Eliminates per-row `lapply` and string concatenation.
- Single pass neighbor expansion, vectorized aggregation via `data.table`.
- Parallel computation for multiple variables.
- Estimated runtime reduction: **86+ hours â†’ ~1â€“2 hours** on a modern laptop.

This preserves the trained model and original numerical estimand while optimizing neighbor feature computation to a practical level.
END_RESPONSE>>>
