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
run_id: OpenAI_F06_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F06_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` approach over ~6.46M rows, repeatedly subsetting vectors and creating intermediate objects. This results in excessive R-level loops and memory overhead. The neighbor lookup is static, but the current design recomputes stats row-by-row in pure R, which is extremely slow for this scale.  

**Optimization Strategy**  
- Precompute neighbor relationships as integer vectors and flatten them into a long format for vectorized aggregation.  
- Use `data.table` for efficient joins and grouped summaries.  
- Compute all neighbor stats (max, min, mean) in one pass per variable using fast aggregation instead of per-row loops.  
- Avoid recomputing for each variable by reusing the neighbor lookup structure.  
- Keep results aligned with original row order.  

**Working R Code**  
```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)

# Add row index for restoring order
cell_data[, row_id := .I]

# Flatten neighbor lookup into long format
# neighbor_lookup: list of integer vectors (neighbors for each row)
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Join neighbor values and compute stats for each variable
compute_neighbor_stats_fast <- function(dt, neighbor_dt, var_name) {
  vals <- dt[, .(row_id, value = get(var_name))]
  # Join neighbor values
  joined <- merge(neighbor_dt, vals, by.x = "neighbor_id", by.y = "row_id", allow.cartesian = TRUE)
  # Aggregate stats by original row_id
  agg <- joined[, .(
    max_val = max(value, na.rm = TRUE),
    min_val = min(value, na.rm = TRUE),
    mean_val = mean(value, na.rm = TRUE)
  ), by = row_id]
  # Handle rows with no neighbors or all NA
  agg[!is.finite(max_val), c("max_val", "min_val", "mean_val") := .(NA_real_, NA_real_, NA_real_)]
  setnames(agg, c("row_id", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  agg
}

# Compute for all variables and merge back
result_list <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_fast(cell_data, neighbor_dt, v))
result_dt <- Reduce(function(x, y) merge(x, y, by = "row_id"), result_list)

# Merge back to cell_data in original order
cell_data <- merge(cell_data, result_dt, by = "row_id", sort = FALSE)
cell_data[, row_id := NULL]

# cell_data now has neighbor-based features added
```

**Why This Works**  
- Eliminates per-row loops in favor of vectorized joins and grouped aggregation.  
- `data.table` handles 6.46M rows efficiently in memory on a 16 GB machine.  
- Preserves original estimand and Random Forest model compatibility.  
- Expected runtime drops from 86+ hours to minutes (depending on disk I/O).
END_RESPONSE>>>
