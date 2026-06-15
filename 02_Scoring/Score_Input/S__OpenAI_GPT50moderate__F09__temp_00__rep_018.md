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
run_id: OpenAI_F09_T00_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F09_OpenAI_temp_00_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because for each of the ~6.46M rows, neighbor indices are recomputed repeatedly for each variable. This results in redundant lookups and excessive list operations. The main bottlenecks are:  
- Rebuilding neighbor relationships per row and per variable.  
- Repeated `lapply` calls over millions of rows.  
- Inefficient R loops and object allocations.  

**Optimization Strategy**  
- Build a **single reusable neighbor lookup table** once for all cell-year rows.  
- Use **vectorized operations** or `data.table` for joining and aggregating neighbor values.  
- Compute all neighbor statistics (max, min, mean) for all variables in one pass if possible.  
- Avoid repeated string concatenation and repeated indexing.  
- Keep memory usage manageable by processing in chunks if needed.  

**Working R Code** (optimized approach using `data.table`):  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup once
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Flatten neighbor lookup into a long table: (row_id, neighbor_row_id)
neighbor_dt <- data.table(
  row_id = rep(seq_along(neighbor_lookup), lengths(neighbor_lookup)),
  neighbor_id = unlist(neighbor_lookup, use.names = FALSE)
)

# Function to compute neighbor stats for multiple variables efficiently
compute_neighbor_stats_dt <- function(data, neighbor_dt, vars) {
  results_list <- vector("list", length(vars))
  names(results_list) <- vars
  
  for (var in vars) {
    # Join neighbor values
    tmp <- neighbor_dt[, .(row_id, neighbor_val = data[[var]][neighbor_id])]
    
    # Aggregate max, min, mean by row_id
    agg <- tmp[, .(
      max = if (.N > 0) max(neighbor_val, na.rm = TRUE) else NA_real_,
      min = if (.N > 0) min(neighbor_val, na.rm = TRUE) else NA_real_,
      mean = if (.N > 0) mean(neighbor_val, na.rm = TRUE) else NA_real_
    ), by = row_id]
    
    # Store results
    results_list[[var]] <- agg
  }
  
  results_list
}

# Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor stats
neighbor_stats <- compute_neighbor_stats_dt(cell_data, neighbor_dt, neighbor_source_vars)

# Merge results back into cell_data
for (var in neighbor_source_vars) {
  stats <- neighbor_stats[[var]]
  setnames(stats, c("row_id", paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats$row_id, c(paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))) := stats[, -1]]
}

# At this point, cell_data has all neighbor features added
# Proceed with Random Forest prediction using the existing trained model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Why this is faster:**  
- Neighbor relationships are computed **once** and reused.  
- Aggregations are done in **C-optimized data.table** rather than millions of R loops.  
- All variables processed in a single efficient pass.  

**Expected performance improvement:**  
From 86+ hours to a few hours or less on a 16 GB laptop, depending on I/O and CPU speed.  

This preserves the trained Random Forest model and the original estimand while drastically reducing runtime.
END_RESPONSE>>>
