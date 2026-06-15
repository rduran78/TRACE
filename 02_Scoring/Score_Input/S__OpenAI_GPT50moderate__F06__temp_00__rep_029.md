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
run_id: OpenAI_F06_T00_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F06_OpenAI_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46 million rows, repeatedly performing lookups and vector operations in R (interpreted loops).  
- Neighbor statistics are computed row-by-row, causing massive overhead.  
- Memory access is inefficient due to repeated indexing and string concatenation.  

**Optimization Strategy**  
- Avoid per-row loops; use **vectorized operations** or **data.table** joins.  
- Precompute neighbor relationships as integer indices and flatten them into a long table for aggregation.  
- Use `data.table` for fast grouping and summarization.  
- Compute max, min, mean in one grouped operation, then join back to the main table.  
- Keep everything in RAM-efficient structures; avoid repeated string operations.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Flatten neighbor relationships into a long table
# neighbor_lookup: list of integer indices per row
neighbor_dt <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

# Add year and id for joining
neighbor_dt[, year := cell_data$year[row_id]]
neighbor_dt[, var_id := cell_data$id[row_id]]

# Join neighbor values for each source variable
compute_neighbor_stats_dt <- function(var_name) {
  vals <- cell_data[[var_name]]
  neighbor_dt[, value := vals[neighbor_id]]
  
  # Aggregate by row_id
  stats <- neighbor_dt[!is.na(value), .(
    max_val = max(value),
    min_val = min(value),
    mean_val = mean(value)
  ), by = row_id]
  
  # Merge back to cell_data
  cell_data[stats, on = .(row_id = .I),
            paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats, on = .(row_id = .I),
            paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats, on = .(row_id = .I),
            paste0(var_name, "_nbr_mean") := i.mean_val]
}

# Apply for all neighbor source variables
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_dt(var_name)
}
```

**Why This Works**  
- Converts the neighbor list into a long format once, avoiding repeated lookups.  
- Uses `data.table` grouping for fast aggregation (C-optimized).  
- Eliminates per-row loops and string concatenation.  
- Preserves original numerical estimand and Random Forest model.  

**Expected Performance**  
- From 86+ hours down to **minutes or a few hours** on a 16 GB laptop, depending on disk I/O and CPU.  
- Memory footprint manageable because operations are columnar and aggregated efficiently.
END_RESPONSE>>>
