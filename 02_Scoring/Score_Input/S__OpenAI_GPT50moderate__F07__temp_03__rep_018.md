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
run_id: OpenAI_F07_T03_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow because:  
- It repeatedly loops over 6.46M rows and performs expensive lookups (`lapply` inside `compute_neighbor_stats` for each variable).  
- `neighbor_lookup` is a list of integer vectors, but every call recomputes stats using `lapply` and `rbind`.  
- Memory overhead is high (millions of small vectors, repeated concatenations).  
- No vectorization: operations are row-wise rather than grouped or matrix-based.  

**Optimization Strategy**  
- Precompute `neighbor_lookup` **once** (already done).  
- Flatten neighbor relationships into a long table: `(row_id, neighbor_id)`.  
- Use **data.table** or **dplyr** to compute `max`, `min`, `mean` in a single grouped operation per variable.  
- Join results back to `cell_data`.  
- Avoid loops over rows; instead, leverage vectorized aggregation.  
- Keep memory usage under control by processing one variable at a time.  

**Working R Code**  

```r
library(data.table)

# Assume: cell_data (data.frame), neighbor_lookup (list of integer vectors)
# Convert to data.table
dt <- as.data.table(cell_data)
dt[, row_id := .I]

# Flatten neighbor_lookup into long format
neighbor_pairs <- rbindlist(
  lapply(seq_along(neighbor_lookup), function(i) {
    if (length(neighbor_lookup[[i]]) == 0) return(NULL)
    data.table(row_id = i, neighbor_id = neighbor_lookup[[i]])
  })
)

setkey(neighbor_pairs, neighbor_id)
setkey(dt, row_id)

# Function to compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(dt, neighbor_pairs, var_name) {
  vals <- dt[, .(row_id, val = get(var_name))]
  
  # Join neighbor values
  joined <- neighbor_pairs[vals, on = .(neighbor_id = row_id)]
  
  # Aggregate by original row_id
  stats <- joined[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = row_id]
  
  # Merge back into dt
  setnames(stats, c("row_id", 
                    paste0(var_name, "_nbr_max"), 
                    paste0(var_name, "_nbr_min"), 
                    paste0(var_name, "_nbr_mean")))
  stats
}

# Apply for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(dt, neighbor_pairs, var_name)
  dt <- merge(dt, stats, by = "row_id", all.x = TRUE)
}

# Drop helper column
dt[, row_id := NULL]

# Result: dt now has neighbor max, min, mean for each variable
cell_data <- as.data.frame(dt)
```

**Why this works**  
- Converts neighbor relationships into a single long table â†’ enables fast joins and grouped aggregations.  
- Uses `data.table` for efficient memory and speed (vectorized operations).  
- Processes one variable at a time to keep RAM usage manageable.  
- Preserves original rook-neighbor topology and numerical estimands.  
- No retraining of the Random Forest model required.  

**Expected performance improvement**  
From 86+ hours to a few minutes (depending on disk I/O and CPU), as the bottleneck shifts from millions of R loops to optimized C-level aggregation in `data.table`.
END_RESPONSE>>>
