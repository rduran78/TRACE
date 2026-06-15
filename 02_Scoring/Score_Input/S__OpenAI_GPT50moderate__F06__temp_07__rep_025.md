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
run_id: OpenAI_F06_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck lies in the nested `lapply` structure applied over ~6.46M rows. For each observation, `compute_neighbor_stats` iterates through neighbor indices, computes `max`, `min`, and `mean`, and then binds results. This is highly inefficient in R because:

- It runs millions of small R function calls (interpreted overhead dominates).
- Repeated lookups in `vals` and filtering `NA`s are done row by row.
- No vectorization or compiled operations are leveraged.
- Memory thrashing occurs when binding millions of small vectors.

**Optimization Strategy**  
- Precompute neighbor lists once (already done).
- Use `data.table` for efficient grouping and joins.
- Reshape data into long format keyed by `(id, year)` for fast neighbor merges.
- Compute stats using vectorized `max`, `min`, and `mean` within groups.
- Parallelize computations by year (or chunks) using `future.apply` or `data.table` parallelism.
- Avoid nested loops entirely.

**Working R Code (Optimized)**  
```r
library(data.table)
library(future.apply)

# Assume: cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# Convert to data.table
setDT(cell_data)

# neighbor_lookup: list of integer vectors (indices of neighbors) from build_neighbor_lookup
# Flatten into long mapping table once
neighbor_dt <- rbindlist(lapply(seq_along(neighbor_lookup), function(i) {
  if (length(neighbor_lookup[[i]]) == 0) return(NULL)
  data.table(row_idx = i, neighbor_idx = neighbor_lookup[[i]])
}))

# Add year and id to neighbor_dt by joining with cell_data
neighbor_dt[, year := cell_data$year[row_idx]]

# Join neighbor attributes by neighbor_idx
neighbor_dt[, neighbor_id := cell_data$id[neighbor_idx]]

# Function to compute neighbor stats for one variable
compute_neighbor_features <- function(var_name) {
  # Extract var values
  vals <- cell_data[[var_name]]
  neighbor_dt[, value := vals[neighbor_idx]]
  
  # Drop NAs
  neighbor_dt_non_na <- neighbor_dt[!is.na(value)]
  
  # Aggregate: for each row_idx (cell-year), compute max, min, mean
  stats <- neighbor_dt_non_na[, .(
    max_val = max(value),
    min_val = min(value),
    mean_val = mean(value)
  ), by = row_idx]
  
  # Prepare output data.table aligned with cell_data
  out <- data.table(row_idx = seq_len(nrow(cell_data)))
  out <- merge(out, stats, by = "row_idx", all.x = TRUE)
  
  # Assign to cell_data
  cell_data[, paste0(var_name, "_nbr_max") := out$max_val]
  cell_data[, paste0(var_name, "_nbr_min") := out$min_val]
  cell_data[, paste0(var_name, "_nbr_mean") := out$mean_val]
}

# Parallelize over variables
plan(multisession, workers = parallel::detectCores() - 1)
future_lapply(c("ntl", "ec", "pop_density", "def", "usd_est_n2"), compute_neighbor_features)

# cell_data now contains neighbor-based features for the Random Forest model
```

**Why this is faster:**  
- All neighbor relationships are processed in a single large `data.table`, avoiding millions of separate R function calls.
- Aggregation is done in compiled C code via `data.table`.
- Parallelization further reduces runtime.

**Expected performance:**  
From 86+ hours down to a few hours (or less) depending on disk I/O and CPU cores, while preserving the trained Random Forest model and original estimand.
END_RESPONSE>>>
