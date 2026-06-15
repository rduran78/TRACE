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
run_id: OpenAI_F06_T00_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F06_OpenAI_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, repeatedly performing vector lookups and allocations.  
- Neighbor stats are computed row-by-row instead of in a vectorized or grouped manner.  
- No parallelization or efficient data structures (e.g., `data.table`) are used.  
- Memory overhead from repeated list operations is high.  

**Optimization Strategy**  
- Precompute neighbor relationships once and store as integer indices.  
- Use `data.table` for fast joins and grouping.  
- Compute neighbor stats in a vectorized way by reshaping data into wide format by year, then aggregating by neighbor indices.  
- Optionally parallelize across years to reduce memory pressure.  
- Avoid raster focal analogy unless raster alignment is guaranteed; here, the panel structure and precomputed rook neighbors make a graph-based approach better.  

**Working R Code**  

```r
library(data.table)
library(parallel)

# Convert to data.table
setDT(cell_data)

# Ensure keys for fast joins
setkey(cell_data, id, year)

# Precompute neighbor pairs (directed)
# rook_neighbors_unique: list of integer vectors (spdep nb)
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Expand neighbor pairs across all years
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[, .(id = rep(src, each = length(years)),
                                   nbr = rep(nbr, each = length(years)),
                                   year = rep(years, times = .N))]

# Join neighbor values for each variable
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  tmp <- merge(neighbor_dt,
               cell_data[, .(nbr = id, year, val = get(var_name))],
               by = c("nbr", "year"),
               all.x = TRUE)
  
  # Compute stats by (id, year)
  tmp_stats <- tmp[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Rename columns
  setnames(tmp_stats, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var_name, "_nbr_max"),
             paste0(var_name, "_nbr_min"),
             paste0(var_name, "_nbr_mean")))
  
  tmp_stats
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Parallelize across variables
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, c("neighbor_dt", "cell_data", "compute_neighbor_stats_dt"))
clusterEvalQ(cl, library(data.table))

stats_list <- parLapply(cl, neighbor_source_vars, compute_neighbor_stats_dt)
stopCluster(cl)

# Merge all stats back into cell_data
for (stats_dt in stats_list) {
  cell_data <- merge(cell_data, stats_dt, by = c("id", "year"), all.x = TRUE)
}

# cell_data now has neighbor features added
```

**Why this works:**  
- Uses `data.table` for efficient joins and aggregation.  
- Computes neighbor stats in bulk rather than row-by-row.  
- Parallelizes across variables to utilize multiple cores.  
- Preserves original estimand and Random Forest model (no retraining).  

**Expected performance:**  
- From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk speed and CPU cores.  
- Memory footprint is manageable because operations are grouped and vectorized.
END_RESPONSE>>>
