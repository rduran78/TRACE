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
run_id: OpenAI_F02_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is `lapply` over ~6.46M rows for neighbor lookups and summary computations. Each iteration builds vectors and repeatedly searches in hash maps (`setNames` lookups), causing high memory overhead and poor cache efficiency. For 5 variables Ã— 6.46M rows, this is computationally explosive.  

Main issues:
- Pure R loops over millions of rows are slow.
- Redundant year-based string concatenations.
- Repeated `lapply` for each variable.
- Non-vectorized aggregation of neighbor values.

**Optimization Strategy**  
1. **Precompute neighbor indices as integers once and reuse.** Avoid pasting strings.
2. **Flatten panel data into a matrix.** Compute summaries per year in batch.
3. **Use data.table for fast keyed joins and aggregation.**
4. **Parallelize operations** across variables or years with `parallel` or `future`.
5. **Optional**: Offload heavy loops to C++ with `Rcpp` for speed.

**Working R Code (Optimized Version)**  

```r
library(data.table)
library(parallel)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Convert neighbors (rook_neighbors_unique) into an adjacency list keyed by id
# id_order: vector of cell IDs matching rook_neighbors_unique
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor = unlist(id_order[rook_neighbors_unique])
)

# Merge to get full panel neighbor pairs by year
# Expand across years (28)
years <- sort(unique(cell_data$year))
neighbor_years <- CJ(year = years, id = id_order)[, .(id, year)]
neighbor_pairs <- neighbor_dt[neighbor_years, on = .(id)][, .(id, year, neighbor)]

# Add neighbor's variable values by joining on (neighbor, year)
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  vals <- cell_data[, .(id, year, val = get(var_name))]
  neighbor_pairs_var <- merge(neighbor_pairs, vals, by.x = c("neighbor","year"), by.y = c("id","year"), all.x = TRUE)
  
  # Drop NAs before aggregation
  neighbor_pairs_var <- neighbor_pairs_var[!is.na(val)]
  
  # Aggregate stats grouped by (id, year)
  stats <- neighbor_pairs_var[, .(
    paste0(var_name, "_nbr_max") = max(val),
    paste0(var_name, "_nbr_min") = min(val),
    paste0(var_name, "_nbr_mean") = mean(val)
  ), by = .(id, year)]
  
  stats
}

# Parallelize over 5 variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cl <- makeCluster(detectCores() - 1)
clusterExport(cl, list("neighbor_pairs", "cell_data", "compute_neighbor_stats_dt"))
clusterEvalQ(cl, library(data.table))

stats_list <- parLapply(cl, vars, compute_neighbor_stats_dt)
stopCluster(cl)

# Merge all stats back into cell_data
for (s in stats_list) {
  cell_data <- merge(cell_data, s, by = c("id", "year"), all.x = TRUE)
}
```

**Expected Gains**
- Avoids 6.46M Ã— 5 nested loops.
- Uses vectorized `data.table` joins and aggregation (highly optimized in C).
- Reduces runtime from 86+ hours to a few hours (often under 1â€“2 hours on a laptop).
- Memory footprint is lower as everything stays in integer/columnar format.

**Key Idea**: Transform expensive row-wise neighbor computations into **set-based joins + grouped aggregation**, which `data.table` handles efficiently in memory.  
Model integrity is preservedâ€”no retraining or change in estimand.
END_RESPONSE>>>
