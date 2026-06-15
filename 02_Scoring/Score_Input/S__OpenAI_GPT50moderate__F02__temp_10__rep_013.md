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
run_id: OpenAI_F02_T10_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow and memory-intensive because:  
- `neighbor_lookup` constructs a full list of neighbor indices for every **cell-year row** (~6.46M rows), duplicating a lot of information and consuming large memory.  
- `compute_neighbor_stats` repeatedly allocates lists and runs `lapply` over millions of rows for every variable, causing billions of expensive R-level operations.  
- The entire process is row-oriented; vectorization or grouping is not utilized.  

**Optimization Strategy**  
1. **Reduce duplication by leveraging base cell neighbors only once**: Neighbors depend only on cell ID, not year. Avoid building cell-year neighbor lookups.  
2. **Use a join-based approach** rather than nested loops: For each variable and year, join each row with its neighbors via a neighbor table and then compute summary stats (max, min, mean) using `data.table` aggregation.  
3. **Memory efficiency**: Process one variable at a time, operate year-by-year or chunked using `data.table` grouping, eliminating huge lists.  
4. **Parallelization**: Optionally use `data.table` parallel ops or `future.apply` if CPU capacity allows.  

**Working Optimized R Code** (uses `data.table` for efficiency):  

```r
library(data.table)

# Assume: cell_data (columns: id, year, ntl, ec, pop_density, def, usd_est_n2)
# id_order: vector of unique cell ids in consistent order
# rook_neighbors_unique: spdep nb object

# Convert to data.table
DT <- as.data.table(cell_data)
setkey(DT, id, year)

# Build neighbor table once (base cell level)
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

setkey(neighbor_pairs, neighbor_id)

# Function to compute neighbor statistics efficiently
compute_neighbor_features_dt <- function(DT, var_name, neighbor_pairs) {
  DT_subset <- DT[, .(id, year, value = get(var_name))]
  
  # Join neighbors by year and neighbor_id
  neighbor_values <- neighbor_pairs[DT_subset, on = .(neighbor_id = id), allow.cartesian = TRUE]
  
  # neighbor_values: columns id (original neighbor cell), neighbor_id (source in join), year, value
  # Now aggregate by (id, year) of the original "id" in neighbor_pairs
  stats <- neighbor_values[, .(
    max_val = suppressWarnings(max(value, na.rm = TRUE)),
    min_val = suppressWarnings(min(value, na.rm = TRUE)),
    mean_val = if (.N > 0) mean(value, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to main DT
  setnames(stats, c("id", "year", paste0(var_name, "_max_nbr"), paste0(var_name, "_min_nbr"), paste0(var_name, "_mean_nbr")))
  DT <- stats[DT, on = .(id, year)]
  return(DT)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  DT <- compute_neighbor_features_dt(DT, var_name, neighbor_pairs)
}

# Convert back to data.frame if needed
cell_data <- as.data.frame(DT)
```

**Why this is faster:**  
- No `lapply` across millions of rows.  
- Only one neighbor pair table (~1.37M pairs), reused across all variables and years.  
- All operations vectorized in `data.table` with efficient joins and grouping.  
- Memory usage roughly proportional to rows in `neighbor_pairs` and single variable subset, not full expanded lookup per row.  

**Expected Performance Gain:**  
- From ~86 hours to roughly minutes or a few hours depending on disk I/O and CPU cores, well within 16 GB RAM on a standard laptop.  
- Further speedup possible by using `setDTthreads(parallel::detectCores())` before running.
END_RESPONSE>>>
