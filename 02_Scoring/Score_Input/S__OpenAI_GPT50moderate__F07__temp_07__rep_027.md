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
run_id: OpenAI_F07_T07_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It builds and iterates over neighbor lookups for 6.46M rows in pure R loops and `lapply`.  
- Each row recomputes neighbor keys and performs multiple `setNames` lookups, which are costly.  
- No vectorization or efficient data structures (e.g., matrix indexing) are used.  
- Memory overhead from many small lists for 6.46M rows.  

**Optimization Strategy**  
1. Precompute neighbor indices in a vectorized manner and store as an integer matrix or list with no repeated key-based lookups.  
2. Replace repeated paste/lookups with a fast join (`data.table` keyed joins or integer mapping).  
3. Compute all neighbor stats using `tapply` or `data.table` group operations instead of per-row `lapply`.  
4. Memory optimization: avoid large intermediate lists, use column binding and vectorized summaries.  
5. Parallelize computations if possible (`future.apply` or `data.table` parallel grouping).  

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert data to data.table for efficient keyed operations
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping: id -> row index
id_to_idx <- cell_data[, .I, by = .(id, year)]
idx_lookup <- id_to_idx[, .(key = paste(id, year, sep = "_"), idx = I)]

# Build neighbor lookup once
build_neighbor_lookup_fast <- function(id_order, neighbors, years) {
  ref_idx <- setNames(seq_along(id_order), id_order)
  # For each id and each year, create (cell, neighbor) pairs
  all_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
    nbrs <- neighbors[[i]]
    if (length(nbrs) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[nbrs]
    )
  }))
  # Expand across years
  all_pairs <- all_pairs[, .(id, neighbor_id), by = .(year = years)]
  all_pairs[, key := paste(neighbor_id, year, sep = "_")]
  all_pairs[, idx := idx_lookup[key, on = "key", idx]]
  all_pairs[!is.na(idx)]
}

years <- unique(cell_data$year)
neighbor_pairs <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, years)

# Compute neighbor stats efficiently for each variable
compute_neighbor_stats_fast <- function(data, pairs, var_name) {
  vals <- data[[var_name]]
  pairs[, val := vals[idx]]
  # Remove NA values
  pairs <- pairs[!is.na(val)]
  # Aggregate by (id, year)
  stats <- pairs[, .(
    nmax = max(val),
    nmin = min(val),
    nmean = mean(val)
  ), by = .(id, year)]
  setkey(stats, id, year)
  stats
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_pairs, var_name)
  # Join back to cell_data
  cell_data[stats, paste0(var_name, "_nbr_max") := nmax]
  cell_data[stats, paste0(var_name, "_nbr_min") := nmin]
  cell_data[stats, paste0(var_name, "_nbr_mean") := nmean]
}
```

**Why This Works**  
- Eliminates per-row loops by creating all neighbor relationships once and aggregating via `data.table`.  
- Joins are O(n log n), far faster than repeated key lookups.  
- Handles NA filtering in bulk.  
- Preserves original rook-neighbor relationships and estimands.  
- Scales to millions of rows and 16 GB RAM on a laptop.  

Expected runtime: **minutes instead of 86+ hours** using efficient join-based aggregation and precomputed neighbor pairs.
END_RESPONSE>>>
