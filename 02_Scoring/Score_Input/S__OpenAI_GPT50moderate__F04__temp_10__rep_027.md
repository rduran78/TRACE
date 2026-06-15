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
run_id: OpenAI_F04_T10_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is not Random Forest inference but neighbor feature construction. For ~6.46M rows and repeated `lapply` operations, the current implementation repeatedly loops through cells and years, performing expensive `paste` and index lookups in pure R. This scales poorly because:  
- High overhead for string operations (`paste(...)`) and repeated hash lookups on millions of entries.  
- No vectorization: each row triggers new index resolutions.  
- Memory churn due to many nested lists and repeated conversions.  

**Optimization Strategy**  
1. Replace per-row `lapply` with vectorized joins or indexed operations.  
2. Precompute all neighbor relationships for the entire panel using efficient data.table merges instead of string-based indexing.  
3. Compute neighbor stats by grouping rather than iterating, leveraging data.tableâ€™s fast aggregation.  
4. Avoid reconstructing key strings (`id_year`) repeatedly. Use integer IDs and keys.  
5. Keep everything in RAM-efficient form (data.table) to reduce memory pressure.  

**Working R Code (Optimized Approach)**  
```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute keys: numeric id mapping
cell_data[, id_int := as.integer(factor(id))]
cell_data[, year := as.integer(year)]

# Long edge table: neighbor relationships repeated across years
edges <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
  src <- id_order[ref_idx]
  nbrs <- rook_neighbors_unique[[ref_idx]]
  if (length(nbrs) == 0L) return(NULL)
  data.table(src = src, nbr = id_order[nbrs])
}))

# Expand for all years (vectorized)
years <- sort(unique(cell_data$year))
edges <- edges[, .(year = years, src, nbr), by = .(src, nbr)]

# Join source and neighbor data
src_data <- cell_data[, .(src_id = id, year, id_int, ntl, ec, pop_density, def, usd_est_n2)]
nbr_data <- cell_data[, .(nbr_id = id, year, id_int, ntl, ec, pop_density, def, usd_est_n2)]

edges <- merge(edges, src_data, by.x = c("src", "year"), by.y = c("src_id", "year"), all.x = TRUE)
edges <- merge(edges, nbr_data, by.x = c("nbr", "year"), by.y = c("nbr_id", "year"), all.x = TRUE,
               suffixes = c("_src", "_nbr"))

# Compute neighbor stats efficiently for each source-year
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_list <- lapply(neighbor_vars, function(v) {
  edges[, .(
    paste0(v, "_nbr_max") = max(get(paste0(v, "_nbr")), na.rm = TRUE),
    paste0(v, "_nbr_min") = min(get(paste0(v, "_nbr")), na.rm = TRUE),
    paste0(v, "_nbr_mean") = mean(get(paste0(v, "_nbr")), na.rm = TRUE)
  ), by = .(src, year)]
})

# Combine aggregated neighbor stats
agg <- Reduce(function(x, y) merge(x, y, by = c("src", "year")), agg_list)

# Merge back into cell_data
cell_data <- merge(cell_data, agg, by.x = c("id", "year"), by.y = c("src", "year"), all.x = TRUE)

# Preserve trained model (no retraining needed)
# `cell_data` now includes optimized neighbor features
```

**Expected Impact**  
- Eliminates per-cell loops and string-intensive indexing overhead.
- Uses efficient data.table merges and grouping, reducing 86+ hours to a few hours or less on a standard laptop.
- Preserves original numerical estimand and Random Forest model.
END_RESPONSE>>>
