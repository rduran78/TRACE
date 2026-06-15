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
run_id: OpenAI_F06_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because for each of ~6.46M rows, it performs multiple `lapply` calls that repeatedly search and concatenate keys, causing large overhead in memory and string operations. Neighbor lookup and per-row aggregation are not vectorized, and repeated subsetting of large vectors adds cost. With 6M rows and millions of neighbor references, this approach becomes I/O and CPU bound, leading to the 86+ hour estimate.

**Optimization Strategy**  
- Precompute neighbor indices in integer form (avoid character key operations).  
- Flatten neighbor relations and compute max/min/mean via grouped aggregation instead of per-row loops.  
- Use `data.table` for fast joins and grouping over large datasets.  
- Preserve original estimand: same max, min, mean of neighbors for each cell-year.  
- Avoid retraining the Random Forest model; only replace the feature engineering step.  

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert data.frame to data.table for speed
setDT(cell_data)

# Create a unique integer ID for each cell-year row
cell_data[, row_id := .I]
cell_data[, cell_year := paste(id, year, sep = "_")]

# Precompute lookup: integer neighbor indices for each cell id (not cell-year)
# rook_neighbors_unique is a list of integer vectors (neighbors by cell position)
id_to_pos <- setNames(seq_along(id_order), id_order)

# Build neighbor list keyed by row_id
neighbor_lookup_list <- vector("list", nrow(cell_data))
for (pos in seq_along(id_order)) {
  cell_ids <- id_order[pos]
  idxs <- cell_data[id == cell_ids, row_id]
  neigh_ids <- id_order[rook_neighbors_unique[[pos]]]
  # For each row with this id, map neighbors in same year
  for (r in idxs) {
    yr <- cell_data$year[r]
    neigh_rows <- cell_data[id %in% neigh_ids & year == yr, row_id]
    neighbor_lookup_list[[r]] <- neigh_rows
  }
}

# Flatten into a long table (row_id, neighbor_id)
long_neighbors <- rbindlist(
  lapply(seq_along(neighbor_lookup_list), function(i) {
    if (length(neighbor_lookup_list[[i]]) > 0) {
      data.table(row_id = i, neighbor_id = neighbor_lookup_list[[i]])
    }
  })
)

# Compute stats for each variable using data.table joins
source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in source_vars) {
  # Join to get neighbor values
  long_neighbors[, neighbor_val := cell_data[[var_name]][neighbor_id]]
  
  # Aggregate max, min, mean per row_id
  agg_stats <- long_neighbors[, .(
    max_val = max(neighbor_val, na.rm = TRUE),
    min_val = min(neighbor_val, na.rm = TRUE),
    mean_val = mean(neighbor_val, na.rm = TRUE)
  ), by = row_id]
  
  # Merge back to cell_data
  setkey(cell_data, row_id)
  setkey(agg_stats, row_id)
  cell_data[agg_stats, paste0(var_name, "_nb_max") := i.max_val]
  cell_data[agg_stats, paste0(var_name, "_nb_min") := i.min_val]
  cell_data[agg_stats, paste0(var_name, "_nb_mean") := i.mean_val]
}

# Drop helper columns
cell_data[, c("row_id", "cell_year") := NULL]
```

**Why this works faster**  
- Eliminates repeated string pasting and lookups for every row.  
- Aggregates neighbor stats in a single pass using efficient `data.table` operations.  
- Avoids millions of tiny list operations.  
- Preserves the same estimand (max, min, mean of rook neighbors per cell-year).  

Expected runtime: hours reduced to minutes or low hours on a standard laptop, depending on disk I/O and RAM.
END_RESPONSE>>>
