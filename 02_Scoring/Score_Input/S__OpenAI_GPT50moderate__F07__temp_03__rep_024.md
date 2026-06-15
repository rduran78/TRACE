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
run_id: OpenAI_F07_T03_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow because:  
- It uses `lapply` over ~6.46M rows, repeatedly performing string concatenation and hash lookups (`paste`, `setNames`), which is highly inefficient.  
- Neighbor lookups are recomputed for each row and each variable.  
- The process is not vectorized and does not leverage efficient joins or matrix operations.  
- Memory overhead from large lists and repeated allocations is significant.  

**Optimization Strategy**  
1. **Precompute neighbor pairs once** as a long table of `(row_id, neighbor_row_id)` for all years.  
2. Use **data.table** for fast joins and aggregation.  
3. Compute max, min, and mean in a single grouped operation per variable.  
4. Avoid repeated string operations; use integer IDs for joins.  
5. Preserve the trained Random Forest model by only changing feature computation, not the modeling pipeline.  

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs
# rook_neighbors_unique: list of neighbor indices (spdep::nb)

# Convert to data.table
setDT(cell_data)

# Map id to index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build neighbor pairs for all years
# Expand neighbors for each cell-year
neighbor_pairs <- rbindlist(lapply(seq_len(nrow(cell_data)), function(i) {
  cell_id <- cell_data$id[i]
  yr      <- cell_data$year[i]
  ref_idx <- id_to_idx[[as.character(cell_id)]]
  neigh_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  if (length(neigh_ids) == 0) return(NULL)
  data.table(row_id = i, neighbor_key = paste0(neigh_ids, "_", yr))
}))

# Create lookup for neighbor_key -> neighbor_row_id
cell_data[, key := paste0(id, "_", year)]
setkey(cell_data, key)
neighbor_pairs[, neighbor_row_id := cell_data[neighbor_key, which = TRUE]]

# Drop missing
neighbor_pairs <- neighbor_pairs[!is.na(neighbor_row_id)]

# Convert to efficient structure
setkey(neighbor_pairs, row_id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  vals <- cell_data[[var_name]]
  dt <- data.table(row_id = neighbor_pairs$row_id,
                   val = vals[neighbor_pairs$neighbor_row_id])
  dt <- dt[!is.na(val)]
  stats <- dt[, .(max = max(val), min = min(val), mean = mean(val)), by = row_id]
  setnames(stats, c("row_id", paste0(var_name, "_nbr_max"),
                             paste0(var_name, "_nbr_min"),
                             paste0(var_name, "_nbr_mean")))
  stats
}

# Compute for all variables and merge
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stats_list <- lapply(neighbor_source_vars, compute_neighbor_stats_dt)

# Merge all stats back to cell_data
for (stats in stats_list) {
  cell_data <- merge(cell_data, stats, by.x = "row_id", by.y = "row_id", all.x = TRUE)
}
```

**Why this is faster:**  
- Builds neighbor relationships once, not repeatedly.  
- Uses integer indexing and `data.table` for efficient joins and aggregation.  
- Avoids repeated string concatenation inside loops.  
- Aggregates in bulk rather than row-by-row.  

**Expected performance:**  
From 86+ hours to a few minutes on a 16 GB laptop, as the heavy operations are now vectorized and memory-efficient.  

**Preserves:**  
- Original rook-neighbor relationships.  
- Exact numerical estimands (max, min, mean).  
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
