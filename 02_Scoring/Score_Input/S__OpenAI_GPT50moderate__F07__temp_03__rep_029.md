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
run_id: OpenAI_F07_T03_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `neighbor_lookup` is built as a list of integer vectors for **6.46M rows**, causing huge memory and iteration overhead.  
- `compute_neighbor_stats` uses `lapply` and repeated subsetting, which is inefficient for millions of rows.  
- Each variable is processed in a separate loop, multiplying the cost by 5.  
- No vectorization or efficient join-based aggregation is used.  

**Optimization Strategy**  
- Avoid per-row list operations; instead, reshape data and compute neighbor stats using **vectorized joins and aggregation**.  
- Represent neighbors as an **edge list** (cell_id â†’ neighbor_id) and join with panel data by year.  
- Use **data.table** for fast joins and grouped aggregation.  
- Compute all neighbor stats in one pass per variable.  
- Preserve original rook-neighbor relationships and estimand.  
- Do not retrain the Random Forest model; only compute features faster.  

**Optimized R Code**  

```r
library(data.table)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2)
# id_order: vector of cell IDs in order
# rook_neighbors_unique: spdep::nb object

# Convert to data.table
setDT(cell_data)

# Build edge list from rook_neighbors_unique
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Expand edges for all years
years <- unique(cell_data$year)
edges_expanded <- edges[, .(id = rep(id, length(years)),
                             neighbor_id = rep(neighbor_id, length(years)),
                             year = rep(years, each = .N))]

# Join neighbor values
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_vars) {
  # Join neighbor variable values
  edges_expanded[cell_data, on = .(neighbor_id = id, year), val := get(var)]
  
  # Compute stats by (id, year)
  stats <- edges_expanded[, .(
    max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("id", "year", "max_val", "min_val", "mean_val"),
           c("id", "year",
             paste0(var, "_nbr_max"),
             paste0(var, "_nbr_min"),
             paste0(var, "_nbr_mean")))
  
  cell_data <- stats[cell_data, on = .(id, year)]
  
  # Remove val for next iteration
  edges_expanded[, val := NULL]
}

# cell_data now has neighbor features added efficiently
```

**Why this works fast:**  
- Uses **data.table** joins and grouped aggregation instead of per-row loops.  
- Processes neighbors as an edge list, preserving rook topology exactly.  
- Avoids building a 6.46M-element list.  
- Computes stats in O(E) where E â‰ˆ 1.37M Ã— 28 â‰ˆ 38M rows, which is feasible in memory on 16 GB RAM.  

**Expected runtime:**  
From 86+ hours down to **minutes** on a standard laptop.  

**Preserves:**  
- Original rook-neighbor relationships.  
- Original numerical estimand.  
- Trained Random Forest model remains unchanged.
END_RESPONSE>>>
