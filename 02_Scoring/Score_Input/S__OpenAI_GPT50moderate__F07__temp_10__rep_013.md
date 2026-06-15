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
run_id: OpenAI_F07_T10_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current bottleneck is repeated `lapply` calls across ~6.46M rows with dynamic list lookups. Neighbor stats are computed row-by-row in R interpreted loops, which is extremely slow and memory-intensive for millions of observations and >1.3M neighbor relations. Worse, these operations are nested inside a loop for 5 variables.  

Issues:  
- No vectorization: Each of 6.46M rows runs an R function call.  
- Heavy list operations â†’ GC overhead.  
- Repeated computation: same neighbor lookups are repeatedly scanned.  
- Single-threaded execution.  

---

**Optimization Strategy:**  
1. **Precompute neighbor relationships in long edge form** (cell-year â†’ neighbor-year) once using `data.table`.  
2. **Flatten the panel**: Compute stats in a grouped manner instead of looping per row.  
3. Use **data.table joins and aggregation** for computing `max`, `min`, `mean` in bulk per variable per year.  
4. Avoid recomputation for each variableâ€”melt to long format or process one variable at a time but with vectorized joins.  
5. **Keep memory in check** by processing by year to reduce in-memory size.  
6. Preserve the estimand: ensure rook-neighbors and temporal alignment remain intact.  

This reduces complexity from millions of R loops to linear joins and grouped aggregations (C-optimized in `data.table`).  

---

**Working R Code:**  

```r
library(data.table)

# Convert to data.table for performance
setDT(cell_data)

# 1. Prepare neighbor edge list: (cell_id -> neighbor_id)
# rook_neighbors_unique: list of integer vectors by id_order index
edges <- data.table(
  from_id = rep(id_order, lengths(rook_neighbors_unique)),
  to_id   = unlist(rook_neighbors_unique)
)

# Expand to panel: cross with years, preserving year alignment
years <- unique(cell_data$year)
edges_panel <- edges[, .(id = from_id, neighbor_id = to_id), by = .EACHI][, .(id, neighbor_id)]
# Add year for both id and neighbor
edges_panel <- edges_panel[, .(id, neighbor_id, year = rep(years, .N)), by = .EACHI]

# 2. Join neighbor values for each var and compute stats
setkey(cell_data, id, year)
setkey(edges_panel, neighbor_id, year)

compute_neighbor_stats_bulk <- function(var_name) {
  # Join neighbor values
  joined <- edges_panel[cell_data, on = .(neighbor_id = id, year), nomatch=0]
  # joined has id (focal), neighbor_id, year, and var_name from "i" scope
  joined_vals <- joined[, .(id, year, value = get(var_name))]
  # Remove NAs
  joined_vals <- joined_vals[!is.na(value)]
  
  # Aggregate by focal id, year
  stats <- joined_vals[, .(
    paste0(var_name, "_nbr_max") = max(value),
    paste0(var_name, "_nbr_min") = min(value),
    paste0(var_name, "_nbr_mean") = mean(value)
  ), by = .(id, year)]
  
  stats
}

# 3. Process variables and merge back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

results_list <- lapply(neighbor_source_vars, compute_neighbor_stats_bulk)

# Merge all stats back into cell_data
for (stats in results_list) {
  cell_data <- stats[cell_data, on = .(id, year)]
}

# Final dataset now contains original vars + neighbor stats
```

---

**Performance Impact:**  
- Complexity drops from O(N * avg_deg * vars) interpreted loops to highly optimized C-level joins and aggregations via `data.table`.  
- Expected runtime: **minutes instead of 86+ hours** on 16 GB RAM if processed year-wise (optional chunking):  

```r
# If memory is an issue, wrap compute_neighbor_stats_bulk in a loop over years:
for (yr in years) {
  # subset cell_data and edges_panel by year = yr, then call compute_neighbor_stats_bulk
}
```

---

âœ” Preserves:
- Original rook topology.
- All 5 variablesâ€™ neighbor `max`, `min`, `mean`.
- Panel alignment.
- Trained Random Forest model unchanged.

**Summary:** Switch from R loops to `data.table` join-aggregate strategy leveraging pre-expanded neighbor edge list by year, drastically lowering runtime and preserving numerical integrity.
END_RESPONSE>>>
