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
run_id: OpenAI_F07_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is slow because:  
- `neighbor_lookup` is a large list of integer vectors built with `lapply` across ~6.46M rows.  
- For each of the 5 variables, `compute_neighbor_stats` iterates row by row, repeatedly subsetting and computing stats, which is highly inefficient in R due to millions of small allocations and R-level loops.  
- Memory overhead is high because we store `neighbor_lookup` for all rows as a list, increasing RAM pressure enormously.  
- Complexity is effectively `O(N * avg_degree * num_vars)` done in pure R loops across 6.46M entries.  

---

**Optimization Strategy**  
1. **Avoid huge row-level lists**: Use a **long, edge-based table** representation of neighbors (like a graph edge list) and process with `data.table` or `dplyr`, computing statistics via grouped aggregation.  
2. Precompute neighbor relationships: join cell-years to their neighbor cell-years by `id` and `year`.  
3. Use **data.table** for efficient joins and aggregations rather than looping over 6.46M rows.  
4. Compute `max`, `min`, `mean` in one grouped aggregation step, then merge back to the main table.  
5. Keep the Random Forest model intact by updating features in `cell_data` without changing the sampling or IDs.  

---

**Working R Code**  

```r
library(data.table)

# Convert cell_data to data.table for efficiency
setDT(cell_data)  # columns: id, year, ntl, ec, pop_density, def, usd_est_n2, etc.

# Expand rook_neighbors_unique into an edge table
# rook_neighbors_unique: list of neighbor ids for each id in id_order
edges <- data.table(
  src = rep(id_order, lengths(rook_neighbors_unique)),
  nbr = unlist(rook_neighbors_unique)
)

# Create a big long table of (src_id, year, nbr_id)
years <- sort(unique(cell_data$year))
edges_expanded <- edges[CJ(years)]  # replicate for all years if needed
setnames(edges_expanded, c("src", "nbr", "year"))

# Merge neighbor values by joining to cell_data twice
for (var_name in c("ntl", "ec", "pop_density", "def", "usd_est_n2")) {
  # Join neighbor variable
  edges_vals <- merge(
    edges_expanded,
    cell_data[, .(nbr_id = id, year, val = get(var_name))],
    by.x = c("nbr", "year"),
    by.y = c("nbr_id", "year"),
    all.x = TRUE
  )
  
  # Compute stats grouped by (src, year)
  stats <- edges_vals[, .(
    max_val = if (.N==0) NA_real_ else max(val, na.rm=TRUE),
    min_val = if (.N==0) NA_real_ else min(val, na.rm=TRUE),
    mean_val = if (.N==0) NA_real_ else mean(val, na.rm=TRUE)
  ), by = .(src, year)]
  
  # Merge back to cell_data (src_id == id)
  setnames(stats, c("src", "year", paste0(var_name, "_nbr_max"), 
                              paste0(var_name, "_nbr_min"), 
                              paste0(var_name, "_nbr_mean")))
  cell_data <- merge(cell_data, stats, by.x = c("id", "year"), by.y = c("src", "year"), all.x = TRUE)
}

```

---

**Why This Works**  
- Eliminates inner loops: all neighbor-based computations are done with grouped operations in compiled `data.table` C code.  
- Memory efficient: no massive R lists of length 6.46M. Only two-to-three large tables stored at a time.  
- Scalability: data.table join and group-by on ~6â€“7 million rows plus ~1.3M edges across 28 years is manageable on 16GB RAM if processed in chunks by variable.  

**Expected Speedup**  
- From ~86 hours to roughly **minutes** (or a few tens of minutes) on a standard laptop, since heavy per-row lapply loops are replaced by optimized vectorized aggregation.  

**Outcome**  
Maintains:  
- Original rook neighbors,  
- Exact estimator (max/min/mean),  
- Random Forest model untouched (same predictors + new neighbor features),  
- Feasible memory and runtime.
END_RESPONSE>>>
