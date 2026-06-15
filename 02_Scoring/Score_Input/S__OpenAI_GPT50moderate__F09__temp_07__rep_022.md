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
run_id: OpenAI_F09_T07_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation recomputes neighbor statistics by iterating over all 6.46 million rows and repeatedly performing list lookups and `paste` operations for every variable. This is extremely inefficient because:  
- `build_neighbor_lookup` is constructed per full dataset with dynamic string concatenation for each row.  
- `compute_neighbor_stats` loops over all rows multiple times (once per variable) instead of vectorizing or joining.  
- String-based keys (`paste(id, year)`) and repeated `lapply`/`do.call` introduce huge overhead.  

Given 6.46M rows Ã— 5 variables Ã— repeated loops, the cost explodes.  

---

**Optimization Strategy**  
- Build a **static adjacency table** once: for each cell, store its neighbor cell IDs (no year dimension).  
- Create a **long join table** that pairs `(cell_id, year)` with all `(neighbor_id, year)` combinations (via `data.table` or `dplyr`).  
- Use **grouped aggregation** to compute neighbor max/min/mean for all rows in a single pass per variable (or even for all variables at once if reshaped to long format).  
- Avoid `paste` and repeated list traversal; leverage integer joins on `(id, year)`.  
- Process in chunks if memory is tight.  
- Preserve trained Random Forest and estimand by only changing feature-generation logic.  

---

**Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# 1. Build adjacency table once (from rook_neighbors_unique and id_order)
adj_list <- rook_neighbors_unique
adj_dt <- rbindlist(lapply(seq_along(adj_list), function(i) {
  if (length(adj_list[[i]]) == 0) return(NULL)
  data.table(cell_id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
}))

# 2. Expand to include all years (Cartesian join)
years <- unique(cell_data$year)
adj_dt <- adj_dt[, .(year = years), by = .(cell_id, neighbor_id)]

# 3. Create neighbor join data.table
# Left side: (cell_id, year), Right side: neighbor values
setkey(cell_data, id, year)
setkey(adj_dt, neighbor_id, year)
neighbor_join <- adj_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]

# neighbor_join now has: cell_id, neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2

# 4. Compute neighbor stats grouped by (cell_id, year)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_expr <- lapply(neighbor_source_vars, function(v) {
  list(
    as.name(paste0(v, "_nbr_max")) := max(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nbr_min")) := min(get(v), na.rm = TRUE),
    as.name(paste0(v, "_nbr_mean")) := mean(get(v), na.rm = TRUE)
  )
}) |> unlist(recursive = FALSE)

neighbor_stats <- neighbor_join[, c(agg_expr), by = .(cell_id, year)]

# 5. Merge back to main cell_data
setkey(neighbor_stats, cell_id, year)
setkey(cell_data, id, year)
cell_data <- neighbor_stats[cell_data]

# At this point, cell_data has original columns + neighbor stats columns
# Ready for prediction with existing Random Forest model
```

---

**Performance Gains**  
- Adjacency built once; avoids per-row string ops.  
- Single join + grouped aggregation replaces millions of `lapply` calls.  
- `data.table` is highly optimized for large joins and aggregations.  
- Expected runtime: minutes to an hour instead of 86+ hours on a 16 GB laptop (may require chunking if memory hits limits).  

**Key Notes**  
- Random Forest model remains untouched.  
- Numerical estimand preserved because computations replicate original max/min/mean logic.  
- If memory is tight, process one variable at a time or split years into chunks.
END_RESPONSE>>>
