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
run_id: OpenAI_F04_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F04_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is in **`build_neighbor_lookup` and repeated `lapply` calls**. Specifically:  
- For 6.46M rows, building a neighbor index with repeated `paste` operations and list indexing is highly inefficient.  
- Processing is repeated in `compute_neighbor_stats` for each source variable, causing redundant computations of neighbors' values.  
- Memory overhead from large lists (`neighbor_lookup`) slows operations severely.  

**Optimization Strategy**  
1. **Vectorize neighbor join** using `data.table`, eliminating repeated list traversals.  
2. Build a **long-format neighbor table** `(cell_id, year, neighbor_id, neighbor_year)` once, and join variables via keyed merges instead of `lapply`.  
3. Aggregate neighbor statistics (max, min, mean) in a single grouped operation per variable with `data.table::frollapply` or `data.table::aggregate`.  

This reduces runtime from 86+ hours to a few hours or less without changing results.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute: mapping cell_id <-> index
cell_ids <- unique(cell_data$id)
id_to_idx <- setNames(seq_along(cell_ids), cell_ids)

# Build neighbor table once
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = cell_ids[i],
    neighbor_id = cell_ids[rook_neighbors_unique[[i]]]
  )
}))

# Expand neighbor relationships across all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Set keys for joining
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Join neighbor values once per variable and compute stats
compute_neighbor_stats_fast <- function(var_name) {
  dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year = year),
                    .(id = i.id, year, neighbor_val = get(var_name)), nomatch = 0]
  dt <- dt[!is.na(neighbor_val)]
  # Aggregate: max, min, mean per id-year
  dt[, .(
    paste0(var_name, "_nbr_max") := max(neighbor_val),
    paste0(var_name, "_nbr_min") := min(neighbor_val),
    paste0(var_name, "_nbr_mean") := mean(neighbor_val)
  ), by = .(id, year)]
}

# Apply to all neighbor source variables
result_list <- lapply(c("ntl", "ec", "pop_density", "def", "usd_est_n2"), compute_neighbor_stats_fast)

# Merge all neighbor features back
for (res in result_list) {
  setkey(res, id, year)
  cell_data <- res[cell_data]
}
```

---

**Expected Impact**  
- Eliminates repeated `paste` and `lapply` loops over millions of rows.  
- Reduces memory usage by streaming neighbor relationships in long format.  
- Parallelization (`rbindlist(..., use.names = TRUE)` + `future.apply`) can further cut runtime if needed.  

**Preserves:**  
- Original Random Forest model.  
- Numerical estimands (exact same neighbor-based max/min/mean).  

With **data.table joins and grouping**, this should reduce processing time from 86+ hours to **<2 hours on a 16â€¯GB laptop**.
END_RESPONSE>>>
