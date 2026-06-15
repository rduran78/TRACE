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
run_id: OpenAI_F07_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is very slow (86+ hours) because:  
- It uses nested `lapply` calls over ~6.46M rows, repeatedly scanning vectors and constructing neighbor indices.  
- The neighbor lookup is recomputed for each variable loop, causing redundant work.  
- `lapply` + `do.call(rbind, ...)` creates large intermediate objects and incurs R interpreter overhead.  
- No vectorization or compiled code leverages the fixed neighbor structure.  

**Optimization Strategy**  
- Precompute `neighbor_lookup` once and reuse it for all variables.  
- Flatten neighbor relationships into an edge list (source â†’ target) so that aggregation can use fast vectorized methods (e.g., `data.table` or `collapse`), avoiding millions of small function calls.  
- Compute max, min, and mean in one grouped operation rather than per-row loops.  
- Use `data.table` for efficient joins and aggregation.  
- Memory: process in chunks if needed, but 16 GB RAM can handle 6.5M rows with efficient structures.  

---

### **Optimized Approach**
Represent the panel as a `data.table`. Create an edge list of `(row_id, neighbor_row_id)`. Join neighbor values via this edge list and aggregate:  

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs
# rook_neighbors_unique: spdep::nb object

# 1. Add row index for fast reference
cell_data[, row_id := .I]

# 2. Build edge list: (source_row, neighbor_row)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

edges_list <- lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  src_id <- id_order[i]
  nbr_ids <- id_order[rook_neighbors_unique[[i]]]
  data.table(src_id = src_id, nbr_id = nbr_ids)
})
edges <- rbindlist(edges_list)

# Expand edge list for all years
years <- unique(cell_data$year)
edges_year <- CJ(year = years, src_id = edges$src_id, nbr_id = edges$nbr_id)

# Map to row indices
idx_lookup <- cell_data[, .(row_id, key = paste(id, year, sep = "_"))]
edges_year[, `:=`(
  src_key = paste(src_id, year, sep = "_"),
  nbr_key = paste(nbr_id, year, sep = "_")
)]
setkey(idx_lookup, key)
edges_year[, src_row := idx_lookup[src_key, row_id]]
edges_year[, nbr_row := idx_lookup[nbr_key, row_id]]
edges_year <- edges_year[!is.na(src_row) & !is.na(nbr_row), .(src_row, nbr_row)]

# 3. For each variable, compute neighbor max, min, mean
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  dt <- data.table(src_row = edges_year$src_row,
                   val = vals[edges_year$nbr_row])
  agg <- dt[, .(
    paste0(var_name, "_nbr_max") = max(val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") = min(val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") = mean(val, na.rm = TRUE)
  ), by = src_row]
  
  # Merge back to cell_data
  setkey(agg, src_row)
  cell_data <- merge(cell_data, agg, by.x = "row_id", by.y = "src_row", all.x = TRUE)
}

# Remove helper columns if desired
cell_data[, row_id := NULL]
```

---

### **Performance Gains**
- The expensive per-row `lapply` loop is replaced by a single edge-based join and grouped aggregation.  
- `data.table` uses optimized C-level loops, dramatically reducing runtime from ~86 hours to a few minutes (depending on disk speed and CPU).  
- Preserves original rook-neighbor relationships, numerical estimand, and trained Random Forest model.  

**Result:** Exact representation, fully vectorized neighbor statistics (max, min, mean) with efficient memory use and speed.
END_RESPONSE>>>
