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
run_id: OpenAI_F02_T07_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 ### **Diagnosis**
The current approach is slow and memory-heavy due to:
- **Inefficient per-row `lapply`** calls for ~6.46M rows.
- Repeated string concatenation (`paste`) and dictionary lookups inside loops.
- Building neighbor stats via nested `lapply` and `do.call(rbind, ...)`.
- No vectorization or batching, causing high R interpreter overhead.

Given 6.46M rows and 1.37M neighbor links, this must be parallelized and vectorized.

---

### **Optimization Strategy**
1. **Precompute neighbor relationships as integer indices**, avoid repeated string concatenation.
2. **Vectorize neighbor stats computation** using `data.table` or `dplyr`.
3. **Parallelize computation** over variables and/or chunks of rows using `future.apply` or `parallel`.
4. **Avoid large in-memory lists**; store neighbor relationships in a long format table for efficient joins.
5. **Use `data.table` for joins and aggregation**, as it is memory-efficient and fast.

---

### **Optimized Approach**
- Flatten `neighbors` into a long table: `(cell_id, neighbor_id)`.
- Join with data by `(id, year)` once for all years.
- Aggregate (mean, min, max) by `(cell_id, year)` for each variable.
- Compute features in one grouped aggregation per variable instead of millions of small loops.

---

### **Working R Code**

```r
library(data.table)
library(future.apply)

# Convert data.frame to data.table
setDT(cell_data)

# Ensure keys
setkey(cell_data, id, year)

# Flatten neighbor list into a long format
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

setkey(neighbor_dt, neighbor_id)

# Expand neighbor relationships across all years by joining with years in cell_data
years <- unique(cell_data$year)
neighbor_year_dt <- neighbor_dt[CJ(neighbor_id = neighbor_id, year = years, unique = TRUE), on = .(neighbor_id)]

# Join to get neighbor values
compute_neighbor_features <- function(var_name) {
  # Join neighbor_year_dt with cell_data to get neighbor values
  dt <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  neighbor_vals <- neighbor_year_dt[dt, on = .(neighbor_id, year)]
  
  # Now aggregate stats by original id and year
  agg <- neighbor_vals[, .(
    paste0(var_name, "_nbr_max") := max(val, na.rm = TRUE),
    paste0(var_name, "_nbr_min") := min(val, na.rm = TRUE),
    paste0(var_name, "_nbr_mean") := mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Handle cases where all neighbors are NA
  for (col in names(agg)[-(1:2)]) set(agg, which(!is.finite(agg[[col]])), col, NA)
  
  agg
}

# Parallelize across variables
plan(multicore, workers = parallel::detectCores() - 1)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

feature_tables <- future_lapply(neighbor_source_vars, compute_neighbor_features)

# Merge all features back into cell_data
for (tbl in feature_tables) {
  cell_data <- tbl[cell_data, on = .(id, year)]
}
```

---

### **Why This Works**
- **No per-row loops**: All operations are vectorized using `data.table`.
- **Single join per variable**: Avoid repeated expensive lookups.
- **Parallelization**: Each variable is computed in parallel.
- **Memory-efficient**: Long-format join avoids massive lists.

---

### **Expected Performance**
- From 86+ hours â†’ **under 1â€“2 hours** on a 16 GB laptop (depending on I/O and CPU).
- Memory footprint: manageable (~few GB), as joins and aggregations are done in chunks.

---

âœ… Preserves:
- Original trained Random Forest model.
- Original numerical estimand.
- Full reproducibility of neighbor-based features.
END_RESPONSE>>>
