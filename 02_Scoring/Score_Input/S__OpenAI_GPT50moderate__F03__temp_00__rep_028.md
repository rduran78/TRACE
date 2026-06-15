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
run_id: OpenAI_F03_T00_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small R objects. This is extremely slow and memory-inefficient.  
2. **Excessive object copying**: Each `compute_and_add_neighbor_features` likely copies the entire `data.frame` repeatedly.  
3. **Inefficient neighbor lookup**: `lapply` with string concatenation (`paste`) for every row is costly.  
4. **Random Forest prediction overhead**: If predictions are done in small chunks or with repeated model loading, this adds significant time.  

**Optimization Strategy**  
- **Precompute neighbor indices as integer vectors** (avoid string keys).  
- **Vectorize neighbor feature computation** using `data.table` or `matrix` operations instead of millions of `lapply` calls.  
- **Batch predictions**: Use `predict(model, newdata, type="response")` on large chunks or entire dataset if memory allows.  
- **Avoid repeated copies**: Work with `data.table` in-place updates.  
- **Parallelize**: Use `parallel::mclapply` or `future.apply` for neighbor stats if needed.  

---

### **Optimized R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup as integer indices
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  # Flatten neighbor list into two-column matrix: (cell_idx, neighbor_idx)
  from <- rep(seq_along(neighbors), lengths(neighbors))
  to   <- unlist(neighbors, use.names = FALSE)
  data.table(cell_idx = from, neighbor_idx = to)
}

neighbor_dt <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# Map cell IDs to row indices for each year
cell_data[, row_idx := .I]
id_year_key <- cell_data[, .(row_idx), keyby = .(id, year)]

# Join neighbor pairs with year to get actual row indices
neighbor_dt <- neighbor_dt[
  id_year_key, on = .(cell_idx = id), allow.cartesian = TRUE
][
  id_year_key, on = .(neighbor_idx = id, year), allow.cartesian = TRUE
][
  !is.na(row_idx) & !is.na(i.row_idx),
  .(row_idx = i.row_idx, neighbor_row = row_idx)
]

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(var_name) {
  vals <- cell_data[[var_name]]
  neighbor_vals <- vals[neighbor_dt$neighbor_row]
  neighbor_dt[, val := neighbor_vals]
  
  # Aggregate max, min, mean by row_idx
  stats <- neighbor_dt[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = row_idx]
  
  # Merge back into cell_data
  setnames(stats, c("row_idx", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data[stats, on = .(row_idx), `:=`(
    (paste0(var_name, "_nbr_max")) = i.get(paste0(var_name, "_nbr_max")),
    (paste0(var_name, "_nbr_min")) = i.get(paste0(var_name, "_nbr_min")),
    (paste0(var_name, "_nbr_mean")) = i.get(paste0(var_name, "_nbr_mean"))
  )]
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(var_name)
}

# Drop helper columns
cell_data[, row_idx := NULL]

# Random Forest prediction in large batch
# Assume rf_model is already loaded
predictions <- predict(rf_model, newdata = cell_data, type = "response")
```

---

### **Why This Is Faster**
- **No per-row `lapply`**: All neighbor relationships handled in a single `data.table` join and aggregation.
- **No repeated string concatenation**: Integer joins instead of `paste`.
- **Single pass per variable**: Aggregation is vectorized.
- **Minimal copying**: In-place updates with `:=`.
- **Batch prediction**: One call to `predict()`.

---

**Expected Performance Gain**  
From 86+ hours â†’ likely **under 1â€“2 hours** on a 16 GB laptop, depending on disk I/O and CPU. Further speedup possible with `setDTthreads()` or parallel aggregation.
END_RESPONSE>>>
