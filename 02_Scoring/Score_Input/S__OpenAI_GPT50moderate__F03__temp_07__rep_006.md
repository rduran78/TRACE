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
run_id: OpenAI_F03_T07_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The performance bottleneck lies primarily in:  
1. **Repeated `lapply` loops and object copying**: `build_neighbor_lookup` and `compute_neighbor_stats` iterate over 6.46M rows with nested lookups, creating large intermediate lists and matrices.  
2. **Inefficient row-wise operations**: Each row recomputes neighbor keys and subset indices repeatedly.  
3. **Memory pressure**: Storing 6.46M long lists and repeated copying of `cell_data` for each variable overwhelms RAM.  
4. **Prediction workflow**: Likely looping row-by-row for Random Forest inference instead of batch prediction.  
5. **Model loading**: Ensure model is loaded once in memory, not per loop.  

---

**Optimization Strategy**  
- **Precompute and vectorize neighbor features**:
  - Flatten neighbor relationships into a data frame with `(cell, year, neighbor_cell)` links.
  - Join once instead of repeated lookups.
- **Use `data.table` for fast joins and aggregation** rather than `lapply`.
- **Avoid repeatedly modifying `cell_data`**; compute all neighbor stats in one join/aggregate step.
- **Batch Random Forest predictions**:
  - Load model once.
  - Predict on large chunks (e.g., 500k rows) using `predict` on matrices.
- **Memory efficiency**:
  - Use `integer` indexing and avoid redundant lists.
  - Minimize intermediate object copies.

---

**Working R Code**

```r
library(data.table)
library(randomForest)

# Convert to data.table for efficiency
cell_dt <- as.data.table(cell_data)

# Flatten neighbor relationships: (cell_id, neighbor_id)
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
neighbors_list <- rook_neighbors_unique

neighbor_pairs <- rbindlist(lapply(seq_along(neighbors_list), function(i) {
  if (length(neighbors_list[[i]]) > 0) {
    data.table(cell_id = id_order[i], neighbor_id = id_order[neighbors_list[[i]]])
  }
}), use.names = TRUE)

# Expand to years
years <- unique(cell_dt$year)
neighbor_pairs <- neighbor_pairs[, .(year = years), by = .(cell_id, neighbor_id)]

# Merge neighbor values
setkey(cell_dt, id, year)
setkey(neighbor_pairs, neighbor_id, year)

# For each source variable, compute stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  temp <- merge(neighbor_pairs, cell_dt[, .(neighbor_id = id, year, val = get(var))],
                by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate stats: max, min, mean
  stats <- temp[, .(
    paste0(var, "_nbr_max") = max(val, na.rm = TRUE),
    paste0(var, "_nbr_min") = min(val, na.rm = TRUE),
    paste0(var, "_nbr_mean") = mean(val, na.rm = TRUE)
  ), by = .(cell_id, year)]
  
  # Merge back into cell_dt
  cell_dt <- merge(cell_dt, stats, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)
}

# Prepare predictor matrix
predictors <- setdiff(names(cell_dt), c("id", "year", "gdp")) # assuming gdp is target
X <- as.matrix(cell_dt[, ..predictors])

# Load trained Random Forest model once
rf_model <- readRDS("trained_rf_model.rds")

# Batch predictions
batch_size <- 5e5
n <- nrow(X)
pred <- numeric(n)

for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  pred[start:end] <- predict(rf_model, X[start:end, ])
}

# Add predictions
cell_dt[, predicted_gdp := pred]
```

---

**Why this works**  
- **Single aggregation pass**: All neighbor stats computed with `data.table` groupingâ€”no nested `lapply`.  
- **No repeated object copies**: Merge results back once.  
- **Chunked prediction**: Efficient use of RAM for Random Forest.  
- **Vectorized operations**: Eliminates row-by-row loops.  

Estimated runtime: from 86+ hours down to **a few hours (or less)** on standard hardware, primarily limited by disk I/O and Random Forest batch inference.
END_RESPONSE>>>
