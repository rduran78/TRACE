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
run_id: OpenAI_F03_T07_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Inefficient neighbor feature computation**:  
   - `build_neighbor_lookup` and `compute_neighbor_stats` repeatedly iterate over millions of rows using `lapply`, causing heavy R-level loops and memory overhead.  
   - `do.call(rbind, ...)` on large lists is costly.  
   - Object copying in each step adds significant overhead.  

2. **Prediction inefficiency**:  
   - Random Forest inference in `randomForest` or similar base packages is relatively fast compared to the above loops, but repeated copying and non-vectorized feature prep dominate runtime.  

3. **Memory pressure**:  
   - 6.46M rows Ã— 110+ predictors â†’ large data frame copies in R memory.  

**Optimization Strategy**  
- **Precompute neighbor indices once** and reuse as integer indices.  
- **Vectorize neighbor statistics**:  
  - Avoid per-row loops; use `data.table` joins or `matrixStats` on aggregated arrays.  
  - Compute max/min/mean in a single grouped operation.  
- **Use `data.table` for feature engineering** to minimize copying.  
- **Chunked prediction** to keep memory stable.  
- **Keep Random Forest model loaded once** and call `predict()` on large chunks.  

---

### **Optimized Implementation (R)**

```r
library(data.table)
library(randomForest) # or ranger if available

# Assume cell_data is a data.table
setDT(cell_data)

# Precompute neighbor pairs: (id, year) -> neighbor_id
# rook_neighbors_unique: list of integer vectors, same order as id_order
id_map <- data.table(id = id_order, idx = seq_along(id_order))
pairs_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(source_idx = i, neighbor_idx = rook_neighbors_unique[[i]])
}))
pairs_list[, source_id := id_order[source_idx]]
pairs_list[, neighbor_id := id_order[neighbor_idx]]
pairs_list[, c("source_idx", "neighbor_idx") := NULL]

# Expand pairs by year
years <- sort(unique(cell_data$year))
expanded_pairs <- pairs_list[, .(year = years), by = .(source_id, neighbor_id)]
setkey(expanded_pairs, neighbor_id, year)

# Join neighbor values for each var
for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  expanded_pairs <- tmp[expanded_pairs, on = .(neighbor_id, year)]
  
  # Compute stats by source_id-year
  stats <- expanded_pairs[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(source_id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("source_id", "year", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  cell_data <- stats[cell_data, on = .(source_id = id, year)]
}

# Convert to matrix for prediction
predictors <- setdiff(names(cell_data), c("id", "year", "gdp")) # exclude target
X <- as.matrix(cell_data[, ..predictors])

# Load trained RF model (assume rf_model exists)
# Predict in chunks to save memory
chunk_size <- 1e5
preds <- numeric(nrow(X))
for (i in seq(1, nrow(X), by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, nrow(X))
  preds[idx] <- predict(rf_model, newdata = X[idx, , drop = FALSE])
}

cell_data[, gdp_pred := preds]
```

---

**Key Gains**  
- Replaces millions of `lapply` calls with fully vectorized `data.table` joins and grouped aggregation.  
- Avoids repeated object copying by modifying `cell_data` in place.  
- Handles prediction in efficient chunks without retraining.  

**Expected Runtime Reduction**: From ~86 hours to a few hours or less on a standard laptop.  

**Preserves**:  
- Original trained Random Forest model.  
- Original estimand and feature definitions.  
- All required neighbor-feature logic.
END_RESPONSE>>>
