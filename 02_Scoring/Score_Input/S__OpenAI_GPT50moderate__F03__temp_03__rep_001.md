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
run_id: OpenAI_F03_T03_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated list-based neighbor lookups and `lapply` loops**: `build_neighbor_lookup` and `compute_neighbor_stats` create and traverse large lists (6.46M rows), causing high memory churn and R interpreter overhead.  
2. **Inefficient row-wise operations**: Each row computes neighbors individually, leading to ~6.46M Ã— 5 passes.  
3. **Random Forest prediction overhead**: If predictions are done in small batches or with repeated model loading, this adds significant time.  
4. **Memory pressure**: Storing large lists of integer vectors for neighbors and repeatedly copying `cell_data`.  

---

### **Optimization Strategy**
- **Vectorize neighbor feature computation**: Avoid per-row `lapply` by using `data.table` joins or matrix aggregation.
- **Precompute neighbor relationships in long format**: Create a table of `(cell_id, year, neighbor_id)` and join features once.
- **Batch Random Forest predictions**: Load the model once, predict in large chunks (or all at once if memory allows).
- **Use `data.table` for all operations**: Efficient joins and aggregations in C.
- **Avoid repeated copying**: Modify in place.

---

### **Optimized Workflow**
1. Convert `cell_data` to `data.table`.
2. Expand neighbor relationships across years in long format.
3. Join neighbor values for each variable, compute `max`, `min`, `mean` via `data.table` aggregation.
4. Merge aggregated stats back to `cell_data`.
5. Predict in large batches using `predict()` on the full feature matrix.

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data (id, year, predictors), id_order, rook_neighbors_unique, rf_model loaded

setDT(cell_data)

# 1. Build neighbor pairs in long format
id_dt <- data.table(id = id_order, idx = seq_along(id_order))
nb_list <- rook_neighbors_unique

# Create edge list
edges <- rbindlist(lapply(seq_along(nb_list), function(i) {
  if (length(nb_list[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[nb_list[[i]]])
}))

# Expand across years
years <- unique(cell_data$year)
edges_expanded <- edges[, .(id = src, neighbor_id = nbr), by = .EACHI][
  , .(year = years), by = .(id, neighbor_id)]
setkey(edges_expanded, neighbor_id, year)

# 2. For each variable, compute neighbor stats
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Join neighbor values
  tmp <- merge(edges_expanded, cell_data[, .(neighbor_id = id, year, val = get(var))],
               by = c("neighbor_id", "year"), all.x = TRUE)
  
  # Aggregate stats by (id, year)
  stats <- tmp[!is.na(val), .(
    paste0(var, "_nbr_max") = max(val),
    paste0(var, "_nbr_min") = min(val),
    paste0(var, "_nbr_mean") = mean(val)
  ), by = .(id, year)]
  
  # Merge back to cell_data
  setkey(cell_data, id, year)
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# 3. Prepare feature matrix for prediction
predictors <- setdiff(names(cell_data), c("id", "year", "gdp"))  # adjust target name
X <- as.data.frame(cell_data[, ..predictors])

# 4. Predict in large batch
preds <- predict(rf_model, X)

# Attach predictions
cell_data[, gdp_pred := preds]
```

---

### **Expected Gains**
- Eliminates 6.46M row-wise loops â†’ replaced by vectorized joins and aggregations.
- Single-pass per variable instead of per-row.
- Memory-efficient `data.table` operations.
- Prediction in one batch avoids repeated overhead.

**Estimated runtime reduction**: From 86+ hours â†’ a few hours or less on 16 GB RAM.
END_RESPONSE>>>
