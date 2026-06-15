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
run_id: OpenAI_F03_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 ### **Diagnosis**

The main bottlenecks in the pipeline are:

1. **Prediction Workflow Inefficiency**:
   - Prediction happens repeatedly for individual rows or chunks instead of batching.
   - The Random Forest (`randomForest` or similar) model inference is relatively fast when vectorized, but repeated data framing and object copying explode run time.

2. **Neighbor Feature Computation**:
   - `build_neighbor_lookup` returns a massive `list` of length â‰ˆ 6.46M, each with vectors of neighbor indices.
   - `compute_neighbor_stats` uses `lapply` and repeated `rbind` (`do.call(rbind, ...)`) â†’ memory thrash and huge overhead.
   - Feature computation happens sequentially for each variable, causing repeated index lookups.

3. **Memory Use & Copying**:
   - Each `compute_and_add_neighbor_features` call copies `cell_data` again.
   - Storing neighbor_lookup as a list of length 6.46M in RAM is impractical on 16 GB system.

---

### **Optimization Strategy**

- **Vectorize neighbor feature stats**:
  - Instead of `lapply` per observation, use long-form data and `data.table` joins or `matrixStats`.
- **Precompute**:
  - Flatten neighbor relationships into a long table for efficient aggregation grouped by `cell-year`.
- **Batch predictions**:
  - Call `predict(rf_model, newdata, type="response")` on the full data frame (or large chunks if RAM limited).
- **Efficient storage**:
  - Avoid huge lists; store neighbors as vectors in a long data structure (`source_id`, `neighbor_id`).
- **Leverage `data.table`**:
  - Fast grouping and aggregation.

---

### **Optimized Approach**

Steps:
1. Represent neighbors in a long format:  
   `(cell_id, neighbor_id, year)`.
2. Join neighbor values for each variable once, compute `max`, `min`, `mean` grouped by `(cell_id, year)`.
3. Add results back to `cell_data`.
4. Perform Random Forest prediction in **large batches or full dataset**.

---

### **Working R Code**

```r
library(data.table)
library(randomForest)

# Assume: cell_data is a data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# rook_neighbors_unique: list where rook_neighbors_unique[[i]] are neighbor cell indices
# id_order: vector mapping index to actual id
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  data.table(source_id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Cartesian join with years to create full neighbor-year mapping
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_dt[, .(id = source_id, neighbor_id, year = rep(years, each = .N)), by = source_id][, source_id := NULL]

setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join value of var_name from cell_data onto neighbor_dt by neighbor_id-year
  neighbor_vals <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(neighbor_vals, neighbor_id, year)
  
  nd <- neighbor_dt[neighbor_vals, nomatch = 0]  # attach var to neighbor mapping
  # Aggregate max, min, mean for each target (id-year)
  stats <- nd[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back to main table
  setkey(stats, id, year)
  cell_data[stats, paste0(var_name, "_nbr_max") := i.max_val]
  cell_data[stats, paste0(var_name, "_nbr_min") := i.min_val]
  cell_data[stats, paste0(var_name, "_nbr_mean") := i.mean_val]
}

# Random Forest Prediction on full dataset
# rf_model: pre-trained Random Forest
# Assume predictors is the vector of column names for model
batch_size <- 1e6
n <- nrow(cell_data)
predictions <- numeric(n)

for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(n, i + batch_size - 1)
  predictions[idx] <- predict(rf_model, newdata = cell_data[idx, ..predictors])
}

cell_data[, rf_pred := predictions]
```

---

### **Expected Gains**

- **Time**: Reduces from **86+ hours to likely under a few hours** (mostly I/O bound).
- **Memory**: Avoids massive lists, uses long `data.table` joins and in-place updates.
- **Preserves** original numerical values and uses the trained Random Forest model unchanged.

This approach is fully parallelizable (`data.table` grouping and join operations and `predict` can run in `parallel::mclapply` for chunks).
END_RESPONSE>>>
