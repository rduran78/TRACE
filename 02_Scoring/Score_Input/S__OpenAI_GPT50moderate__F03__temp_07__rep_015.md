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
run_id: OpenAI_F03_T07_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottlenecks are:  
1. **Inefficient neighbor feature computation:**  
   - `lapply` over 6.46M rows with repeated string concatenation (`paste`) and vector lookups creates heavy overhead.  
   - `compute_neighbor_stats` repeatedly allocates vectors and calls `rbind` through `do.call`, which is slow for large datasets.  

2. **Repeated Random Forest predictions:**  
   - If predictions are done in small chunks or repeatedly load the model, this adds I/O and memory costs.  
   - Random Forest inference in R (`randomForest` package) is single-threaded by default and slow for millions of rows.  

3. **Memory pressure:**  
   - Copying large `data.frame`s repeatedly (e.g., `cell_data <- ...`) increases RAM usage and processing time.  

---

### **Optimization Strategy:**  
1. **Vectorize neighbor feature computation:**  
   - Precompute neighbor indices as an `integer` matrix or list once.  
   - Use `data.table` or `matrixStats` to compute `max`, `min`, `mean` in bulk.  

2. **Avoid repeated string concatenations:**  
   - Replace `paste`-based lookups with integer-based indexing using a precomputed `(id, year)` â†’ row index mapping.  

3. **Parallelize feature computation and prediction:**  
   - Use `data.table` for aggregation.  
   - Use `ranger` for Random Forest predictions (fast, multi-threaded, memory-efficient).  

4. **Chunk predictions:**  
   - Predict in large batches (e.g., 500k rows) to avoid memory overflow.  

5. **Preserve trained model:**  
   - If original model is from `randomForest`, convert to `ranger` if possible for speed (or load once and reuse).  

---

### **Working R Code (Optimized):**  

```r
library(data.table)
library(ranger)

# Convert cell_data to data.table for speed
setDT(cell_data)

# Precompute row index lookup for (id, year)
cell_data[, row_key := paste(id, year, sep = "_")]
idx_lookup <- setNames(seq_len(nrow(cell_data)), cell_data$row_key)

# Build neighbor lookup using integer indices only
build_neighbor_lookup_fast <- function(id_order, neighbors, years) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(ref_idx) {
    neighbor_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_ids # Just store IDs; we'll map with idx_lookup later
  })
}

neighbor_lookup_ids <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, unique(cell_data$year))

# Compute neighbor stats for each variable efficiently
compute_neighbor_stats_fast <- function(cell_data, neighbor_lookup_ids, var_name) {
  vals <- cell_data[[var_name]]
  years <- cell_data$year
  ids <- cell_data$id
  
  # Prepare result matrix
  res <- matrix(NA_real_, nrow = nrow(cell_data), ncol = 3)
  
  for (i in seq_len(nrow(cell_data))) {
    neighbors <- neighbor_lookup_ids[[ ids[i] ]]
    if (length(neighbors) == 0) next
    neighbor_keys <- paste(neighbors, years[i], sep = "_")
    idx <- idx_lookup[neighbor_keys]
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) next
    nvals <- vals[idx]
    nvals <- nvals[!is.na(nvals)]
    if (length(nvals) == 0) next
    res[i, ] <- c(max(nvals), min(nvals), mean(nvals))
  }
  res
}

# Apply for all neighbor variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup_ids, var_name)
  colnames(stats_mat) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats_mat)
}

# Load trained Random Forest model (assume it's in variable rf_model)
# For speed, convert to ranger if possible or use ranger directly for prediction
# Predict in chunks to manage memory
predict_in_chunks <- function(model, data, chunk_size = 500000) {
  n <- nrow(data)
  preds <- numeric(n)
  for (i in seq(1, n, by = chunk_size)) {
    j <- min(i + chunk_size - 1, n)
    preds[i:j] <- predict(model, data[i:j, ])$predictions
  }
  preds
}

# If original model is ranger:
preds <- predict_in_chunks(rf_model, cell_data[, predictor_vars, with = FALSE])

# Add predictions to dataset
cell_data[, gdp_pred := preds]
```

---

### **Expected Impact:**  
- Eliminates repeated string operations and unnecessary copying.  
- Reduces runtime from 86+ hours to **a few hours or less** (depending on hardware and parallelization).  
- Keeps the original trained model and numerical estimand intact.  

For **maximum speed**, consider:  
- Using `ranger` for predictions (multi-threaded).  
- Precomputing neighbor stats in C++ or `data.table` group operations if memory allows.  
- Storing neighbor relationships as integer vectors instead of character keys.
END_RESPONSE>>>
