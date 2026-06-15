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
run_id: OpenAI_F03_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive neighbor computations**: `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of rows, creating large intermediate lists and repeated object copying.  
2. **Inefficient row-wise operations**: Each row recomputes neighbor stats separately, leading to huge overhead.  
3. **Memory pressure**: Storing large lists in R (6.46M elements) is inefficient.  
4. **Random Forest prediction overhead**: If predictions are done in small batches or loops, the model repeatedly processes overhead instead of vectorized prediction.  

**Optimization Strategy**  
- Precompute neighbor indices once as an **integer matrix** instead of lists.  
- Use **vectorized aggregation** for neighbor stats (e.g., `matrixStats` or `data.table` joins) instead of per-row `lapply`.  
- Avoid repeated `rbind` calls; build a numeric matrix directly.  
- Perform **batch predictions**: call `predict()` on large chunks (e.g., 500k rows) instead of row-wise.  
- Use **data.table** for fast joins and memory efficiency.  
- Keep the trained Random Forest model as-is; only optimize data prep and prediction workflow.  

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Convert to data.table for efficiency
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as a matrix
build_neighbor_matrix <- function(id_order, neighbors) {
  max_neighbors <- max(lengths(neighbors))
  mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_neighbors)
  for (i in seq_along(neighbors)) {
    nbs <- neighbors[[i]]
    if (length(nbs) > 0) {
      mat[i, seq_along(nbs)] <- nbs
    }
  }
  mat
}

neighbor_matrix <- build_neighbor_matrix(id_order, rook_neighbors_unique)

# Map id to row index for fast lookup
id_to_idx <- setNames(seq_along(id_order), id_order)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(dt, var_name, neighbor_matrix, id_to_idx) {
  vals <- dt[[var_name]]
  n <- nrow(dt)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    cell_id <- dt$id[i]
    year <- dt$year[i]
    nb_ids <- neighbor_matrix[id_to_idx[[as.character(cell_id)]], ]
    nb_ids <- nb_ids[!is.na(nb_ids)]
    if (length(nb_ids) == 0) next
    
    # Compute neighbor row indices for same year
    nb_keys <- paste(nb_ids, year, sep = "_")
    nb_idx <- dt[.(nb_ids, year), which = TRUE, on = .(id, year)]
    nb_idx <- nb_idx[!is.na(nb_idx)]
    if (length(nb_idx) == 0) next
    
    nb_vals <- vals[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    
    res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }
  
  colnames(res) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  res
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_features <- lapply(neighbor_source_vars, function(v) {
  compute_neighbor_stats_fast(cell_data, v, neighbor_matrix, id_to_idx)
})

neighbor_features <- do.call(cbind, neighbor_features)
cell_data <- cbind(cell_data, neighbor_features)

# Batch prediction
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (start in seq(1, n, by = batch_size)) {
  end <- min(start + batch_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

cell_data[, gdp_pred := preds]
```

---

### **Key Improvements**
- **Neighbor matrix** avoids large lists and repeated lookups.
- **Single-pass computation** for neighbor stats.
- **Batch predictions** minimize Random Forest overhead.
- **data.table** ensures fast joins and memory efficiency.

**Expected result**: Reduce runtime from 86+ hours to a few hours (depending on I/O and CPU).
END_RESPONSE>>>
