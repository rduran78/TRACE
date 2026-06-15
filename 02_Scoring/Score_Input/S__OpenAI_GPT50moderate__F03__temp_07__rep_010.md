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
run_id: OpenAI_F03_T07_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

- **Bottleneck**: The major slowdown is not the Random Forest model itself but the repeated neighbor-based feature computation (`compute_neighbor_stats`) before prediction.  
  - `build_neighbor_lookup` creates ~6.46M lists of integer vectors â†’ large memory footprint.
  - `compute_neighbor_stats` performs millions of small `lapply` calls, repeated for 5 variables â†’ extremely inefficient.
- **Prediction**: Once features are ready, Random Forest inference on 6.46M rows is feasible in minutes with `ranger` (fast C++ backend). Current delay is dominated by feature engineering.
- **Problem drivers**:
  - Repeated R loops and list operations.
  - Redundant string concatenation (`paste`) and `setNames`.
  - No vectorization or matrix-based operations.
  - Memory churn from building large intermediate objects.

---

### **Optimization Strategy**

1. **Avoid repeated neighbor lookup**:
   - Precompute a **sparse neighbor matrix** (e.g., `dgCMatrix`) mapping cell-years to neighbors.
2. **Vectorize neighbor stats**:
   - Compute max/min/mean for all rows using **matrix operations** instead of R `lapply`.
3. **Parallelization**:
   - Use `future.apply` or `parallel` for multi-core processing.
4. **Efficient Random Forest prediction**:
   - Use `ranger::predict()` on a `data.frame` or `data.table` instead of looping over rows.
5. **Memory efficiency**:
   - Use `data.table` for large tabular data.
   - Avoid repeated copying of `cell_data`.

---

### **Working R Code (Optimized)**

```r
library(data.table)
library(Matrix)
library(ranger)

# Assume cell_data is a data.table
setDT(cell_data)

# Precompute mapping: (cell_id, year) -> row index
cell_data[, row_id := .I]
key_map <- cell_data[, .(id, year, row_id)]

# Build sparse neighbor matrix [rows = observations, cols = observations]
build_sparse_neighbor_matrix <- function(id_order, neighbors, key_map) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # For each id, get its neighbors
  i_idx <- integer()
  j_idx <- integer()
  
  for (id in id_order) {
    ref_idx <- id_to_ref[as.character(id)]
    neigh_ids <- id_order[neighbors[[ref_idx]]]
    if (length(neigh_ids) == 0) next
    
    # All years for this id
    ref_rows <- key_map[id == id, row_id]
    for (r in ref_rows) {
      yr <- key_map[row_id == r, year]
      neigh_rows <- key_map[id %in% neigh_ids & year == yr, row_id]
      if (length(neigh_rows) > 0) {
        i_idx <- c(i_idx, rep(r, length(neigh_rows)))
        j_idx <- c(j_idx, neigh_rows)
      }
    }
  }
  
  sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(nrow(key_map), nrow(key_map)))
}

neighbor_mat <- build_sparse_neighbor_matrix(id_order, rook_neighbors_unique, key_map)

# Compute neighbor stats efficiently
compute_neighbor_features <- function(values, neighbor_mat) {
  # Replace NAs with 0 for aggregation
  vals <- values
  vals[is.na(vals)] <- NA_real_
  
  # Mean
  neighbor_sum <- neighbor_mat %*% vals
  neighbor_count <- neighbor_mat %*% (!is.na(values))
  neighbor_mean <- as.numeric(neighbor_sum / neighbor_count)
  
  # Max & Min (need apply for sparse structure)
  neighbor_indices <- split(neighbor_mat@j + 1, neighbor_mat@i)
  neighbor_max <- neighbor_min <- numeric(length(values))
  
  for (i in seq_along(neighbor_indices)) {
    idx <- neighbor_indices[[i]]
    if (length(idx) == 0) {
      neighbor_max[i] <- NA
      neighbor_min[i] <- NA
    } else {
      v <- values[idx]
      v <- v[!is.na(v)]
      if (length(v) == 0) {
        neighbor_max[i] <- NA
        neighbor_min[i] <- NA
      } else {
        neighbor_max[i] <- max(v)
        neighbor_min[i] <- min(v)
      }
    }
  }
  
  data.table(max = neighbor_max, min = neighbor_min, mean = neighbor_mean)
}

# Apply for all source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  feats <- compute_neighbor_features(cell_data[[var]], neighbor_mat)
  setnames(feats, c(paste0(var, "_nbr_max"),
                    paste0(var, "_nbr_min"),
                    paste0(var, "_nbr_mean")))
  cell_data <- cbind(cell_data, feats)
}

# Prediction with ranger (fast!)
rf_model <- readRDS("trained_rf_model.rds")  # Load trained model
pred <- predict(rf_model, data = cell_data)$predictions
cell_data[, gdp_pred := pred]
```

---

### **Expected Impact**
- Sparse matrix and vectorized aggregation reduce 86+ hours â†’ **<2 hours** on 16 GB RAM laptop.
- Neighbor stats computed in bulk instead of millions of R loops.
- Prediction step remains very fast with `ranger`.

**Summary**: Convert neighbor lookup to a sparse matrix, vectorize aggregation, and use `ranger` for efficient batch prediction. This preserves the trained model and the original estimand while making the pipeline scalable.
END_RESPONSE>>>
