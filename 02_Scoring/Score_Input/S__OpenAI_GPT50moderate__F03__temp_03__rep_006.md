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
run_id: OpenAI_F03_T03_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Inefficient neighbor feature computation**:  
   - `build_neighbor_lookup` returns a large list of integer vectors (â‰ˆ6.46M elements), causing huge memory overhead and repeated list traversals.  
   - `compute_neighbor_stats` uses `lapply` + `do.call(rbind, ...)`, which is slow for millions of rows.  
2. **Repeated feature computation**: For each of 5 variables, the entire neighbor loop is recomputed.  
3. **Prediction inefficiency**:  
   - Random Forest inference on 6.46M rows with 110+ features is heavy.  
   - If predictions are done in small chunks or with repeated model loading, it adds overhead.  
4. **Memory pressure**: Copying large data frames repeatedly (e.g., `cell_data <- ...`) is expensive.  

---

### **Optimization Strategy**
- **Precompute neighbor indices once** as an integer matrix for all rows.  
- **Vectorize neighbor stats**: Avoid `lapply` for each row; use `matrixStats` or `data.table` for aggregation.  
- **Use `data.table`** for in-place updates to avoid copies.  
- **Batch predictions**: Use `predict(..., newdata, type="response")` in large chunks (e.g., 500k rows) to reduce overhead.  
- **Keep model in memory**: Load once, reuse.  
- **Parallelize where possible**: Use `parallel::mclapply` or `future.apply` for neighbor stats if vectorization is insufficient.  

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute neighbor lookup as a matrix
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  n <- nrow(data)
  max_neighbors <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, nrow = n, ncol = max_neighbors)
  
  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    idx <- idx_lookup[neighbor_keys]
    idx <- as.integer(idx[!is.na(idx)])
    if (length(idx) > 0) {
      neighbor_mat[i, seq_along(idx)] <- idx
    }
  }
  neighbor_mat
}

neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(vals, neighbor_mat) {
  n <- nrow(neighbor_mat)
  max_vals <- numeric(n)
  min_vals <- numeric(n)
  mean_vals <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_mat[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) {
      max_vals[i] <- NA
      min_vals[i] <- NA
      mean_vals[i] <- NA
    } else {
      neigh_vals <- vals[idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
        mean_vals[i] <- NA
      } else {
        max_vals[i] <- max(neigh_vals)
        min_vals[i] <- min(neigh_vals)
        mean_vals[i] <- mean(neigh_vals)
      }
    }
  }
  cbind(max_vals, min_vals, mean_vals)
}

# Add neighbor features for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_mat)
  set(cell_data, j = paste0(var_name, "_nbr_max"), value = stats[, 1])
  set(cell_data, j = paste0(var_name, "_nbr_min"), value = stats[, 2])
  set(cell_data, j = paste0(var_name, "_nbr_mean"), value = stats[, 3])
}

# Batch prediction
predict_in_batches <- function(model, data, batch_size = 500000) {
  n <- nrow(data)
  preds <- numeric(n)
  for (start in seq(1, n, by = batch_size)) {
    end <- min(start + batch_size - 1, n)
    preds[start:end] <- predict(model, newdata = data[start:end])
  }
  preds
}

# Load trained model once
rf_model <- readRDS("trained_rf_model.rds")

# Perform predictions
cell_data[, prediction := predict_in_batches(rf_model, cell_data)]
```

---

### **Expected Gains**
- **Neighbor computation**: From hours to minutes (due to matrix-based lookup and in-place updates).  
- **Prediction**: Single model load + batch inference avoids overhead.  
- **Memory**: `data.table` prevents repeated copies.  

This approach should reduce runtime from **86+ hours to a few hours** on a 16 GB laptop.
END_RESPONSE>>>
