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
run_id: OpenAI_F03_T03_R002
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 2
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive neighbor computations**: `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of rows, creating large intermediate lists and repeated object copying.  
2. **Inefficient memory usage**: Storing neighbor indices as lists of varying lengths and repeatedly binding results (`do.call(rbind, ...)`) is costly.  
3. **Prediction loop inefficiency**: Likely calling `predict()` on small chunks or per-row instead of vectorized batches.  
4. **Model loading**: If the Random Forest model is reloaded repeatedly, that adds overhead.  

**Optimization Strategy**  
- **Precompute neighbor indices once** as an integer matrix or compressed representation.  
- **Vectorize neighbor feature computation** using `data.table` or `matrixStats` instead of `lapply`.  
- **Avoid repeated rbind** by preallocating numeric matrices.  
- **Batch predictions**: Call `predict()` on the full feature matrix or large chunks.  
- **Use data.table for joins and updates** to minimize copying.  
- **Keep model in memory** and reuse it.  

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute lookup as integer matrix
build_neighbor_lookup_matrix <- function(id_order, neighbors) {
  max_n <- max(lengths(neighbors))
  mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_n)
  for (i in seq_along(neighbors)) {
    nbs <- neighbors[[i]]
    if (length(nbs) > 0) {
      mat[i, seq_along(nbs)] <- nbs
    }
  }
  mat
}

neighbor_mat <- build_neighbor_lookup_matrix(id_order, rook_neighbors_unique)

# Map id -> row index for quick lookup
idx_lookup <- setNames(seq_len(nrow(cell_data)), paste(cell_data$id, cell_data$year, sep = "_"))
id_to_ref  <- setNames(seq_along(id_order), as.character(id_order))

# Compute neighbor features efficiently
compute_neighbor_stats_fast <- function(dt, neighbor_mat, var_name) {
  vals <- dt[[var_name]]
  n <- nrow(dt)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- paste(var_name, c("max", "min", "mean"), sep = "_")

  # Precompute key lookup
  keys <- paste(dt$id, dt$year, sep = "_")

  for (i in seq_len(n)) {
    ref_idx <- id_to_ref[[as.character(dt$id[i])]]
    nb_ids <- neighbor_mat[ref_idx, ]
    nb_ids <- nb_ids[!is.na(nb_ids)]
    if (length(nb_ids) == 0) next
    nb_keys <- paste(id_order[nb_ids], dt$year[i], sep = "_")
    nb_idx <- idx_lookup[nb_keys]
    nb_idx <- nb_idx[!is.na(nb_idx)]
    if (length(nb_idx) == 0) next
    nb_vals <- vals[nb_idx]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    out[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }
  out
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  cell_data <- cbind(cell_data, stats_mat)
}

# Prepare feature matrix for prediction
feature_cols <- setdiff(names(cell_data), c("id", "year", "target_var")) # adjust target name
X <- as.matrix(cell_data[, ..feature_cols])

# Load trained RF model once
rf_model <- readRDS("trained_rf_model.rds")

# Batch prediction
pred <- predict(rf_model, X)

cell_data[, prediction := pred]
```

---

**Key Gains**  
- Eliminates repeated `lapply` and `rbind` overhead.  
- Uses preallocated matrices for neighbor stats.  
- Predicts in one batch instead of row-wise.  
- Expected runtime reduction from **86+ hours to a few hours**, depending on I/O and CPU.  

**Further improvements**:  
- Parallelize neighbor feature computation with `parallel::mclapply` or `future.apply`.  
- Persist neighbor lookup matrix to disk for reuse.  
- If memory is tight, process in chunks (e.g., yearly batches) but keep batch size large for prediction efficiency.
END_RESPONSE>>>
