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
run_id: OpenAI_F03_T03_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is in the neighbor feature computation and repeated Random Forest predictions. Issues include:  
- **Inefficient neighbor lookup**: `lapply` over 6.46M rows with repeated string concatenation and list indexing is extremely slow and memory-heavy.  
- **Repeated object copying**: Each `compute_and_add_neighbor_features` call copies `cell_data`.  
- **Prediction loop inefficiency**: Random Forest inference on millions of rows in small batches or per-row loops is very costly.  
- **Memory pressure**: Large intermediate lists and rbind operations for 6.46M rows consume RAM.  

**Optimization Strategy**  
1. **Vectorize neighbor feature computation**: Precompute neighbor indices as an integer matrix and compute stats using fast operations.  
2. **Avoid repeated string concatenation**: Use integer mapping rather than keys.  
3. **Use `data.table` or `matrix` for fast column operations**.  
4. **Batch Random Forest predictions**: Use `predict()` on large chunks or the entire dataset if RAM allows.  
5. **Preallocate results** instead of repeated `rbind`.  
6. **Parallelize** neighbor stats computation using `parallel` or `future.apply`.  

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)
library(randomForest)

# Convert to data.table for efficiency
cell_dt <- as.data.table(cell_data)

# Precompute lookup as integer matrix
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  max_neighbors <- max(lengths(neighbors))
  n_ids <- length(id_order)
  lookup_mat <- matrix(NA_integer_, nrow = n_ids, ncol = max_neighbors)
  for (i in seq_len(n_ids)) {
    nb <- neighbors[[i]]
    if (length(nb) > 0) {
      lookup_mat[i, seq_along(nb)] <- nb
    }
  }
  lookup_mat
}

neighbor_lookup_mat <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# Map cell ids to row indices by (id, year)
id_to_idx <- cell_dt[, .I, by = .(id, year)]
id_map <- setNames(id_to_idx$I, paste(id_to_idx$id, id_to_idx$year, sep = "_"))

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(cell_dt, neighbor_lookup_mat, var_name, id_order) {
  vals <- cell_dt[[var_name]]
  n_rows <- nrow(cell_dt)
  result <- matrix(NA_real_, nrow = n_rows, ncol = 3)
  
  # Parallel processing by chunks
  cl <- makeCluster(detectCores() - 1)
  clusterExport(cl, c("vals", "neighbor_lookup_mat", "id_order", "cell_dt", "var_name"), envir = environment())
  
  chunk_fun <- function(rows) {
    out_chunk <- matrix(NA_real_, nrow = length(rows), ncol = 3)
    for (j in seq_along(rows)) {
      i <- rows[j]
      ref_idx <- match(cell_dt$id[i], id_order)
      nb_ids <- neighbor_lookup_mat[ref_idx, ]
      nb_ids <- nb_ids[!is.na(nb_ids)]
      if (length(nb_ids) == 0) next
      # Map neighbor ids to same year
      nb_keys <- paste(id_order[nb_ids], cell_dt$year[i], sep = "_")
      nb_idx <- id_map[nb_keys]
      nb_idx <- nb_idx[!is.na(nb_idx)]
      if (length(nb_idx) == 0) next
      nb_vals <- vals[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0) next
      out_chunk[j, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
    }
    out_chunk
  }
  
  chunks <- split(seq_len(n_rows), ceiling(seq_len(n_rows) / 50000))
  result_list <- parLapply(cl, chunks, chunk_fun)
  stopCluster(cl)
  
  result <- do.call(rbind, result_list)
  result
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_dt, neighbor_lookup_mat, var_name, id_order)
  colnames(stats_mat) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  cell_dt <- cbind(cell_dt, stats_mat)
}

# Random Forest prediction in large batch
# Assume rf_model is loaded
preds <- predict(rf_model, newdata = cell_dt)
cell_dt[, gdp_pred := preds]
```

---

### **Expected Gains**
- **Neighbor feature computation**: From hours to minutes using parallelization and matrix-based lookup.
- **Random Forest inference**: Single batch prediction avoids repeated overhead.
- **Memory efficiency**: Preallocated matrices and `data.table` reduce copying.

This preserves the trained Random Forest model and original estimand while making the pipeline feasible on a 16 GB machine.
END_RESPONSE>>>
