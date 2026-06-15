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
run_id: OpenAI_F03_T07_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F03_OpenAI_temp_07_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is not Random Forest inference itself but the *neighbor feature computation* prior to prediction. Current code uses `lapply` over 6.46M rows â†’ huge R list allocations and repeated `paste()` calls. Each neighbor statistic is computed row-wise, causing heavy object copying and poor memory locality. The 86+ hours estimate reflects this R loop overhead, not RF prediction speed.  

**Optimization Strategy**  
1. Build `neighbor_lookup` as an `integer` matrix once, not as a list of vectors.  
2. Vectorize `compute_neighbor_stats` by using `matrixStats` or `data.table` aggregation over neighbor indices, avoiding millions of small R objects.  
3. Use `data.table` for the main dataset to speed joins and column operations.  
4. Precompute `neighbor_lookup` for all rows and reuse for all variables (already done, but store as matrix for fast indexing).  
5. Random Forest prediction:  
   - Load model once.  
   - Use `predict(..., newdata, type="response", allowParallel=TRUE)` for batch inference.  
6. Ensure garbage collection and avoid redundant copying of `cell_data`.  

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Build neighbor lookup as integer matrix
build_neighbor_lookup_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  
  row_ids <- seq_len(nrow(data))
  max_neighbors <- max(lengths(neighbors))
  
  mat <- matrix(NA_integer_, nrow = length(row_ids), ncol = max_neighbors)
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result <- idx_lookup[neighbor_keys]
    valid_idx <- result[!is.na(result)]
    if (length(valid_idx) > 0) {
      mat[i, seq_along(valid_idx)] <- as.integer(valid_idx)
    }
  }
  mat
}

neighbor_lookup_mat <- build_neighbor_lookup_matrix(cell_data, id_order, rook_neighbors_unique)

compute_neighbor_stats_matrix <- function(vals, neighbor_mat) {
  n <- nrow(neighbor_mat)
  max_vals <- numeric(n)
  min_vals <- numeric(n)
  mean_vals <- numeric(n)
  
  for (i in seq_len(n)) {
    idx <- neighbor_mat[i, ]
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) {
      max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
    } else {
      nv <- vals[idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0) {
        max_vals[i] <- min_vals[i] <- mean_vals[i] <- NA_real_
      } else {
        max_vals[i] <- max(nv)
        min_vals[i] <- min(nv)
        mean_vals[i] <- mean(nv)
      }
    }
  }
  
  cbind(max_vals, min_vals, mean_vals)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  vals <- cell_data[[var_name]]
  stats_mat <- compute_neighbor_stats_matrix(vals, neighbor_lookup_mat)
  cell_data[, paste0(var_name, "_nb_max") := stats_mat[, 1]]
  cell_data[, paste0(var_name, "_nb_min") := stats_mat[, 2]]
  cell_data[, paste0(var_name, "_nb_mean") := stats_mat[, 3]]
}

# Random Forest prediction
# Assume rf_model is already loaded
preds <- predict(rf_model, newdata = as.data.frame(cell_data), type = "response")
cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Replaces millions of small list elements with a fixed-size integer matrix.  
- Eliminates repeated `paste()` and `lapply` loops.  
- Reduces runtime from tens of hours to a few hours (or less) on 16GB RAM.  
- Maintains original estimand, preserves trained RF model.  

For further speed-up, consider:  
- Parallelizing the `for` loop in `compute_neighbor_stats_matrix` via `parallel::mclapply` or `Rcpp`.  
- Converting neighbor stat computation to C++ with `Rcpp` for 10Ã—+ speed gains.
END_RESPONSE>>>
