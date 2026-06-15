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
run_id: OpenAI_F03_T03_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows** in `build_neighbor_lookup` and `compute_neighbor_stats`. These create millions of small vectors and heavy object copying.  
2. **Neighbor feature computation** is done in R loops rather than vectorized or matrix-based operations.  
3. **Random Forest inference** is likely slowed by repeated model loading and prediction on small chunks instead of batch prediction.  
4. **Memory overhead** from large lists and repeated `do.call(rbind, â€¦)` calls.  

---

**Optimization Strategy**  
- **Precompute neighbor index matrix once** as an integer matrix (rows = cells, cols = neighbors) to avoid millions of list elements.  
- **Use `data.table` or `matrixStats`** for fast aggregation (max, min, mean) across neighbors.  
- **Vectorize neighbor feature computation**: operate on entire columns at once instead of looping per row.  
- **Batch Random Forest predictions**: load model once, predict on large chunks or full dataset.  
- **Avoid repeated object copying**: modify in place with `data.table`.  

---

**Optimized R Code**  

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor index matrix (cells x max_neighbors)
build_neighbor_matrix <- function(id_order, neighbors) {
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

neighbor_mat <- build_neighbor_matrix(id_order, rook_neighbors_unique)

# Map id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Compute neighbor stats for all years efficiently
compute_neighbor_features <- function(dt, var_name, neighbor_mat, id_to_idx) {
  vals <- dt[[var_name]]
  n_cells <- length(id_order)
  n_years <- length(unique(dt$year))
  
  # Reshape data to matrix: rows = cells, cols = years
  val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  year_levels <- sort(unique(dt$year))
  for (i in seq_along(year_levels)) {
    yr <- year_levels[i]
    idx <- dt$year == yr
    val_mat[id_to_idx[as.character(dt$id[idx])], i] <- vals[idx]
  }
  
  # Compute neighbor stats per cell-year
  max_mat <- min_mat <- mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbor_mat[i, ]
    nb_idx <- nb_idx[!is.na(nb_idx)]
    if (length(nb_idx) > 0) {
      nb_vals <- val_mat[nb_idx, , drop = FALSE]
      max_mat[i, ]  <- colMaxs(nb_vals, na.rm = TRUE)
      min_mat[i, ]  <- colMins(nb_vals, na.rm = TRUE)
      mean_mat[i, ] <- colMeans2(nb_vals, na.rm = TRUE)
    }
  }
  
  # Melt back to long and join
  res <- data.table(
    id = rep(id_order, times = n_years),
    year = rep(year_levels, each = n_cells),
    paste0(var_name, "_nb_max") = as.vector(max_mat),
    paste0(var_name, "_nb_min") = as.vector(min_mat),
    paste0(var_name, "_nb_mean") = as.vector(mean_mat)
  )
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features and merge
neighbor_features <- NULL
for (var_name in neighbor_source_vars) {
  feat <- compute_neighbor_features(cell_data, var_name, neighbor_mat, id_to_idx)
  neighbor_features <- if (is.null(neighbor_features)) feat else merge(neighbor_features, feat, by = c("id", "year"))
}

# Merge with original data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"))

# Random Forest prediction
# Load model once
rf_model <- readRDS("trained_rf_model.rds")

# Predict in large batches
batch_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)
for (i in seq(1, n, by = batch_size)) {
  idx <- i:min(i + batch_size - 1, n)
  preds[idx] <- predict(rf_model, newdata = cell_data[idx])
}

cell_data[, gdp_pred := preds]
```

---

**Expected Gains**  
- Eliminates millions of small `lapply` calls â†’ replaced with matrix operations.  
- Reduces memory overhead by using dense matrices and `data.table`.  
- Batch prediction avoids repeated model loading.  
- Should cut runtime from 86+ hours to a few hours (depending on I/O and CPU).
END_RESPONSE>>>
