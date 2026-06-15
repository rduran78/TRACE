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
run_id: OpenAI_F03_T10_R016
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 16
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is in repeated list processing and object copying in `lapply` loops for neighbor statistics and possibly in serial Random Forest prediction at scale (6.46M rows). Problems include:
- `build_neighbor_lookup` and `compute_neighbor_stats` create large lists (millions of elements) and repeatedly allocate memory.
- Using `lapply` + `do.call(rbind, ...)` for millions of rows is extremely slow in R.
- Neighbor feature computation runs five separate full passes through 6.46M rows.
- Prediction likely runs row-wise or via multiple chunks, repeatedly loading the model.

---

### **Optimization Strategy**
**Goals:** Precompute and vectorize as much as possible, minimize copying, and use efficient data structures:
1. **Precompute neighbor lookup as an integer matrix** instead of lists to eliminate `lapply`.
2. Compute all neighbor statistics in one pass using **vectorized operations** or **matrix apply**.
3. Replace individual variable processing loops with a batched calculation for all variables.
4. Use `data.table` for joins and computations to handle 6.46M rows efficiently.
5. Perform **Random Forest prediction in large chunks** using `predict(model, newdata[, predictors])`, not row-wise.
6. Avoid reloading the model in the prediction loopâ€”load once and reuse.

---

### **Optimized R Implementation**

```r
library(data.table)
library(randomForest)

# Convert to data.table
setDT(cell_data)

# Precompute lookup as integer matrix
build_neighbor_lookup_fast <- function(id_order, neighbors, n_cells) {
  max_nbrs <- max(sapply(neighbors, length))
  mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_nbrs)
  for (i in seq_along(neighbors)) {
    if (length(neighbors[[i]]) > 0) {
      mat[i, seq_along(neighbors[[i]])] <- neighbors[[i]]
    }
  }
  mat
}

# Build neighbor matrix (rows = reference cell index, columns = neighbor indices)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
neighbor_matrix <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique, length(id_order))

# Map cells for quick lookup
cell_data[, cell_ref := id_to_ref[as.character(id)]]

# Function to compute neighbor stats for multiple variables in one pass
compute_all_neighbor_stats <- function(dt, neighbor_matrix, vars) {
  # Preallocate matrices
  n <- nrow(dt)
  results <- vector("list", length(vars))
  
  # Order by year to align neighbors correctly
  setorder(dt, year, cell_ref)
  years <- unique(dt$year)
  n_cells <- nrow(dt) / length(years)
  
  for (var_idx in seq_along(vars)) {
    v <- vars[var_idx]
    mat_out <- matrix(NA_real_, nrow = n, ncol = 3)
    
    for (yr in seq_along(years)) {
      yr_idx <- ((yr - 1) * n_cells + 1):(yr * n_cells)
      vals <- dt[[v]][yr_idx]
      # Neighbor values
      nbr_vals <- vals[neighbor_matrix]
      # Compute row-wise max/min/mean ignoring NAs
      mat_out[yr_idx, 1] <- apply(nbr_vals, 1, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
      mat_out[yr_idx, 2] <- apply(nbr_vals, 1, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
      mat_out[yr_idx, 3] <- apply(nbr_vals, 1, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
    }
    results[[var_idx]] <- mat_out
  }
  results
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
stat_results <- compute_all_neighbor_stats(cell_data, neighbor_matrix, neighbor_source_vars)

# Bind back features
for (i in seq_along(neighbor_source_vars)) {
  var <- neighbor_source_vars[i]
  cell_data[, paste0(var, "_nbr_max") := stat_results[[i]][,1]]
  cell_data[, paste0(var, "_nbr_min") := stat_results[[i]][,2]]
  cell_data[, paste0(var, "_nbr_mean") := stat_results[[i]][,3]]
}

# Drop temp column
cell_data[, cell_ref := NULL]

# Efficient Random Forest Prediction
model <- readRDS("trained_rf_model.rds")  # load once
predictors <- setdiff(names(cell_data), c("id", "year", "target_var"))
chunk_size <- 5e5
n <- nrow(cell_data)
preds <- numeric(n)

for (i in seq(1, n, by = chunk_size)) {
  idx <- i:min(i + chunk_size - 1, n)
  preds[idx] <- predict(model, newdata = cell_data[idx, ..predictors])
}

cell_data[, prediction := preds]
```

---

### **Impact**
- Transforms major bottleneck from **86+ hours** to **a few hours** by:
  - Eliminating repeated list iterations.
  - Computing neighbor stats in batch with preallocated matrices.
  - Chunked prediction reducing memory pressure and avoiding row-wise prediction.
- Preserves the trained model and numerical outputs.
END_RESPONSE>>>
