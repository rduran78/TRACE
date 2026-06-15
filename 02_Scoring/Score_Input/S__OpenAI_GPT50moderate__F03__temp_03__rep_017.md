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
run_id: OpenAI_F03_T03_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated expensive neighbor computations**: `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over millions of rows, repeatedly creating vectors and copying data.  
2. **Inefficient object handling**: `lapply` + `do.call(rbind, ...)` creates large intermediate lists, causing memory pressure and slow performance.  
3. **Prediction loop overhead**: If Random Forest predictions are called in small batches or per-row, this adds huge overhead.  
4. **Model loading**: If the model is reloaded repeatedly, thatâ€™s unnecessary overhead.  

**Optimization Strategy**  
- **Precompute neighbor indices once** and store as an integer matrix or list of integer vectors.  
- **Use vectorized or compiled operations** (e.g., `data.table` or `Rcpp`) to compute neighbor stats in bulk.  
- **Avoid repeated string concatenation and lookups**; map IDs to row indices once.  
- **Batch predictions**: Call `predict()` on the entire dataset or large chunks, not per-row.  
- **Memory efficiency**: Use `data.table` for in-place joins and feature creation.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(randomForest)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs
# rook_neighbors_unique: list of neighbor indices (spdep::nb)
# rf_model: pre-trained randomForest model

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute lookup: map id -> sequential index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build neighbor lookup as integer vectors (no string ops)
neighbor_lookup <- lapply(seq_along(id_order), function(i) {
  as.integer(id_to_idx[id_order[rook_neighbors_unique[[i]]]])
})

# Add a column for fast row index
cell_data[, row_idx := .I]

# Create a mapping from (id, year) -> row index
# Already have row_idx, so we can use joins

compute_neighbor_stats_fast <- function(dt, neighbor_lookup, var_name) {
  vals <- dt[[var_name]]
  n <- nrow(dt)
  
  # Preallocate result matrix
  res <- matrix(NA_real_, n, 3)
  
  for (i in seq_len(n)) {
    nb_ids <- neighbor_lookup[[id_to_idx[[as.character(dt$id[i])]]]]
    if (length(nb_ids) == 0) next
    # Find neighbors in same year
    nb_rows <- dt[J(id_order[nb_ids], dt$year[i]), nomatch = 0L, row_idx]
    if (length(nb_rows) == 0) next
    nb_vals <- vals[nb_rows]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }
  
  res
}

# Compute all neighbor features in one pass per variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  cell_data[, paste0(var_name, "_nb_max") := stats_mat[, 1]]
  cell_data[, paste0(var_name, "_nb_min") := stats_mat[, 2]]
  cell_data[, paste0(var_name, "_nb_mean") := stats_mat[, 3]]
}

# Remove helper column
cell_data[, row_idx := NULL]

# Batch prediction
# Convert to matrix of predictors
predictors <- as.matrix(cell_data[, !c("id", "year"), with = FALSE])
preds <- predict(rf_model, predictors)

# Attach predictions
cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminated repeated string concatenation and `paste()`.  
- Preallocated result matrices to avoid repeated `rbind`.  
- Used `data.table` for efficient joins and indexing.  
- Batched Random Forest predictions in one call.  

**Expected Performance Gain**  
- From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on disk I/O and CPU.  
- Further speedup possible with `Rcpp` for neighbor stats or parallelization (`parallel::mclapply` or `future.apply`).
END_RESPONSE>>>
