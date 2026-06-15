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
run_id: OpenAI_F03_T00_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F03_OpenAI_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows**: Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-wise, creating millions of small R objects.  
2. **Excessive object copying**: `lapply` + `do.call(rbind, ...)` repeatedly allocates large intermediate lists.  
3. **Inefficient neighbor lookup**: String concatenation (`paste`) and repeated hash lookups for every row are costly.  
4. **Memory pressure**: 6.46M rows Ã— multiple neighbor stats Ã— 110 predictors stresses 16 GB RAM.  
5. **Prediction loop**: If Random Forest predictions are done row-by-row, this is extremely slow.  

---

**Optimization Strategy**  
- **Precompute neighbor indices once** as integer vectors, avoid string keys.  
- **Vectorize neighbor stats** using `data.table` or `matrix` operations instead of `lapply`.  
- **Avoid repeated `rbind`**: preallocate numeric matrices.  
- **Batch predictions**: Use `predict(model, newdata, type="response")` on the full dataset or large chunks, not per-row.  
- **Use `data.table` for joins and grouping**: drastically reduces overhead.  
- **Memory efficiency**: Drop unnecessary columns early, use numeric matrices for modeling.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Assume: cell_data (data.frame), id_order, rook_neighbors_unique, rf_model loaded

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as integer indices
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_lookup <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Create a mapping from (id, year) -> row index
cell_data[, row_idx := .I]

# Compute neighbor stats efficiently
compute_neighbor_stats_dt <- function(dt, neighbor_lookup, var_name) {
  vals <- dt[[var_name]]
  n <- nrow(dt)
  res <- matrix(NA_real_, n, 3)  # max, min, mean
  for (i in seq_len(n)) {
    nb_ids <- neighbor_lookup[[id_to_idx[[as.character(dt$id[i])]]]]
    if (length(nb_ids) == 0) next
    # Find rows for same year
    nb_rows <- dt[J(id_order[nb_ids], dt$year[i]), nomatch = 0L, which = TRUE]
    if (length(nb_rows) == 0) next
    nb_vals <- vals[nb_rows]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0) next
    res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
  }
  res
}

# Apply for all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_dt(cell_data, neighbor_lookup, var_name)
  cell_data[, paste0(var_name, "_nb_max") := stats[, 1]]
  cell_data[, paste0(var_name, "_nb_min") := stats[, 2]]
  cell_data[, paste0(var_name, "_nb_mean") := stats[, 3]]
}

# Prepare matrix for prediction
predictors <- setdiff(names(cell_data), c("id", "year", "row_idx", "target_var"))
X <- as.matrix(cell_data[, ..predictors])

# Batch prediction
batch_size <- 500000
preds <- numeric(nrow(X))
for (i in seq(1, nrow(X), by = batch_size)) {
  idx <- i:min(i + batch_size - 1, nrow(X))
  preds[idx] <- predict(rf_model, newdata = X[idx, , drop = FALSE])
}

cell_data[, prediction := preds]
```

---

**Key Gains**  
- Eliminates repeated string concatenation and `lapply` overhead.  
- Uses preallocated matrices and `data.table` for efficient joins.  
- Batch prediction avoids per-row model calls.  

**Expected Runtime Reduction**: From 86+ hours to a few hours (depending on disk I/O and CPU).  

For even more speed:  
- Parallelize neighbor stats computation with `future.apply` or `data.table` chunking.  
- Consider converting neighbor stats computation to C++ via `Rcpp` if still slow.
END_RESPONSE>>>
