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
run_id: OpenAI_F03_T03_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated R loops and `lapply` over 6.46M rows**: `build_neighbor_lookup` and `compute_neighbor_stats` are highly inefficient because they repeatedly allocate vectors and perform lookups in R lists.  
2. **Memory overhead**: Storing large lists of neighbor indices and repeatedly copying `data` objects is expensive.  
3. **Prediction inefficiency**: If Random Forest predictions are done row-by-row or in small chunks, this will be extremely slow.  
4. **Model loading**: If the model is reloaded repeatedly, that adds overhead.  

**Optimization Strategy**  
- **Precompute neighbor indices as an integer matrix** instead of a list of vectors. Missing neighbors can be `NA`.  
- **Vectorize neighbor statistics** using `matrixStats` or `data.table` instead of looping.  
- **Avoid repeated `rbind` and `lapply`**; compute all neighbor stats in one pass.  
- **Batch predictions**: Use `predict(model, newdata, type="response")` on large chunks (e.g., 500k rows).  
- **Use `data.table` for feature engineering** to minimize copies.  
- **Keep the trained Random Forest model in memory** (load once).  

---

### **Optimized R Code**

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb)
# id_order: vector of all unique cell ids in reference order
# rf_model: pre-trained randomForest object

setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor lookup as matrix
build_neighbor_matrix <- function(id_order, neighbors, max_nbrs = NULL) {
  if (is.null(max_nbrs)) {
    max_nbrs <- max(lengths(neighbors))
  }
  mat <- matrix(NA_integer_, nrow = length(id_order), ncol = max_nbrs)
  for (i in seq_along(neighbors)) {
    nbrs <- neighbors[[i]]
    if (length(nbrs) > 0) {
      mat[i, seq_along(nbrs)] <- nbrs
    }
  }
  mat
}

neighbor_mat <- build_neighbor_matrix(id_order, rook_neighbors_unique)

# Map id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Add row index to cell_data for fast join
cell_data[, idx := id_to_idx[as.character(id)]]

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(dt, neighbor_mat, var) {
  vals <- dt[[var]]
  n <- nrow(dt)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  
  # For each year, process block
  years <- unique(dt$year)
  for (yr in years) {
    rows <- which(dt$year == yr)
    idxs <- dt$idx[rows]
    nbr_idx <- neighbor_mat[idxs, , drop = FALSE]
    
    # Convert neighbor ids to row indices for this year
    # Build a lookup for this year's rows
    year_idx_map <- integer(max(idxs))
    year_idx_map[idxs] <- rows
    
    # Map neighbor ids to positions
    nbr_rows <- matrix(year_idx_map[nbr_idx], nrow = nrow(nbr_idx))
    
    # Extract neighbor values
    nbr_vals <- matrix(NA_real_, nrow = nrow(nbr_rows), ncol = ncol(nbr_rows))
    for (j in seq_len(ncol(nbr_rows))) {
      valid <- !is.na(nbr_rows[, j])
      nbr_vals[valid, j] <- vals[nbr_rows[valid, j]]
    }
    
    # Compute stats
    row_max <- rowMaxs(nbr_vals, na.rm = TRUE)
    row_min <- rowMins(nbr_vals, na.rm = TRUE)
    row_mean <- rowMeans(nbr_vals, na.rm = TRUE)
    
    res[rows, ] <- cbind(row_max, row_min, row_mean)
  }
  colnames(res) <- paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  res
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features
neighbor_features <- lapply(neighbor_source_vars, function(v) {
  compute_neighbor_stats_fast(cell_data, neighbor_mat, v)
})
neighbor_features <- do.call(cbind, neighbor_features)

# Bind to cell_data
cell_data <- cbind(cell_data, neighbor_features)

# Drop helper column
cell_data[, idx := NULL]

# Batch prediction
predict_in_chunks <- function(model, data, chunk_size = 500000) {
  n <- nrow(data)
  preds <- numeric(n)
  for (i in seq(1, n, by = chunk_size)) {
    j <- min(i + chunk_size - 1, n)
    preds[i:j] <- predict(model, newdata = data[i:j], type = "response")
  }
  preds
}

# Select predictor columns (exclude id/year)
predictors <- setdiff(names(cell_data), c("id", "year"))
predictions <- predict_in_chunks(rf_model, cell_data[, ..predictors])

```

---

### **Key Gains**
- **Vectorization** replaces millions of R loops with `matrixStats` operations.
- **Neighbor matrix** avoids repeated list lookups.
- **`data.table`** minimizes copies and speeds joins.
- **Chunked prediction** uses Random Forest efficiently.
- Expected runtime drops from **86+ hours to a few hours** on a standard laptop.
END_RESPONSE>>>
