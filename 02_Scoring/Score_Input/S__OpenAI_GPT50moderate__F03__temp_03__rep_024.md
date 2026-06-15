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
run_id: OpenAI_F03_T03_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows** in `build_neighbor_lookup` and `compute_neighbor_stats`. This creates millions of small R objects, causing high memory overhead and garbage collection.  
2. **Inefficient neighbor lookup**: Using character-based keys (`paste`) and repeated name-based indexing is very slow.  
3. **Repeated copying of `cell_data`** in `compute_and_add_neighbor_features`.  
4. **Prediction loop inefficiency**: If predictions are done row-by-row or in small chunks, Random Forest inference becomes slow.  
5. **Model loading**: Ensure the model is loaded once and predictions are vectorized.  

---

**Optimization Strategy**  
- Precompute **numeric indices** for neighbors instead of character keys.  
- Replace `lapply` with **vectorized or matrix-based operations** using `data.table` or `matrixStats`.  
- Compute all neighbor stats in a **single pass** per variable.  
- Avoid repeated `cbind` or `merge`â€”use in-place assignment with `data.table`.  
- For Random Forest prediction:  
  - Use `predict(model, newdata, type="response", predict.all=FALSE)` on the full dataset or large chunks.  
  - Ensure the model is in memory only once.  

---

**Optimized R Code** (using `data.table` for speed and memory efficiency):  

```r
library(data.table)
library(matrixStats)
library(randomForest)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor index lookup as integer vectors
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  # neighbors is a list of integer vectors (indices in id_order)
  neighbors
}

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# Compute neighbor stats for all variables efficiently
compute_neighbor_stats_fast <- function(dt, neighbor_lookup, vars) {
  n <- nrow(dt)
  res_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    var_name <- vars[v]
    vals <- dt[[var_name]]
    
    # Preallocate matrix: rows = n, cols = 3 (max, min, mean)
    stats_mat <- matrix(NA_real_, n, 3)
    
    for (i in seq_len(n)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          stats_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    
    colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    res_list[[v]] <- stats_mat
  }
  
  res <- do.call(cbind, res_list)
  as.data.table(res)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats_dt <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, neighbor_source_vars)

# Bind neighbor features to main data
cell_data <- cbind(cell_data, neighbor_stats_dt)

# Random Forest prediction in large chunks
# Load model once
rf_model <- readRDS("trained_rf_model.rds")

chunk_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end], type = "response")
}

cell_data[, gdp_pred := preds]
```

---

**Expected Gains**  
- Eliminates repeated string operations and object copying.  
- Uses preallocated matrices and `data.table` for efficient memory use.  
- Vectorized Random Forest prediction reduces runtime drastically.  
- On a 16 GB machine, this should reduce runtime from **86+ hours to a few hours** (depending on RF complexity).  

Further speedups:  
- Parallelize neighbor stats computation with `parallel::mclapply`.  
- Consider `ranger` for much faster Random Forest inference if model retraining is allowed (but here it is not).
END_RESPONSE>>>
