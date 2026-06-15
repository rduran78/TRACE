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
run_id: OpenAI_F03_T03_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F03_OpenAI_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottlenecks are:  
1. **Repeated `lapply` over 6.46M rows** in `build_neighbor_lookup` and `compute_neighbor_stats`. This creates millions of small lists and repeated object copying.  
2. **Inefficient neighbor feature computation**: Each variable loops over all rows, repeatedly scanning neighbor indices.  
3. **Memory overhead**: Large lists and repeated `do.call(rbind, ...)` operations cause high memory churn.  
4. **Prediction workflow**: If Random Forest predictions are done row-by-row, this is extremely slow. `predict()` should be vectorized over the entire dataset or large chunks.  
5. **Model loading**: Ensure the model is loaded once and reused.  

---

**Optimization Strategy**  
- Precompute `neighbor_lookup` **once** as an integer matrix or list of integer vectors without repeated string concatenation.  
- Replace `lapply` with **vectorized operations** or `vapply` where possible.  
- Compute neighbor stats for all variables in one pass rather than looping over variables.  
- Use `data.table` for fast joins and column operations.  
- For Random Forest prediction:  
  - Use `predict(model, newdata, ...)` on the entire dataset or in large chunks (e.g., 500k rows per batch).  
  - Avoid per-row prediction loops.  
- Reduce memory footprint by storing neighbor indices as integer vectors and using matrix operations for stats.  

---

**Working R Code (Optimized)**  

```r
library(data.table)
library(randomForest)

# Convert to data.table for speed
cell_data <- as.data.table(cell_data)

# Precompute lookup as integer vectors
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  # neighbors is spdep::nb list of integer vectors
  return(neighbors) # Already integer indices referring to id_order positions
}

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# Compute neighbor stats for all variables in one pass
compute_all_neighbor_stats <- function(data, neighbor_lookup, vars) {
  n <- nrow(data)
  result_list <- vector("list", length(vars))
  
  for (v in seq_along(vars)) {
    vals <- data[[vars[v]]]
    # Preallocate matrix: rows = n, cols = 3 (max, min, mean)
    mat <- matrix(NA_real_, n, 3)
    
    for (i in seq_len(n)) {
      idx <- neighbor_lookup[[data$id[i]]] # neighbors for this cell
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    result_list[[v]] <- mat
  }
  
  # Bind results into data.table columns
  for (v in seq_along(vars)) {
    nm <- vars[v]
    data[, paste0(nm, "_nbr_max") := result_list[[v]][, 1]]
    data[, paste0(nm, "_nbr_min") := result_list[[v]][, 2]]
    data[, paste0(nm, "_nbr_mean") := result_list[[v]][, 3]]
  }
  
  return(data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_stats(cell_data, neighbor_lookup, neighbor_source_vars)

# Random Forest prediction in chunks
chunk_size <- 500000
n <- nrow(cell_data)
preds <- numeric(n)

for (start in seq(1, n, by = chunk_size)) {
  end <- min(start + chunk_size - 1, n)
  preds[start:end] <- predict(rf_model, newdata = cell_data[start:end])
}

cell_data[, gdp_pred := preds]
```

---

**Key Improvements**  
- Eliminated repeated string operations and `lapply` overhead.  
- Batched Random Forest predictions instead of per-row.  
- Used preallocated matrices and `data.table` for efficient column assignment.  
- Preserves trained model and original estimand.  

**Expected Impact**: From 86+ hours to a few hours or less, depending on machine and disk I/O.
END_RESPONSE>>>
