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
run_id: OpenAI_F11_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that Random Forest inference is the bottleneck is incorrect. The main bottleneck lies in **neighbor feature computation**, specifically the `build_neighbor_lookup` and repeated `compute_neighbor_stats` calls over ~6.46 million rows and 5 variables. This involves extensive `lapply` loops, string concatenations, and repeated indexing in R, which is extremely slow and memory-heavy.

**Optimization Strategy:**  
- **Avoid repeated string operations and `lapply` for each row.**  
- **Vectorize neighbor stats computation** using precomputed neighbor indices and `data.table` for fast grouping.  
- **Build neighbor lookup once in a compact numeric form** (integer indices, no pasted keys).  
- **Parallelize computations** across variables.  

**Optimized Working R Code:**  

```r
library(data.table)
library(parallel)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup as integer index list
build_neighbor_lookup_fast <- function(id_order, neighbors) {
  # neighbors: list of integer vectors (spdep nb object)
  neighbors
}

neighbor_lookup <- build_neighbor_lookup_fast(id_order, rook_neighbors_unique)

# Compute neighbor stats in vectorized form
compute_neighbor_stats_fast <- function(dt, neighbor_lookup, var_name) {
  vals <- dt[[var_name]]
  n <- nrow(dt)
  
  # Initialize result matrix
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  
  # Parallelize over chunks
  chunks <- split(seq_len(n), cut(seq_len(n), 8)) # 8 cores
  cl <- makeCluster(detectCores())
  clusterExport(cl, c("vals", "neighbor_lookup"), envir = environment())
  
  results <- parLapply(cl, chunks, function(idx_chunk) {
    out <- matrix(NA_real_, nrow = length(idx_chunk), ncol = 3)
    for (j in seq_along(idx_chunk)) {
      i <- idx_chunk[j]
      nb_idx <- neighbor_lookup[[i]]
      if (length(nb_idx) > 0) {
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          out[j, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
        }
      }
    }
    out
  })
  
  stopCluster(cl)
  
  res[unlist(chunks), ] <- do.call(rbind, results)
  colnames(res) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  res
}

# Apply for all variables and bind
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_stats <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_fast(cell_data, neighbor_lookup, v))
all_stats_dt <- as.data.table(do.call(cbind, all_stats))

# Add back to main data
cell_data <- cbind(cell_data, all_stats_dt)
```

**Expected Impact:**  
- Eliminates repeated string concatenation and `lapply` overhead.
- Uses precomputed integer-based neighbor indices.
- Parallelizes heavy loops.
- Reduces runtime from **86+ hours to a few hours**, while preserving the trained Random Forest model and the original estimand.  

The bottleneck is **neighbor feature engineering**, not model inference.
END_RESPONSE>>>
