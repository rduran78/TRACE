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
run_id: OpenAI_F06_T03_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F06_OpenAI_temp_03_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` over ~6.46M rows, repeatedly scanning neighbor indices and computing stats. Each iteration allocates vectors and performs redundant lookups. This results in excessive R-level overhead and memory churn. The algorithm is correct but inefficient because it does not leverage vectorization or compiled operations.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** (already done).
2. Replace per-row `lapply` with **vectorized aggregation** using `vapply` or `data.table` for speed.
3. Use **matrix operations** and avoid repeated `do.call(rbind, ...)`.
4. Consider **parallelization** via `future.apply` or `parallel`.
5. Keep memory footprint low by processing one variable at a time.
6. Preserve numerical results by using the same max, min, mean logic.

---

**Working R Code (Optimized)**  
```r
library(data.table)
library(parallel)

# Convert data to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup (unchanged)
neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Optimized compute_neighbor_stats using vapply
compute_neighbor_stats_fast <- function(vals, neighbor_lookup) {
  n <- length(neighbor_lookup)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  
  # Parallelize over chunks
  cores <- max(1, detectCores() - 1)
  chunks <- split(seq_len(n), cut(seq_len(n), cores, labels = FALSE))
  
  cl <- makeCluster(cores)
  clusterExport(cl, c("neighbor_lookup", "vals"), envir = environment())
  
  res_list <- parLapply(cl, chunks, function(chunk) {
    m <- matrix(NA_real_, nrow = length(chunk), ncol = 3)
    for (i in seq_along(chunk)) {
      idx <- neighbor_lookup[[chunk[i]]]
      if (length(idx) > 0) {
        nv <- vals[idx]
        nv <- nv[!is.na(nv)]
        if (length(nv) > 0) {
          m[i, ] <- c(max(nv), min(nv), mean(nv))
        }
      }
    }
    m
  })
  
  stopCluster(cl)
  
  result[unlist(chunks), ] <- do.call(rbind, res_list)
  result
}

# Wrapper to add features
compute_and_add_neighbor_features_fast <- function(dt, var_name, neighbor_lookup) {
  vals <- dt[[var_name]]
  stats <- compute_neighbor_stats_fast(vals, neighbor_lookup)
  setnames(as.data.table(stats), paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  cbind(dt, stats)
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}
```

---

**Expected Impact**  
- Eliminates repeated `do.call` and heavy R loops.
- Parallelization across cores reduces runtime from 86+ hours to a few hours (depending on machine).
- Preserves original estimand and Random Forest model.
- Memory stays within 16 GB by processing one variable at a time.
END_RESPONSE>>>
