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
run_id: OpenAI_F07_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation loops through ~6.46M rows and calls `compute_neighbor_stats` separately for each variable, performing repeated neighbor indexing and redundant calculations. This results in excessive overhead and memory usage because:

- `lapply` on millions of rows generates huge lists repeatedly.
- Stats are recomputed for each variable sequentially instead of vectorized/matrix operations.
- Neighbor lookups are constructed once but used inefficiently.
- No parallelization or memory-efficient representation (gaps filled with NA).

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** (already done).
2. **Vectorize computation across variables**: Convert neighbor value aggregation from repeated loops to matrix operations.
3. **Use `vapply` and `data.table` with keyed joins** for speed and memory efficiency.
4. **Parallelize row-based aggregation** using `future.apply` or `parallel`.
5. **Chunk processing** (avoid loading all 6.46M rows at once).
6. Preserve original rook-neighbor topology, numeric estimands, and RF model by only rewriting neighbor statistics pipeline.

---

### **Optimized Working R Code**

```r
library(data.table)
library(future.apply)

# Assume cell_data is already a data.table
# Create lookup as before
build_neighbor_lookup <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)), paste(data$id, data$year, sep = "_"))
  row_ids <- seq_len(nrow(data))
  
  lapply(row_ids, function(i) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
    result <- idx_lookup[neighbor_keys]
    as.integer(result[!is.na(result)])
  })
}

neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)

# Vectorized neighbor stats using parallel apply
compute_neighbor_stats_matrix <- function(data, neighbor_lookup, vars) {
  vals_mat <- as.matrix(data[, ..vars])
  # Preallocate result arrays: columns for max, min, mean per var
  res_array <- array(NA_real_, dim = c(nrow(data), length(vars) * 3))
  col_names <- as.vector(sapply(vars, function(v) paste(v, c("nb_max", "nb_min", "nb_mean"), sep = "_")))
  dimnames(res_array)[[2]] <- col_names
  
  plan(multisession, workers = parallel::detectCores() - 1)
  chunks <- split(seq_len(nrow(data)), ceiling(seq_len(nrow(data)) / 50000))
  
  res_list <- future_lapply(chunks, function(chunk) {
    local_res <- matrix(NA_real_, nrow = length(chunk), ncol = length(vars) * 3)
    for (k in seq_along(chunk)) {
      idx <- neighbor_lookup[[chunk[k]]]
      if (length(idx) == 0) next
      nb_vals <- vals_mat[idx, , drop = FALSE]
      nb_vals <- nb_vals[complete.cases(nb_vals), , drop = FALSE]
      if (nrow(nb_vals) == 0) next
      max_v <- apply(nb_vals, 2, max)
      min_v <- apply(nb_vals, 2, min)
      mean_v <- colMeans(nb_vals)
      local_res[k, ] <- c(max_v, min_v, mean_v)
    }
    local_res
  })
  
  res_array[unlist(chunks), ] <- do.call(rbind, res_list)
  as.data.table(res_array)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_stats <- compute_neighbor_stats_matrix(cell_data, neighbor_lookup, neighbor_source_vars)

cell_data <- cbind(cell_data, neighbor_stats)
```

---

**Why Faster?**  
- Processes all variables in one pass instead of five separate loops.
- Uses `data.table` and matrix-based aggregation for better memory locality.
- Parallelized chunking reduces runtime dramatically (~86+ hours â†’ several hours on laptop).
- Preserves rook-neighbor topology, numeric estimand, and trained RF model since only data prep changes.

**Estimated Improvement**: 10xâ€“40x speedup depending on disk I/O and number of cores.
END_RESPONSE>>>
