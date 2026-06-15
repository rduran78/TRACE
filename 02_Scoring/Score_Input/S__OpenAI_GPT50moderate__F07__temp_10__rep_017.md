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
run_id: OpenAI_F07_T10_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach is extremely slow because:
- It performs 6.46 million `lapply` calls with repeated `if` checks inside loops.
- Neighbor lookups and keyed name matching (`paste` + mapping) are recomputed for every row.
- R list operations and repeated function calls create heavy overhead on a large dataset.
- Memory inefficiency: intermediate lists and repeated vector allocations strain the 16â€¯GB RAM.

**Optimization Strategy:**  
1. **Precompute neighbor index lookup once for all rows** â€” already done but can be simplified and converted to integer indexing.
2. **Use vectorized matrix operations** instead of millions of small list operations.
3. **Use `data.table` for fast joins and aggregation** on large datasets.
4. **Avoid string-based key assembly per row** â€” join on integer IDs.
5. **Preallocate output matrices** for neighbor stats, fill in a single pass.
6. **Parallelize computations via `parallel` or `future.apply`**.
7. **Keep model intact** â€” only change preprocessing.

---

### **Optimized R Code**

```r
library(data.table)
library(parallel)

# Convert data to data.table for efficiency
setDT(cell_data)

# Build neighbor lookup matrix in integer IDs (once)
id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(seq_len(nrow(cell_data)),
                       paste(cell_data$id, cell_data$year, sep = "_"))

neighbor_lookup <- lapply(seq_len(nrow(cell_data)), function(i) integer(0)) # placeholder

# Efficient neighbor index construction
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  idx_lookup <- setNames(seq_len(nrow(data)),
                         paste(data$id, data$year, sep = "_"))
  res <- vector("list", nrow(data))
  ids <- data$id
  yrs <- data$year
  for (i in seq_len(nrow(data))) {
    ref_idx <- id_to_ref[[as.character(ids[i])]]
    nb_ids <- id_order[neighbors[[ref_idx]]]
    if (length(nb_ids)) {
      keys <- paste(nb_ids, yrs[i], sep = "_")
      res[[i]] <- as.integer(idx_lookup[keys])
    }
  }
  res
}

neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats in parallel fully vectorized
compute_neighbor_stats_fast <- function(dt, var, neighbor_lookup, cores = detectCores() - 1) {
  vals <- dt[[var]]
  n <- length(vals)
  
  cl <- makeCluster(cores)
  clusterExport(cl, c("vals", "neighbor_lookup"), envir = environment())
  
  result <- parLapply(cl, seq_len(n), function(i) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0) return(c(NA_real_, NA_real_, NA_real_))
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (!length(nv)) return(c(NA_real_, NA_real_, NA_real_))
    c(max(nv), min(nv), mean(nv))
  })
  stopCluster(cl)
  
  do.call(rbind, result)
}

# Apply for each variable and bind results
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, var_name, neighbor_lookup)
  set(cell_data, , paste0(var_name, "_nb_max"), stats[, 1])
  set(cell_data, , paste0(var_name, "_nb_min"), stats[, 2])
  set(cell_data, , paste0(var_name, "_nb_mean"), stats[, 3])
}
```

---

### **Why this is Efficient**
- Uses integer index joins, not repeated string manipulation.
- Parallelized over CPU cores for 6.46M entries.
- Avoids repeated `if` checks inside deeply nested loops.
- No redundant recomputation of neighbor relationships.
- Preallocates and binds results directly into `data.table`.

**Expected gain:** From 86+ hours to a few hours depending on CPU cores (6â€“12x faster with 6 cores).

**Random Forest model remains unchanged** â€” only preprocessing is accelerated with the same numerical estimand preserved.
END_RESPONSE>>>
