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
run_id: OpenAI_F08_T00_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each of the ~6.46 million cell-year rows individually, repeatedly scanning neighbor indices. This results in redundant work across years since the neighbor structure is static. The complexity is roughly `O(N * k)` per variable, where `N â‰ˆ 6.46M` and `k` is average neighbor count, multiplied by 5 variables â†’ huge overhead. Memory pressure is also high because of repeated list operations.

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute neighbor indices once at the cell level (344,208 cells), not per cell-year.
- **Vectorize by year**: For each year, slice the data and compute neighbor stats in bulk using matrix operations.
- **Avoid repeated list traversals**: Use `vapply` or `matrixStats` for speed.
- **Chunk by year**: Process 28 yearly slices sequentially to keep memory manageable.
- **Reuse neighbor lookup**: A list of integer vectors of neighbor row indices for cells (not cell-years).
- **Write results back efficiently**: Preallocate columns and fill by year.

This reduces complexity to `O(C * k * Y)` where `C â‰ˆ 344k`, `Y = 28`, which is far smaller than `O(N * k)`.

---

### **Optimized R Code**

```r
# Precompute neighbor lookup at cell level (static)
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_ref[as.character(neighbor_ids)])
  })
}

# Compute neighbor stats for one variable and one year
compute_yearly_neighbor_stats <- function(vals, neighbor_lookup) {
  n <- length(vals)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(res) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        res[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  res
}

# Main optimized pipeline
optimize_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  # Precompute static neighbor lookup
  neighbor_lookup <- build_cell_neighbor_lookup(id_order, neighbors)
  
  # Prepare output columns
  for (v in vars) {
    cell_data[[paste0(v, "_nbr_max")]] <- NA_real_
    cell_data[[paste0(v, "_nbr_min")]] <- NA_real_
    cell_data[[paste0(v, "_nbr_mean")]] <- NA_real_
  }
  
  years <- sort(unique(cell_data$year))
  
  for (yr in years) {
    message("Processing year: ", yr)
    idx_year <- which(cell_data$year == yr)
    slice <- cell_data[idx_year, ]
    
    # Ensure slice is in id_order
    slice <- slice[match(id_order, slice$id), ]
    
    for (v in vars) {
      stats <- compute_yearly_neighbor_stats(slice[[v]], neighbor_lookup)
      cell_data[idx_year, paste0(v, "_nbr_max")]  <- stats[, 1]
      cell_data[idx_year, paste0(v, "_nbr_min")]  <- stats[, 2]
      cell_data[idx_year, paste0(v, "_nbr_mean")] <- stats[, 3]
    }
  }
  
  cell_data
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

### **Why This Works**
- **Static neighbor structure**: We compute neighbor indices once for 344k cells.
- **Yearly slicing**: Each year processes 344k rows instead of 6.46M at once.
- **Memory efficiency**: Only one year in memory for neighbor stats at a time.
- **Preserves estimand**: Same max/min/mean logic, just reorganized.
- **No retraining**: Random Forest model remains unchanged.

**Expected runtime**: From 86+ hours down to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU speed. Further speedups possible with `data.table` or `Rcpp` if needed.
END_RESPONSE>>>
