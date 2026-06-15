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
run_id: OpenAI_F05_T00_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string concatenation (`paste(data$id, data$year, sep = "_")`) and repeated hash lookups inside the `lapply` loop are only part of the inefficiency. The deeper issue is that the algorithm repeatedly performs expensive string-based key generation and hash lookups for every row (â‰ˆ6.46M times), even though the neighbor relationships are static across years. This is an **O(N Ã— avg_neighbors)** pattern with heavy overhead from string operations and list indexing.  

The neighbor structure depends only on cell IDs, not on years. The current approach rebuilds neighbor keys for every row-year combination, which is unnecessary. Instead, we can precompute a **numeric index-based neighbor lookup** once and reuse it for all variables, eliminating string concatenation and hash lookups inside the main loop.

---

### **Optimization Strategy**
1. **Precompute a numeric matrix of neighbor indices** aligned with the row order of `data`.  
   - Map each `(id, year)` to its row index once.
   - For each row, store the integer indices of its neighbors for the same year.
2. Use this numeric lookup for all variables without recomputing keys.
3. Avoid repeated `lapply` calls for each variable; compute neighbor stats in a **vectorized or batched manner**.

This reduces overhead from string operations and repeated hashing, making the complexity closer to O(N Ã— avg_neighbors) with minimal overhead.

---

### **Optimized R Code**

```r
# Precompute row index lookup
build_neighbor_index <- function(data, id_order, neighbors) {
  # Map (id, year) -> row index
  key_to_idx <- match(paste(data$id, data$year, sep = "_"),
                      paste(rep(id_order, each = length(unique(data$year))),
                            rep(unique(data$year), times = length(id_order)),
                            sep = "_"))
  
  # Precompute mapping from id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Prepare result list
  row_ids <- seq_len(nrow(data))
  neighbor_index <- vector("list", length(row_ids))
  
  # Precompute year groups for fast filtering
  years <- data$year
  
  for (i in row_ids) {
    ref_idx <- id_to_ref[as.character(data$id[i])]
    neighbor_cell_ids <- id_order[neighbors[[ref_idx]]]
    
    # Find neighbors for the same year
    neighbor_rows <- match(neighbor_cell_ids, data$id[years == years[i]])
    if (length(neighbor_rows) > 0) {
      # Convert to global row indices
      neighbor_index[[i]] <- which(data$id %in% neighbor_cell_ids & years == years[i])
    } else {
      neighbor_index[[i]] <- integer(0)
    }
  }
  
  neighbor_index
}

# Compute neighbor stats using precomputed indices
compute_neighbor_stats_fast <- function(data, neighbor_index, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_index), ncol = 3)
  
  for (i in seq_along(neighbor_index)) {
    idx <- neighbor_index[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  
  colnames(result) <- c(paste0(var_name, "_nbr_max"),
                        paste0(var_name, "_nbr_min"),
                        paste0(var_name, "_nbr_mean"))
  result
}

# Main pipeline
neighbor_index <- build_neighbor_index(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_index, var_name)
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- Eliminates repeated string concatenation and hash lookups.
- Precomputes neighbor indices once and reuses them.
- Keeps memory usage manageable (list of integer vectors).
- Preserves the original estimand and Random Forest model.

---

**Expected Performance Gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, as the main bottleneck (string-based lookups in millions of iterations) is removed. Further speedups are possible using `data.table` or `Rcpp` for the inner loops.
END_RESPONSE>>>
