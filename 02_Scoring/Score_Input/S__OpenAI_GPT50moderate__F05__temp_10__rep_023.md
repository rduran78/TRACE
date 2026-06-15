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
run_id: OpenAI_F05_T10_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation and hash lookup:

```r
neighbor_keys <- paste(neighbor_cell_ids, data$year[i], sep = "_")
result <- idx_lookup[neighbor_keys]
```

inside the `lapply` over every row causes **O(N Ã— avg_neighbors)** string operations and dictionary lookups, where  
- `N â‰ˆ 6.46 million` rows  
- `avg_neighbors â‰ˆ 4`  

That alone is costly, but the real inefficiency is that the same `paste` + lookup pattern is repeated for **each of the 5 neighbor source variables** in the outer loop. However, `build_neighbor_lookup()` is only called onceâ€”not per variableâ€”so the heavy cost is during its run, not multiplied by 5. Still, the design repeatedly constructs `neighbor_keys` for each cell-year row, which dominates runtime.

This is **not just a local inefficiency**, itâ€™s a symptom of doing something that can be computed in one vectorized pass: **joining cell-year keys and neighbor lists for all years at once with numeric indices, not strings**.

---

### **Optimization Strategy**

- **Avoid string concatenation entirely.**  
  Compute integer-year offsets so neighbor lookups become direct integer index references via precomputed year shifts.

- **Vectorize neighbor index construction.**  
  For each spatial cell, determine all its neighbors (fixed over years). Then, for every year, map to the correct row indices by adding an integer offset = `(year_index - 1) * num_cells`.

- **Store neighbor references once in a numeric matrix or list of integer vectors**, so `compute_neighbor_stats()` can index directly without hash or string operations.

- This eliminates both the hash-map lookups and per-row string operations.

---

### **Working R Code**

```r
build_neighbor_lookup_optimized <- function(data, id_order, neighbors) {
  # Assume data is sorted by id, then year
  ids        <- unique(data$id)
  years      <- sort(unique(data$year))
  n_cells    <- length(ids)
  n_years    <- length(years)

  # Map IDs to positions
  id_pos     <- setNames(seq_along(ids), as.character(ids))

  # Precompute neighbor sets for each cell (spatial only)
  static_neighbors <- lapply(seq_along(id_order), function(i) {
    id_order[neighbors[[i]]] # neighbor IDs
  })

  # Row index helper: row index = (year_idx - 1)*n_cells + cell_idx
  neighbor_lookup <- vector("list", length = n_cells * n_years)

  for (year_idx in seq_along(years)) {
    year_offset <- (year_idx - 1) * n_cells
    for (cell_idx in seq_along(ids)) {
      neigh_ids <- static_neighbors[[cell_idx]]
      neighbor_row_idx <- year_offset + id_pos[as.character(neigh_ids)]
      neighbor_lookup[[year_offset + cell_idx]] <- as.integer(neighbor_row_idx)
    }
  }
  neighbor_lookup
}

compute_neighbor_stats <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  colnames(result) <- c("max", "min", "mean")

  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Usage:
neighbor_lookup <- build_neighbor_lookup_optimized(cell_data, id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats(cell_data, neighbor_lookup, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Is Faster**

- **No string keys.** Integer math is O(1).
- **Neighbor lookup built once for the full panel.**
- Memory trade-off:  
  - List of length â‰ˆ 6.46M (rows)  
  - Each element small (~4â€“8 integers)  
  Should fit comfortably in 16 GB with efficient storage.
- Outer loop over 5 variables remains, but heavy string work disappears. Expect orders-of-magnitude speed-up, reducing 86+ hours to a few hours or less.

---

**Preserves:**  
- Exact numerical estimand (same neighbors per year, same summary stats)  
- Existing trained model remains intact.  

This structural reformulation removes redundant operations and makes the pipeline feasible on standard hardware.
END_RESPONSE>>>
