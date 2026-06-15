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
run_id: OpenAI_F05_T10_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key work is **not just a local inefficiency**; it is part of a larger algorithmic pattern where for each of ~6.46M rows, the pipeline constructs neighbor keys and does a hash lookup into `idx_lookup`. This happens inside `build_neighbor_lookup`, creating **millions of string concatenations and environment lookups**. Given the large dataset (344,208 cells Ã— 28 years = 6.46M rows) and that this lookup occurs for all rows, the observed runtime ballooning to 86+ hours is expected.

The broader issue:  
- Neighbor relationships are **static** across years.
- String concatenation using `paste(neighbor_cell_ids, data$year[i])` repeats unnecessarily because the year dimension is predictable.
- We compute `neighbor_lookup` as a list of integer vectors but recompute string-based keys each time instead of precomputing an efficient mapping.

**Optimization Strategy**  
- **Eliminate string-based keys entirely**. Compute numeric indices by leveraging structured indexing:
  - Precompute a matrix mapping `(cell_id, year)` â†’ row index.
  - Use integer arithmetic for lookups instead of strings.
- Build a **dense matrix of row positions**: rows represent cell id, columns represent years.
- Build neighbor lookups **once** for cells, then replicate across years without costly concatenations.
- Result: All lookups become pure integer operations (fast).

---

### **Optimized Approach**

1. Create `pos_matrix[cells, years]` that maps to row indices in `data`.
2. For each cell-year row `i`, find its neighbors via `neighbors[cell]` and extract indices directly from `pos_matrix[, year]`.
3. Compute stats using vectorized operations.

---

#### **Working R Code**

```r
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Ensure id and year are as expected
  stopifnot(all(sort(unique(data$id)) == sort(id_order)))
  years <- sort(unique(data$year))

  # Precompute matrix: rows=cell ids, cols=years, values=row index in data
  pos_matrix <- matrix(NA_integer_, nrow = length(id_order), ncol = length(years),
                       dimnames = list(as.character(id_order), as.character(years)))
  pos_matrix[cbind(match(as.character(data$id), as.character(id_order)),
                   match(data$year, years))] <- seq_len(nrow(data))

  # lapply row ids: but we only need positions, not string lookups anymore
  row_ids <- seq_len(nrow(data))
  year_idx <- match(data$year, years)
  cell_idx <- match(data$id, id_order)

  lapply(row_ids, function(i) {
    ref_idx <- cell_idx[i]
    neigh_cells <- neighbors[[ref_idx]]
    # For this row's year:
    nbr_rows <- pos_matrix[neigh_cells, year_idx[i]]
    as.integer(nbr_rows[!is.na(nbr_rows)])
  })
}

compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  result <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
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

# Usage
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data, neighbor_lookup, var_name)
  colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Expected Gains**
- Remove ~6.46M Ã— avg. neighbors string concatenations and hash lookups.
- Memory-efficient: one integer matrix plus neighbor lists (minimal overhead).
- Runtime reduction from 86 hours â†’ ~tens of minutes or a few hours, depending on disk I/O and CPU.

**Key Advantages**
- Preserves trained Random Forest model (only transforming features faster).
- Same numerical estimand as original (identical neighbor lookup logic but integer-based).
- Leverages stable `id_order` and `years` for direct indexing rather than dynamic string maps.
END_RESPONSE>>>
