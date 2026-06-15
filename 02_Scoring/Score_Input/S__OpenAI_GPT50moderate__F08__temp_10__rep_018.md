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
run_id: OpenAI_F08_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach computes neighbor statistics (`max`, `min`, `mean`) for each variable across 6.46M cell-year rows using `lapply` inside `compute_neighbor_stats`. This results in massive repeated work because:

- The neighbor relationships (graph) are **static** (do not change across years).
- For each year-variable combination, all neighbors are recomputed using inefficient row-wise lookups.
- No vectorization or grouping by year is utilized â†’ ~6.46M Ã— 5 Ã— 3 operations.

Hence **86+ hours** runtime: the pipeline is doing redundant computation instead of exploiting structure.

---

**Optimization Strategy**  
- **Precompute neighbor graph once** (already done with `rook_neighbors_unique`).
- **Group by year**, slice a vector for that year, compute neighbor stats in a **vectorized way**.
- Use **matrix or `vapply`** to avoid repeated list overhead.
- Produce results block-wise: for each `year`, for all cells simultaneously, using fast aggregation.
- Append results back to the main table.
- Keep alignment of cell-year rows with `id_order` and year.

**Key Idea:**  
For each `year`, extract subset of `data[var]` as a vector, map neighbors using integer index (static), compute stats via `vapply`. Do this for all years Ã— variables, then `rbind`.

---

**Optimized R Code**

```r
# neighbor_lookup_static: list of integer vectors indexed by cell position in id_order
neighbor_lookup_static <- rook_neighbors_unique  # already precomputed via spdep::nb
id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

compute_neighbor_stats_fast <- function(data, var_name, id_order, neighbor_lookup) {
  # Prepare output matrix
  n <- nrow(data)
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(out) <- paste(var_name, c("nb_max", "nb_min", "nb_mean"), sep = "_")
  
  # Process by year
  years <- sort(unique(data$year))
  
  for (yr in years) {
    idx_year <- which(data$year == yr)
    ids_year <- data$id[idx_year]
    vals_year <- data[[var_name]][idx_year]
    
    # Position mapping for this year (id -> index within subset)
    id_to_idx_year <- setNames(seq_along(idx_year), as.character(ids_year))
    
    # For each row in idx_year, compute neighbor stats
    out[idx_year, ] <- t(vapply(ids_year, function(cell_id) {
      pos <- id_to_pos[[as.character(cell_id)]]
      neighbor_ids <- id_order[neighbor_lookup[[pos]]]  # neighbor cell IDs
      neighbor_idx <- id_to_idx_year[as.character(neighbor_ids)]
      neighbor_idx <- neighbor_idx[!is.na(neighbor_idx)]
      if (length(neighbor_idx) == 0) return(c(NA, NA, NA))
      neighbor_vals <- vals_year[neighbor_idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
      c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }, numeric(3)))
  }
  
  out
}

# Apply to all neighbor source vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Main loop
for (var_name in neighbor_source_vars) {
  stats_matrix <- compute_neighbor_stats_fast(cell_data, var_name, id_order, rook_neighbors_unique)
  cell_data <- cbind(cell_data, stats_matrix)
}
```

---

**Why Faster?**
- Eliminates redundant per-row neighbor search across full 6.46M set.
- Reduces complexity to:  
  *(344,208 operations Ã— 28 years Ã— minimal overhead)* instead of full per-row recomputation.
- Uses `vapply` (compiled loop) and processes year subsets in-memory.
- Memory usage: just the current year's slice + output matrices â†’ fits in 16GB RAM.

---

**Preserves:**  
- Original estimand (same max/min/mean definitions).
- Original neighbor graph and order.
- Pretrained Random Forest unchanged.

**Expected runtime:** from **86+ hrs â†’ roughly 1â€“3 hrs** on a 16â€¯GB laptop, dominated by R loops and memory access but drastically reduced redundancy.
END_RESPONSE>>>
