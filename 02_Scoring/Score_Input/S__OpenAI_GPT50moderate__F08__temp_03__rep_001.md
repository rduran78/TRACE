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
run_id: OpenAI_F08_T03_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all ~6.46M rows for each variable, performing repeated lookups into `neighbor_lookup`. This results in redundant computations across years since the neighbor structure is static, but the code recomputes neighbor indices for every row-year combination. Additionally, the `lapply` approach with repeated indexing is memory- and time-intensive.

---

**Optimization Strategy**  
1. **Exploit Static Neighbor Structure**:  
   - Compute a neighbor index map **once per cell** (not per cell-year).
   - For each year, extract the relevant slice of data and compute neighbor stats using **vectorized operations**.

2. **Batch Processing by Year**:  
   - For each year, create a matrix of values for all variables.
   - Use precomputed neighbor indices to compute max, min, mean for all cells in that year.

3. **Memory Efficiency**:  
   - Avoid building a giant list of length 6.46M.
   - Work year-by-year and append results.

4. **Preserve Estimand and Model**:  
   - The Random Forest model remains unchanged.
   - The computed features remain the same (neighbor max, min, mean).

---

**Working R Code**

```r
# Precompute neighbor lookup by cell (static)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # neighbors is spdep::nb object
  lapply(seq_along(id_order), function(i) {
    id_order[neighbors[[i]]]
  })
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

# Compute neighbor stats for one year
compute_neighbor_stats_year <- function(data_year, neighbor_lookup_static, var_name) {
  vals <- setNames(data_year[[var_name]], data_year$id)
  n <- nrow(data_year)
  max_vals <- numeric(n)
  min_vals <- numeric(n)
  mean_vals <- numeric(n)

  for (i in seq_len(n)) {
    neigh_ids <- neighbor_lookup_static[[i]]
    neigh_vals <- vals[as.character(neigh_ids)]
    neigh_vals <- neigh_vals[!is.na(neigh_vals)]
    if (length(neigh_vals) == 0) {
      max_vals[i] <- NA
      min_vals[i] <- NA
      mean_vals[i] <- NA
    } else {
      max_vals[i] <- max(neigh_vals)
      min_vals[i] <- min(neigh_vals)
      mean_vals[i] <- mean(neigh_vals)
    }
  }

  data.frame(
    paste0(var_name, "_nbr_max") = max_vals,
    paste0(var_name, "_nbr_min") = min_vals,
    paste0(var_name, "_nbr_mean") = mean_vals
  )
}

# Main optimized loop
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))

for (y_idx in seq_along(years)) {
  y <- years[y_idx]
  data_year <- cell_data[cell_data$year == y, ]
  stats_list <- lapply(neighbor_source_vars, function(var_name) {
    compute_neighbor_stats_year(data_year, neighbor_lookup_static, var_name)
  })
  result_list[[y_idx]] <- cbind(data_year, do.call(cbind, stats_list))
}

cell_data_with_neighbors <- do.call(rbind, result_list)
```

---

**Why This is Faster**  
- We compute neighbor relationships **once** (344,208 entries), not 6.46M times.
- We process **year by year** (28 iterations), reducing memory pressure.
- Each yearâ€™s computation is O(n_cells Ã— avg_neighbors), which is manageable.

**Expected Performance**  
- From 86+ hours â†’ a few hours or less on a 16 GB laptop, depending on I/O and CPU speed.

This preserves the original estimand and the trained Random Forest model while making the pipeline feasible.
END_RESPONSE>>>
