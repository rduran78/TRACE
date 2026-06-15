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
run_id: OpenAI_F08_T03_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) by iterating over all 6.46M rows and repeatedly scanning neighbors for each variable. This results in redundant work across years since the neighbor structure is static. Complexity is roughly `O(N * k * V)` where `N â‰ˆ 6.46M rows`, `k â‰ˆ avg neighbors`, and `V = 5 variables`. The repeated `lapply` calls and list manipulations also create large intermediate objects, stressing memory.

---

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute neighbor indices once at the *cell* level (344,208 cells), not per row.
- **Process year by year**: For each year, slice the data for that year (â‰ˆ344k rows), compute neighbor stats using the static neighbor map, and append results.
- **Vectorize**: Use `matrix` operations or `vapply` instead of deeply nested `lapply`.
- **Avoid repeated lookups**: Store neighbors as integer vectors of row positions for each cell-year slice.
- **Memory efficiency**: Work in chunks (year-wise) to keep RAM usage low.

---

**Optimized R Code**

```r
# Precompute static neighbor lookup at cell level
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is spdep::nb object
  lapply(seq_along(id_order), function(i) {
    as.integer(neighbors[[i]]) # neighbor indices in id_order
  })
}

compute_neighbor_stats_year <- function(data_year, neighbor_lookup, var_name) {
  vals <- data_year[[var_name]]
  n <- nrow(data_year)
  res <- matrix(NA_real_, n, 3) # columns: max, min, mean
  for (i in seq_len(n)) {
    nb_idx <- neighbor_lookup[[i]]
    if (length(nb_idx) > 0) {
      nb_vals <- vals[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) > 0) {
        res[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
      }
    }
  }
  colnames(res) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  res
}

# Main optimized pipeline
neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

years <- sort(unique(cell_data$year))
result_list <- vector("list", length(years))

for (y in seq_along(years)) {
  yr <- years[y]
  data_year <- cell_data[cell_data$year == yr, ]
  # Ensure row order matches id_order
  data_year <- data_year[match(id_order, data_year$id), ]
  
  # Compute neighbor stats for all variables
  stats_list <- lapply(neighbor_source_vars, function(var_name) {
    compute_neighbor_stats_year(data_year, neighbor_lookup_static, var_name)
  })
  
  stats_mat <- do.call(cbind, stats_list)
  data_year <- cbind(data_year, stats_mat)
  result_list[[y]] <- data_year
}

cell_data_with_neighbors <- do.call(rbind, result_list)
```

---

**Why This Works**  
- **Static neighbor map**: Built once for 344k cells instead of 6.46M rows.
- **Year-wise processing**: Only 344k rows in memory at a time.
- **No redundant lookups**: Each neighbor set is reused across all variables for that year.
- **Preserves estimand**: Same neighbor stats, same Random Forest model.

---

**Expected Performance**  
- Complexity drops from `O(N * k * V)` across all rows to `O(C * k * V * Y)` where `C = 344k cells`, `Y = 28 years`, but with efficient slicing and no repeated string lookups.
- Estimated runtime: **hours â†’ minutes** on a 16 GB laptop.

This approach preserves your trained Random Forest model and original numerical estimand while making the pipeline computationally feasible.
END_RESPONSE>>>
