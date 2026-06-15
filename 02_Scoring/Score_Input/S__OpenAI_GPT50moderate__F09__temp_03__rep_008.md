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
run_id: OpenAI_F09_T03_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor relationships for each cell-year row repeatedly inside `compute_neighbor_stats`. With 6.46M rows and 5 variables, this results in massive redundant lookups and memory churn. The neighbor structure is static across years, so rebuilding neighbor indices per row/year is unnecessary. This inefficiency explains the 86+ hour runtime.

---

**Optimization Strategy**  
1. **Precompute a reusable neighbor lookup table** once using cell IDs (not cell-year).
2. For each year, subset the data, join attributes, and compute neighbor stats using the prebuilt lookup.
3. Use `matrix` operations or `vapply` instead of repeated `lapply`/`rbind` for speed.
4. Process year-by-year to keep memory within 16 GB.
5. Append computed features back to the main dataset and preserve the original estimand.

---

**Optimized R Code**

```r
# Precompute neighbor lookup by cell index (static across years)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_idx[as.character(neighbor_ids)])
  })
}

compute_neighbor_stats_matrix <- function(vals, neighbor_lookup) {
  n <- length(vals)
  res <- matrix(NA_real_, nrow = n, ncol = 3)
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neigh_vals <- vals[idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        res[i, ] <- c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
      }
    }
  }
  res
}

# Main optimized pipeline
id_order <- unique(cell_data$id)
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output list
results_list <- vector("list", length = length(unique(cell_data$year)))

years <- sort(unique(cell_data$year))

for (yr in years) {
  message("Processing year: ", yr)
  year_data <- cell_data[cell_data$year == yr, ]
  year_data <- year_data[match(id_order, year_data$id), ]  # align by id_order

  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_neighbor_stats_matrix(year_data[[var_name]], neighbor_lookup_static)
    colnames(stats_mat) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    year_data <- cbind(year_data, stats_mat)
  }

  results_list[[as.character(yr)]] <- year_data
}

# Combine all years back
optimized_cell_data <- do.call(rbind, results_list)

# Now you can run predictions with the existing Random Forest model
# predictions <- predict(trained_rf_model, optimized_cell_data)
```

---

**Why This Is Faster**  
- Neighbor lookup built **once** (O(n_cells)).
- Year-by-year processing avoids 6.46M-row full memory load.
- Vectorized loops over static neighbor lists reduce overhead.
- Eliminates repeated string concatenation and hash lookups.

**Expected Runtime**  
From 86+ hours down to a few hours (dominated by I/O and 28 yearly passes), well within laptop constraints.
END_RESPONSE>>>
