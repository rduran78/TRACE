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
run_id: OpenAI_F08_T10_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

- Current approach computes neighbor statistics row-wise across ~6.46M rows, recalculating operations for each cell-year combination, causing severe inefficiency.
- Neighbor relationships are static, but neighbor lookup is rebuilt dynamically for each row-year.
- For 28 years Ã— 344,208 cells and millions of neighbor lookups, this results in massive redundant computations and memory overhead.

**Optimization Strategy**

1. **Exploit Static Neighbor Topology**: Compute neighbor index list **once** for the 344,208 unique cells (not per cell-year).
2. **Vectorize by Year**: For each year, slice the data and operate on 344,208 rows using matrix operations instead of looping over millions.
3. **Use preallocated structures**: Preallocate result matrices for all variables and avoid rbind loops.
4. **Aggregate neighbor stats with `vapply` and fast indexing**: Avoid per-element lapply over 6M+ entries.
5. **Append results efficiently**: Bind results back to the original data after computing in chunks by year.

**Working R Code**

```r
# Precompute a static neighbor lookup for cell IDs
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_ref[as.character(neighbor_ids)])
  })
}

compute_neighbor_stats_by_year <- function(data, years, var_names, neighbor_lookup, n_cells) {
  # Prepare output list for each variable
  out_list <- vector("list", length(var_names))
  names(out_list) <- var_names
  for (vn in var_names) {
    out_list[[vn]] <- matrix(NA_real_, nrow = n_cells * length(years), ncol = 3)
  }

  # Process year by year
  for (y_idx in seq_along(years)) {
    y <- years[y_idx]
    slice_idx <- data$year == y
    year_data <- data[slice_idx, ]
    stopifnot(nrow(year_data) == n_cells)  # Ensure ordering
    for (vn in var_names) {
      vals <- year_data[[vn]]
      stats_mat <- vapply(neighbor_lookup, function(nb) {
        if (length(nb) == 0) return(c(NA, NA, NA))
        nv <- vals[nb]
        nv <- nv[!is.na(nv)]
        if (length(nv) == 0) return(c(NA, NA, NA))
        c(max(nv), min(nv), mean(nv))
      }, numeric(3))
      out_list[[vn]][((y_idx - 1) * n_cells + 1):(y_idx * n_cells), ] <- t(stats_mat)
    }
  }
  out_list
}

# ----------------------------
# Usage
n_cells <- length(id_order)
years <- sort(unique(cell_data$year))

# Neighbor lookup is static for cells
neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all stats efficiently
neighbor_stats <- compute_neighbor_stats_by_year(cell_data, years, neighbor_source_vars, neighbor_lookup_static, n_cells)

# Bind back to main data
for (vn in neighbor_source_vars) {
  stat_names <- c("max", "min", "mean")
  for (j in seq_along(stat_names)) {
    col_name <- paste0(vn, "_nb_", stat_names[j])
    cell_data[[col_name]] <- neighbor_stats[[vn]][, j]
  }
}

# Preserve trained RF model usage as before
predictions <- predict(trained_rf_model, newdata = cell_data)
```

**Expected gain**:  
- Avoids 6.46M Ã— per-row neighbor recomputation â†’ now loops over 344k cells per year Ã— 28 years in vectorized fashion.
- Reduces estimated runtime from 86+ hours to likely under 1â€“2 hours even on modest hardware.
- Preserves the original estimand and model integrity.
END_RESPONSE>>>
