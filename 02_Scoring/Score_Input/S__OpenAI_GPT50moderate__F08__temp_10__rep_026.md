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
run_id: OpenAI_F08_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation repeatedly computes neighbor statistics for each `(cell, year)` using an `lapply` across ~6.46M rows. For each row, it builds neighbor lookups by matching both `id` and `year`. This introduces massive redundancy since the neighbor structure is static but being recalculated for every row-year combination. Result: excessive memory access, repeated lookups, and very poor performance (86+ hours).

---

**Optimization Strategy**  
1. **Precompute static neighbor mapping at cell level only** (not per year).
2. **Iterate by year** and compute neighbor stats in batches:  
   - For each year, slice the dataset (all cells for that year) into a temporary data frame.  
   - Use static neighbor index arrays to directly compute stats for that year.  
3. **Vectorized computation** instead of nested `lapply` per row when possible.  
4. Avoid building huge lists of per-row neighbors; reuse the same `neighbor_lookup` across years.  
   
This reduces overhead from millions of duplicated computations and memory lookups.

---

**Working R Code**

```r
# Static neighbor lookup per cell (just indices of cells, no year)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # neighbors is a list: each element corresponds to a cell's neighbors
  # Create a named list of integer vectors representing positions in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_pos[as.character(neighbor_ids)])
  })
}

compute_neighbor_stats_year <- function(year_data, neighbor_lookup, var_name) {
  vals <- year_data[[var_name]]
  n <- nrow(year_data)
  out <- matrix(NA_real_, nrow = n, ncol = 3)  # max, min, mean
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    nbr_vals <- vals[idx]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) > 0) {
      out[i, ] <- c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    } else {
      out[i, ] <- c(NA_real_, NA_real_, NA_real_)
    }
  }
  out
}

add_neighbor_features_batch <- function(data, neighbor_lookup, neighbor_vars) {
  years <- sort(unique(data$year))
  result_list <- vector("list", length(years))
  
  for (y in seq_along(years)) {
    yr <- years[y]
    year_data <- data[data$year == yr, ]
    feature_mats <- lapply(neighbor_vars, function(var) {
      compute_neighbor_stats_year(year_data, neighbor_lookup, var)
    })
    # Combine into data frame
    for (j in seq_along(neighbor_vars)) {
      stats_mat <- feature_mats[[j]]
      colnames(stats_mat) <- paste0(neighbor_vars[j], c("_nbr_max", "_nbr_min", "_nbr_mean"))
      year_data <- cbind(year_data, stats_mat)
    }
    result_list[[y]] <- year_data
  }
  do.call(rbind, result_list)
}

# Usage:
# Precompute static neighbors (no year)
neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_batch(cell_data, neighbor_lookup_static, neighbor_source_vars)
```

---

**Why This Works**  
- **Static neighbor lookup** eliminates repeated year-based lookups.  
- **Year-wise batching** confines vector operations to ~344K rows vs 6.46M repeated overhead.  
- **Preserves estimands** because each neighbor stat is still computed for the same `(cell, year)` using its year-specific variable values.  
- No model retraining; only preprocessing is optimized.

Expected runtime improves from **86+ hours to a few hours** on standard hardware.
END_RESPONSE>>>
