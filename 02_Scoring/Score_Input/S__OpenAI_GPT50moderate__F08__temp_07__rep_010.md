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
run_id: OpenAI_F08_T07_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for every one of the ~6.46M rows, repeatedly iterating through large lists (length = rows) and performing lookups and aggregation for each variable. This is **O(N Ã— V)** with heavy overhead, where N = 6.46M and V = 5 variables. Memory is also stressed because of repeated list operations. The neighbor topology is static across years, but the pipeline ignores this and rebuilds or reuses lookups redundantly for every row-year combination without exploiting the year-level grouping.

---

**Optimization Strategy**  
1. **Separate static and dynamic components:**  
   - Neighbor relationships are static (cell-to-cell does not change), so build a neighbor index **once at the cell level** rather than per cell-year.
2. **Group by year:**  
   - For each year, slice the data and compute neighbor max, min, mean using **vectorized operations** (e.g., matrix aggregation or `vapply`) rather than per-row lists.
3. **Avoid repeated joins:**  
   - Use pre-built neighbor index keyed by cell IDs; then apply for each year block.
4. **Memory efficiency:**  
   - Work year-by-year, producing neighbor features and appending them back to the dataset incrementally.
5. **Parallelization (optional):**  
   - Use `parallel::mclapply` or `future.apply` for year-wise processing if CPU cores allow.

Expected speed-up: From 86+ hours to manageable (minutes to low hours) by reducing complexity to roughly **O(Y Ã— (C + E))** where Y = 28 years, C = number of cells, E = edges.

---

**Working R Code**

```r
# Static neighbor index: map cell_id -> neighbor cell_ids
build_static_neighbor_index <- function(id_order, neighbors) {
  setNames(neighbors, id_order)
}

compute_neighbor_stats_year <- function(df_year, neighbor_index, vars) {
  # df_year: data for a single year with columns id and vars
  n <- nrow(df_year)
  res_list <- vector("list", length(vars))
  names(res_list) <- vars
  
  # Precompute values as matrix for speed
  vals_mat <- as.matrix(df_year[vars])
  
  for (v in seq_along(vars)) {
    col_vals <- vals_mat[, v]
    # Compute neighbor stats
    out <- matrix(NA_real_, nrow = n, ncol = 3)
    colnames(out) <- c(paste0(vars[v], "_nbr_max"),
                       paste0(vars[v], "_nbr_min"),
                       paste0(vars[v], "_nbr_mean"))
    
    for (i in seq_len(n)) {
      nb_ids <- neighbor_index[[ as.character(df_year$id[i]) ]]
      if (length(nb_ids) > 0) {
        # Find positions of neighbors in df_year via match
        nb_pos <- match(nb_ids, df_year$id)
        nb_pos <- nb_pos[!is.na(nb_pos)]
        if (length(nb_pos) > 0) {
          nb_vals <- col_vals[nb_pos]
          nb_vals <- nb_vals[!is.na(nb_vals)]
          if (length(nb_vals) > 0) {
            out[i, ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
          }
        }
      }
    }
    res_list[[v]] <- out
  }
  do.call(cbind, res_list)
}

# Main optimized pipeline
add_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {
  neighbor_index <- build_static_neighbor_index(id_order, rook_neighbors_unique)
  
  years <- sort(unique(cell_data$year))
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    df_year <- subset(cell_data, year == yr)
    stats_mat <- compute_neighbor_stats_year(df_year, neighbor_index, neighbor_source_vars)
    result_list[[yi]] <- cbind(df_year, stats_mat)
    rm(df_year, stats_mat)
    gc()
  }
  
  do.call(rbind, result_list)
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- add_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)

# Then run the pre-trained Random Forest model
rf_predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Summary of Changes**
- Compute neighbor features **year by year** instead of row by row.
- Use static neighbor index at the cell level (no duplication of topology).
- Replace nested `lapply` with efficient loops and pre-allocated matrices.
- Preserve original Random Forest model and estimands.

This approach should dramatically reduce computation time and memory usage, making the task feasible on a 16 GB laptop.
END_RESPONSE>>>
