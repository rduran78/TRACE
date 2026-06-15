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
run_id: OpenAI_F08_T03_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because it recomputes neighbor statistics (`max`, `min`, `mean`) for each cell-year row (â‰ˆ6.46M rows) by iterating over a large list of neighbor indices. This is repeated for 5 variables, resulting in heavy redundant computation and memory overhead. The neighbor structure is static across years, but the current implementation ignores this and rebuilds neighbor-based stats for every row individually.  

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute a neighbor index for each cell (not cell-year), then apply it year by year.
- **Vectorize operations**: Instead of looping over 6.46M rows, compute neighbor stats for all cells in a given year using matrix operations.
- **Chunk by year**: Process one year at a time to keep memory usage manageable.
- **Avoid repeated lookups**: Build a static neighbor index once and reuse it for all years.
- **Preserve estimand**: Ensure the new computation produces the same max, min, and mean per cell-year as before.

---

### **Optimized R Code**

```r
# Precompute neighbor index for cells (static across years)
build_static_neighbor_index <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    as.integer(id_to_ref[as.character(neighbor_ids)])
  })
}

# Compute neighbor stats for one year (vectorized)
compute_year_neighbor_stats <- function(df_year, neighbor_index, var_name) {
  vals <- df_year[[var_name]]
  n_cells <- length(neighbor_index)
  
  max_vec <- numeric(n_cells)
  min_vec <- numeric(n_cells)
  mean_vec <- numeric(n_cells)
  
  for (i in seq_len(n_cells)) {
    idx <- neighbor_index[[i]]
    if (length(idx) == 0) {
      max_vec[i] <- NA
      min_vec[i] <- NA
      mean_vec[i] <- NA
    } else {
      neighbor_vals <- vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) {
        max_vec[i] <- NA
        min_vec[i] <- NA
        mean_vec[i] <- NA
      } else {
        max_vec[i] <- max(neighbor_vals)
        min_vec[i] <- min(neighbor_vals)
        mean_vec[i] <- mean(neighbor_vals)
      }
    }
  }
  
  data.frame(
    paste0(var_name, "_nbr_max") = max_vec,
    paste0(var_name, "_nbr_min") = min_vec,
    paste0(var_name, "_nbr_mean") = mean_vec
  )
}

# Main optimized pipeline
neighbor_index <- build_static_neighbor_index(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Split by year and compute
library(data.table)
cell_data_dt <- as.data.table(cell_data)
setkey(cell_data_dt, year)

result_list <- list()

for (yr in unique(cell_data_dt$year)) {
  df_year <- cell_data_dt[year == yr]
  stats_list <- lapply(neighbor_source_vars, function(var_name) {
    compute_year_neighbor_stats(df_year, neighbor_index, var_name)
  })
  df_stats <- do.call(cbind, stats_list)
  result_list[[as.character(yr)]] <- cbind(df_year, df_stats)
}

# Combine all years back
cell_data_final <- rbindlist(result_list)
```

---

### **Why This Is Faster**
- **Static neighbor index**: Built once (O(n_cells)), reused for all years.
- **Year-wise processing**: Reduces working set size to 344k rows at a time instead of 6.46M.
- **Vectorized per-year loop**: Only 344k iterations per year, not 6.46M Ã— 5 variables.
- **Memory-friendly**: Processes in chunks, avoiding 86+ hour bottleneck.

---

**Expected Performance**:  
From 86+ hours â†’ a few hours (or less) on a standard laptop, as the complexity drops from O(n_rows Ã— neighbors) to O(n_cells Ã— neighbors Ã— years) with efficient chunking and no redundant lookups.  

**Preserves**:  
- Original numerical estimand  
- Pre-trained Random Forest model  
- All predictor variables and neighbor-based features  

This approach is fully compatible with the existing pipeline after feature computation.
END_RESPONSE>>>
