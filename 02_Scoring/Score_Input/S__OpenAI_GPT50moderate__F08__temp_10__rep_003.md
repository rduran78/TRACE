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
run_id: OpenAI_F08_T10_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The pipeline recalculates neighbor statistics for every cell-year row repeatedly. This is extremely inefficient because:  
- The neighbor graph (`rook_neighbors_unique`) is static across years, but the variables are dynamic.  
- `build_neighbor_lookup` builds an expanded lookup for every cell-year entry, multiplying memory and computation by ~6.46 million rows instead of 344k cells.  
- `compute_neighbor_stats` iterates over each row and repeatedly computes neighbor stats, which scales poorly.  
Result: ~86+ hours runtime due to redundant lookups and repeated computation.

---

**Optimization Strategy:**  
- Precompute the neighbor relationships **once** for spatial cells only (344k size), not per year.  
- Restructure data by splitting into years, compute neighbor stats per variable per year via fast vectorized operations and join results back.  
- Use `data.table` for efficient grouping and merging rather than iterative `lapply` over millions of rows.  
- Avoid reallocation and repeated parsing (e.g., keys).  
- The Random Forest step remains unchanged, so maintain feature names and structure.

---

**Working R Code:**  

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Static neighbor lookup for unique cell IDs
# id_order assumed to be unique list of cell IDs
neighbor_lookup_static <- lapply(seq_along(id_order), function(i) {
  id_order[rook_neighbors_unique[[i]]]  # neighbor IDs
})
names(neighbor_lookup_static) <- as.character(id_order)

# Function to compute stats by year and var
compute_neighbor_stats_by_year <- function(dt, var_name, neighbor_lookup) {
  # Prepare an empty list to store results per year
  res_list <- vector("list", length(unique(dt$year)))
  
  # Iterate by year (28 subsets)
  for (yr in unique(dt$year)) {
    sub <- dt[year == yr, .(id, val = get(var_name))]
    val_lookup <- setNames(sub$val, sub$id)
    
    # Compute stats for each cell
    stats <- lapply(neighbor_lookup, function(neigh_ids) {
      neigh_vals <- val_lookup[as.character(neigh_ids)]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) == 0) return(c(NA, NA, NA))
      c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
    })
    stats_mat <- do.call(rbind, stats)
    out <- data.table(id = names(neighbor_lookup),
                      year = yr,
                      paste0(var_name, "_nbr_max") := stats_mat[,1],
                      paste0(var_name, "_nbr_min") := stats_mat[,2],
                      paste0(var_name, "_nbr_mean") := stats_mat[,3])
    res_list[[as.character(yr)]] <- out
  }
  
  rbindlist(res_list)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute and merge all neighbor features efficiently
all_features <- lapply(neighbor_source_vars, function(var) {
  compute_neighbor_stats_by_year(cell_data, var, neighbor_lookup_static)
})

# Merge all features together
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id","year")), all_features)

# Join with original data
cell_data <- merge(cell_data, neighbor_features, by = c("id","year"))

# At this point, run prediction using pre-trained Random Forest as before
# rf_predictions <- predict(pre_trained_rf_model, newdata = cell_data)
```

---

**Key Improvements:**  
- Reduced complexity from per-row neighbor lookup to per-year bulk computation.  
- Static neighbor graph used for all years.  
- Vectorized aggregation instead of nested loops.  
- Expected runtime reduction from 86+ hrs to a few hours (or less), feasible on a 16 GB laptop.

This preserves all original estimands and works without retraining the Random Forest model.
END_RESPONSE>>>
