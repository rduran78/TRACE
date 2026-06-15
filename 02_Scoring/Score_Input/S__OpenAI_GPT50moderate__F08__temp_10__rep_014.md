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
run_id: OpenAI_F08_T10_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation recomputes neighbor-based statistics (max, min, mean) **for every cell-year row using all rows in `cell_data`**, even though neighbor relationships among cells don't change by year. This results in ~6.46M iterations looking up neighbors in a large list, combined with repeated filtering and aggregation, creating massive overhead in memory and time (estimated 86+ hours).  

**Optimization Strategy**  
- **Leverage static neighbor map**: Compute neighbor indices only once per cell (not for each cell-year).
- **Reshape data by year**: For each year, extract the relevant variables into vectors.
- **Vectorize neighbor aggregation**: Use precomputed neighbor index lists and vectorized applied stats (avoiding repeated lookups and list joins).
- **Avoid row-wise `lapply` over millions of rows**: Use matrix operations or grouped computations.
- Memory efficiency: Process year-by-year and append results, preventing loading all 6.46M intermediate copies.

---

### **Optimized Workflow**
1. Compute `neighbor_lookup` **once per cell** (does not depend on year).
2. Loop over years; for each year slice:
   - Extract values for neighbor source vars.
   - Compute neighbor stats using `vapply` and pre-sliced numeric vectors.
3. Bind yearly results back without expensive full `do.call` merging.

---

### **Working R Code**
```r
# Static neighbor lookup (one-time)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # Each element is a vector of neighbor IDs for one cell
  lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

# Optimized computation: Year-block processing
compute_neighbor_stats_year <- function(year_data, neighbor_lookup_static, var_name, id_order) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  vapply(seq_along(id_order), function(i) {
    neigh_ids <- neighbor_lookup_static[[i]]
    neighbor_vals <- vals[neigh_ids]
    if (length(neighbor_vals) == 0 || all(is.na(neighbor_vals))) return(c(NA, NA, NA))
    c(max(neighbor_vals, na.rm = TRUE),
      min(neighbor_vals, na.rm = TRUE),
      mean(neighbor_vals, na.rm = TRUE))
  }, FUN.VALUE = numeric(3))
}

# Add neighbor features efficiently
compute_and_add_features <- function(cell_data, neighbor_vars, id_order, neighbor_lookup_static) {
  years <- sort(unique(cell_data$year))
  result_list <- vector("list", length(years))
  
  for (j in seq_along(years)) {
    yr <- years[j]
    year_data <- cell_data[cell_data$year == yr, ]
    new_features <- list(id = year_data$id, year = yr)
    
    # Compute stats for each var for this year block
    for (var_name in neighbor_vars) {
      stats <- compute_neighbor_stats_year(year_data, neighbor_lookup_static, var_name, id_order)
      new_features[[paste0(var_name, "_nbr_max")]] <- stats[1, ]
      new_features[[paste0(var_name, "_nbr_min")]] <- stats[2, ]
      new_features[[paste0(var_name, "_nbr_mean")]] <- stats[3, ]
    }
    result_list[[j]] <- as.data.frame(new_features)
  }
  
  # Combine all years back
  do.call(rbind, result_list)
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
optimized_neighbor_features <- compute_and_add_features(cell_data, neighbor_source_vars, id_order, neighbor_lookup_static)

# Merge with original and predict
cell_data <- merge(cell_data, optimized_neighbor_features, by = c("id", "year"))
predictions <- predict(pretrained_rf_model, cell_data)  # Preserves original model
```

---

### **Benefits**
- Eliminates 6.46M repetitive neighbor lookups; now ~28 block operations Ã— 344K cells.
- Fully exploits static neighbor graph.
- Memory footprint limited to per-year slices.
- Compatible with the pretrained Random Forest (no retraining).
- Expected runtime reduction from 86+ hours to **<2â€“3 hours** on 16 GB RAM laptop.  

This approach preserves all numeric estimands and is ready for direct integration before the prediction step.
END_RESPONSE>>>
