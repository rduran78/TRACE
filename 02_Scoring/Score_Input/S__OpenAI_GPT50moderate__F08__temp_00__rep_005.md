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
run_id: OpenAI_F08_T00_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each of the ~6.46 million cell-year rows by iterating through a large list (`neighbor_lookup`) and repeatedly subsetting the full dataset. This results in massive redundant work:  
- The neighbor structure is static across years, but the code recomputes neighbor indices for every row-year combination.  
- For each variable, the loop traverses all rows again, causing repeated expensive lookups.  
- Memory overhead is high because of large lists and repeated `lapply` calls.  

**Optimization Strategy**  
1. **Precompute static neighbor relationships once** at the cell level (not cell-year).  
2. **Process data year by year**: For each year, extract the relevant slice of the data and compute neighbor stats using the static neighbor map.  
3. Use **vectorized operations** instead of per-row `lapply`.  
4. Avoid building large intermediate lists; instead, compute matrices or data frames directly.  
5. Append results incrementally to avoid holding everything in memory at once.  

This reduces complexity from `O(n_rows * avg_neighbors)` to roughly `O(n_years * n_cells * avg_neighbors)` with efficient vectorization, which is far faster and memory-friendly.

---

### **Optimized R Code**

```r
# Precompute static neighbor lookup at cell level
# neighbors: spdep::nb object
# id_order: vector of cell IDs in the same order as neighbors
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # Return a list where each element is the neighbor indices for that cell
  lapply(neighbors, function(nbr_idx) id_order[nbr_idx])
}

# Compute neighbor stats for one year and one variable
compute_year_neighbor_stats <- function(year_data, var_name, static_lookup) {
  vals <- setNames(year_data[[var_name]], year_data$id)
  
  # For each cell, compute max, min, mean of neighbors
  res <- t(sapply(static_lookup, function(nbr_ids) {
    nbr_vals <- vals[as.character(nbr_ids)]
    nbr_vals <- nbr_vals[!is.na(nbr_vals)]
    if (length(nbr_vals) == 0) return(c(NA, NA, NA))
    c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
  }))
  
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  res
}

# Main optimized pipeline
compute_all_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  static_lookup <- build_static_neighbor_lookup(id_order, neighbors)
  
  years <- sort(unique(cell_data$year))
  result_list <- vector("list", length(years))
  
  for (i in seq_along(years)) {
    yr <- years[i]
    year_data <- cell_data[cell_data$year == yr, ]
    year_data <- year_data[match(id_order, year_data$id), ]  # align with id_order
    
    # Compute neighbor stats for all vars and bind
    neighbor_features <- do.call(cbind, lapply(vars, function(v) {
      compute_year_neighbor_stats(year_data, v, static_lookup)
    }))
    
    # Combine with year_data
    result_list[[i]] <- cbind(year_data, neighbor_features)
  }
  
  do.call(rbind, result_list)
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# Now cell_data has neighbor features ready for Random Forest prediction
```

---

**Why This Works**  
- **Static neighbor map** avoids recomputation across years.  
- **Year-by-year processing** keeps memory usage manageable.  
- **Vectorized `sapply`** over cells is much faster than nested `lapply` over rows.  
- Preserves original estimand and Random Forest model (no retraining).  

**Expected Performance**  
On a standard laptop, this approach should reduce runtime from 86+ hours to a few hours or less, depending on I/O and CPU speed, because it eliminates redundant computations and leverages vectorization.
END_RESPONSE>>>
