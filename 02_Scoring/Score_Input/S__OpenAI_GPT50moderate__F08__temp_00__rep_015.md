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
run_id: OpenAI_F08_T00_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics for all 6.46 million rows by iterating through each row and looking up neighbors dynamically. This results in repeated work across years, even though the neighbor structure is static. The complexity is roughly `O(N * avg_neighbors)` for each variable, multiplied by 28 years and 5 variables, leading to massive overhead. Additionally, the `lapply` approach with repeated indexing and string concatenation is inefficient in R for such large data.

---

**Optimization Strategy**  
- **Exploit static neighbor structure**: Precompute a neighbor index for each cell (not cell-year) once.
- **Process by year in blocks**: For each year, extract the relevant slice of data and compute neighbor stats using vectorized operations.
- **Avoid repeated string operations**: Use integer indices instead of concatenated keys.
- **Use matrix operations**: Compute max, min, and mean for neighbors using `vapply` or `apply` on precomputed neighbor lists.
- **Memory efficiency**: Work year-by-year to avoid loading all intermediate results in memory.

---

**Optimized R Code**

```r
# Precompute neighbor lookup at cell level (static)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is a list of integer vectors (spdep::nb)
  # Return as-is but ensure it's aligned with id_order
  neighbors
}

compute_neighbor_stats_year <- function(values, neighbor_lookup) {
  # values: numeric vector for one year, length = number of cells
  n <- length(values)
  result <- matrix(NA_real_, nrow = n, ncol = 3)
  colnames(result) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neighbor_vals <- values[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) > 0) {
        result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
      }
    }
  }
  result
}

# Main optimized loop
add_neighbor_features <- function(cell_data, id_order, neighbor_lookup, vars) {
  # Assume cell_data has columns: id, year, and vars
  years <- sort(unique(cell_data$year))
  n_cells <- length(id_order)
  
  # Preallocate result list
  all_results <- vector("list", length(vars))
  names(all_results) <- vars
  
  for (var_name in vars) {
    all_results[[var_name]] <- matrix(NA_real_, nrow = nrow(cell_data), ncol = 3)
  }
  
  for (yr in years) {
    year_idx <- which(cell_data$year == yr)
    # Ensure order matches id_order
    year_data <- cell_data[year_idx, ]
    year_data <- year_data[match(id_order, year_data$id), ]
    
    for (var_name in vars) {
      stats_mat <- compute_neighbor_stats_year(year_data[[var_name]], neighbor_lookup)
      # Place back into all_results
      all_results[[var_name]][year_idx, ] <- stats_mat
    }
  }
  
  # Bind results to cell_data
  for (var_name in vars) {
    colnames(all_results[[var_name]]) <- paste0(var_name, "_nbr_", c("max", "min", "mean"))
    cell_data <- cbind(cell_data, all_results[[var_name]])
  }
  
  cell_data
}

# Usage
neighbor_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features(cell_data, id_order, neighbor_lookup, neighbor_source_vars)
```

---

**Why This Works**  
- **Static neighbor structure**: We compute neighbors once at the cell level.
- **Year-by-year processing**: Reduces memory footprint and avoids redundant computation.
- **Vectorized assignment**: Avoids repeated string concatenation and dynamic lookups.
- **Preserves estimand**: The logic for max, min, and mean remains identical to the original.

**Expected Performance**  
This approach reduces complexity from `O(N * years * vars)` with heavy overhead to `O(cells * avg_neighbors * years * vars)` with minimal overhead and efficient memory use. On a standard laptop, this should reduce runtime from 86+ hours to a few hours or less, depending on I/O speed.
END_RESPONSE>>>
