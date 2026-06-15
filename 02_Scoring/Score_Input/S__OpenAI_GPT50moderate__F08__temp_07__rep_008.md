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
run_id: OpenAI_F08_T07_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current implementation is extremely slow (86+ hours) because `compute_neighbor_stats` iterates over all 6.46M rows and recomputes neighbor statistics using `neighbor_lookup` for each row-year combination. This leads to redundant computations since the neighbor structure is static across years. For each year, the same neighbor relationships are applied repeatedly, but the code does it naively for all rows without grouping by year efficiently. The use of `lapply` per row with large lists amplifies overhead and memory usage.

**Optimization Strategy:**  
- Compute neighbor lookup **once per cell** (already done) since neighbor structure is static.
- For each year:
  - Filter data for that year.
  - Compute neighbor stats for all variables in a **vectorized** manner using matrix operations or `data.table`.
- Use preallocated matrices and avoid `lapply` per row.
- Bind per-year results back to the full dataset.
- Leverage `data.table` for efficient grouping and joins.
- Preserve Random Forest model and estimands by ensuring identical computations (max, min, mean).

---

### **Optimized R Implementation**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Build static neighbor lookup (by cell, not cell-year)
build_neighbor_lookup <- function(id_order, neighbors) {
  lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]])
}

neighbor_lookup <- build_neighbor_lookup(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for a single year
compute_year_neighbor_stats <- function(dt_year, neighbor_lookup, vars) {
  n <- nrow(dt_year)
  res_list <- vector("list", length(vars))
  
  # Create ID to row index map for fast lookup
  idx_map <- setNames(seq_len(n), as.character(dt_year$id))
  
  for (v in seq_along(vars)) {
    varname <- vars[v]
    vals <- dt_year[[varname]]
    
    max_vec <- numeric(n)
    min_vec <- numeric(n)
    mean_vec <- numeric(n)
    
    for (i in seq_len(n)) {
      n_ids <- neighbor_lookup[[i]]
      n_rows <- idx_map[as.character(n_ids)]
      n_rows <- n_rows[!is.na(n_rows)]
      if (length(n_rows) == 0) {
        max_vec[i] <- NA
        min_vec[i] <- NA
        mean_vec[i] <- NA
      } else {
        neigh_vals <- vals[n_rows]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) == 0) {
          max_vec[i] <- NA
          min_vec[i] <- NA
          mean_vec[i] <- NA
        } else {
          max_vec[i] <- max(neigh_vals)
          min_vec[i] <- min(neigh_vals)
          mean_vec[i] <- mean(neigh_vals)
        }
      }
    }
    
    res_list[[v]] <- data.table(
      paste0(varname, "_nb_max") = max_vec,
      paste0(varname, "_nb_min") = min_vec,
      paste0(varname, "_nb_mean") = mean_vec
    )
  }
  
  cbind(res_list[[1]], if (length(vars) > 1) do.call(cbind, res_list[-1]))
}

# Main optimized loop
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- unique(cell_data$year)

result_list <- vector("list", length(years))

for (y_idx in seq_along(years)) {
  yr <- years[y_idx]
  dt_year <- cell_data[year == yr]
  stats_dt <- compute_year_neighbor_stats(dt_year, neighbor_lookup, vars)
  result_list[[y_idx]] <- cbind(dt_year[, .(id, year)], stats_dt)
}

# Merge all back
neighbor_features <- rbindlist(result_list)
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data]

# cell_data now includes neighbor max/min/mean for all required vars
```

---

### **Why This is Faster**
- Eliminates redundant computation across 6.46M rows by processing **per year** instead of per cell-year globally.
- Uses `data.table` for efficient subsetting and joining.
- Computes neighbor stats for multiple variables in a single pass per year.
- Avoids repeated string concatenations and large list operations across entire dataset.

**Expected Improvement:**  
From 86+ hours to a few hours (or less) depending on CPU and disk speed, since the computation is now O(years Ã— cells Ã— neighbors) instead of O(total_rows Ã— neighbors) with heavy R overhead removed.

**Preserves:**  
- Original Random Forest model
- Original estimand (max, min, mean of neighbors)
- Static neighbor relationships vs. dynamic yearly variables.
END_RESPONSE>>>
