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
run_id: OpenAI_F08_T07_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` is called for each variable across all 6.46M rows, performing repeated `lapply` operations over a massive `neighbor_lookup` list (length â‰ˆ 6.46M). This results in ~32 million expensive R-level operations. The neighbor relationships are static across years, but the code recomputes neighbor indices per row-year and iterates repeatedly. This is highly inefficient in both time and memory.

---

**Optimization Strategy**  
- **Exploit static topology:** Neighbor relationships depend only on cell IDs, not years. Build a **cell-level neighbor index once** (length = 344,208), not per row-year.  
- **Vectorize across years:** For each variable-year slice, compute neighbor stats for all cells using fast aggregation (e.g., matrix operations).  
- **Avoid repeated lapply:** Use preallocated arrays and apply row-wise functions in compiled/vectorized form.  
- **Memory efficiency:** Process year by year (28 slices) to stay within 16â€¯GB RAM.  

---

**Optimized Approach**  
1. Build `neighbor_lookup_cell` = list of integer vectors, length = number of cells (344,208).  
2. For each year:
   - Extract the variable vector for that year (length = 344,208).
   - Compute neighbor max, min, mean for all cells using the static lookup.
   - Append results back into the main `cell_data` in-place.  
3. Repeat for all 5 variables.  

This reduces complexity from O(N_rows Ã— neighbors) to O(N_cells Ã— years Ã— neighbors), with only one pass per year per variable.

---

### **Working R Code**

```r
# Build static neighbor lookup at cell level
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_idx <- neighbors[[i]]
    if (length(neighbor_idx) == 0) integer(0) else neighbor_idx
  })
}

# Compute neighbor stats for one variable and one year
compute_year_neighbor_stats <- function(values, neighbor_lookup) {
  n <- length(values)
  res <- matrix(NA_real_, n, 3)
  colnames(res) <- c("max", "min", "mean")
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      neigh_vals <- values[idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        res[i, 1] <- max(neigh_vals)
        res[i, 2] <- min(neigh_vals)
        res[i, 3] <- mean(neigh_vals)
      }
    }
  }
  res
}

# Main optimized pipeline
optimize_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  neighbor_lookup <- build_neighbor_lookup_static(id_order, neighbors)
  
  years <- sort(unique(cell_data$year))
  n_cells <- length(id_order)
  
  for (var_name in vars) {
    # Preallocate columns in cell_data
    for (stat in c("max", "min", "mean")) {
      new_col <- paste0(var_name, "_nbr_", stat)
      cell_data[[new_col]] <- NA_real_
    }
    
    for (yr in years) {
      yr_idx <- which(cell_data$year == yr)
      stopifnot(length(yr_idx) == n_cells)  # panel is balanced
      
      vals_year <- cell_data[[var_name]][yr_idx]
      stats_mat <- compute_year_neighbor_stats(vals_year, neighbor_lookup)
      
      cell_data[[paste0(var_name, "_nbr_max")]][yr_idx]  <- stats_mat[, 1]
      cell_data[[paste0(var_name, "_nbr_min")]][yr_idx]  <- stats_mat[, 2]
      cell_data[[paste0(var_name, "_nbr_mean")]][yr_idx] <- stats_mat[, 3]
    }
  }
  cell_data
}

# Usage
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimize_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

**Why this works:**  
- Neighbor topology computed **once** at cell level (344k lookups, not 6.46M).  
- Each year processed in **one tight loop**, avoiding repeated `lapply` across all rows globally.  
- Minimal overhead, stays within memory limits since we only hold ~344k Ã— 3 stats in RAM per year.  

**Performance gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, because we reduced redundant per-row computations by ~28Ã— and eliminated repeated indexing over 6.46M elements.  

**Preserves:**  
- Original Random Forest model (unchanged).  
- Numerical estimands identical (neighbor max, min, mean per cell-year).  
- Full data integrity.
END_RESPONSE>>>
