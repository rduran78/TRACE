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
run_id: OpenAI_F08_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is that `compute_neighbor_stats` recomputes max, min, and mean by looping over **all rows** (â‰ˆ6.46M) Ã— **all variables** (5), performing repeated index lookups for neighbors. Since the neighbor graph is **static across years** but the variables change by year, the current approach redundantly processes the same cell-neighbor structure 28 times per variable, leading to extremely high complexity and memory thrashing.  

**Optimization Strategy**  
1. **Exploit Static Relationships**: Build `neighbor_lookup` **once** at the cell level (length = 344,208), not per row.
2. **Split by Year**: For each year, slice values and compute neighbor stats using the precomputed static lookup.
3. **Vectorize Aggregation**: Use fast in-memory operations (`vapply`/matrix ops or `data.table`) rather than inner `lapply` over all rows.
4. **Incremental Write**: Bind results per year, reducing memory overhead.
5. **Preserve Model & Estimand**: Do **not** retrain RF; produce identical statistics.

**Working R Code**

```r
library(data.table)

# Step 1: Static neighbor lookup at cell level (not cell-year)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # neighbors is spdep::nb list with rook neighbors
  lapply(seq_along(id_order), function(i) {
    as.integer(neighbors[[i]])
  })
}

# Step 2: Compute neighbor stats per year efficiently
compute_neighbor_stats_year <- function(year_data, neighbor_lookup, var_name) {
  vals <- year_data[[var_name]]
  # Pre-allocate matrix: rows = n_cell, cols = 3 (max, min, mean)
  res <- matrix(NA_real_, nrow = length(neighbor_lookup), ncol = 3)
  for (i in seq_along(neighbor_lookup)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) > 0) {
      nvals <- vals[idx]
      nvals <- nvals[!is.na(nvals)]
      if (length(nvals) > 0) {
        res[i, ] <- c(max(nvals), min(nvals), mean(nvals))
      }
    }
  }
  res
}

# Step 3: Main Optimized Loop
optimized_neighbor_features <- function(cell_data, id_order, neighbors, vars) {
  # Precompute neighbor lookup (cell-level, static)
  neighbor_lookup <- build_neighbor_lookup_static(id_order, neighbors)
  
  setDT(cell_data)
  setkey(cell_data, year, id)
  
  out_list <- vector("list", length(vars))
  
  for (var_name in vars) {
    # Initialize result container per variable
    var_res <- vector("list", length(unique(cell_data$year)))
    yrs <- sort(unique(cell_data$year))
    
    for (y in seq_along(yrs)) {
      year_val <- yrs[y]
      year_data <- cell_data[year == year_val][order(id)]
      
      stats <- compute_neighbor_stats_year(year_data, neighbor_lookup, var_name)
      colnames(stats) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
      
      var_res[[y]] <- cbind(year_data[, .(id, year)], stats)
    }
    out_list[[which(vars == var_name)]] <- rbindlist(var_res)
  }
  
  # Merge back all new features into original data
  new_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year"), all = TRUE), out_list)
  merge(cell_data, new_features, by = c("id", "year"), all.x = TRUE)
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- optimized_neighbor_features(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)

# Preserve Random Forest step as is, just feed updated cell_data
```

---

**Why This Reduces Runtime from 86+ Hours**  
- `neighbor_lookup` length = 344,208 vs. 6.46M rows.  
- Process each year's data in memory-friendly chunks.  
- Avoid 6.46M Ã— 5 repeated lookups: reuse static neighbor structure.  
- Pure R with `data.table` speeds up merging. Parallelizing the inner loop can reduce runtime further (â‰ˆminutes per variable Ã— 28 years instead of days).  

This approach **preserves all estimands** and maintains full compatibility with the pre-trained Random Forest model.
END_RESPONSE>>>
