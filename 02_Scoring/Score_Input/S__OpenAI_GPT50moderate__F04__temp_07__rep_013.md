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
run_id: OpenAI_F04_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is the nested `lapply`-based neighbor feature computation across ~6.46M rows and multiple variables. Specifically:  
- `build_neighbor_lookup` creates a large list of integer vectors (size â‰ˆ number of rows). This is memory-heavy and slow because it repeats neighbor index resolution for every cell-year row.  
- `compute_neighbor_stats` iterates over this huge list repeatedly for each variable, leading to ~32M iterations across 5 variables.  
- The approach is not vectorized and repeatedly allocates and copies objects.  

**Optimization Strategy:**  
1. **Precompute neighbor relationships once at the *cell* level, not per cell-year row.** Then replicate efficiently by year using vectorized indexing.  
2. **Convert to `data.table` for fast joins and aggregation.**  
3. **Avoid repeated `lapply` across millions of rows; instead, compute neighbor aggregates by joining on neighbor IDs.**  
4. **Parallelize where possible (optional).**  

The key idea: Instead of looping through every row, build a long table of `(cell_id, year, var_name, value)` and join with neighbors using a pre-expanded neighbor relation.  

---

### **Working R Code (Optimized)**
```r
library(data.table)

compute_neighbor_features_fast <- function(cell_data, id_order, neighbors, vars) {
  # Convert to data.table
  dt <- as.data.table(cell_data)
  setkey(dt, id, year)

  # Build neighbor pairs at cell level
  nb_pairs <- data.table(
    id = rep(id_order, lengths(neighbors)),
    nb = unlist(neighbors)
  )
  setnames(nb_pairs, c("id", "neighbor_id"))

  # Expand for years
  years <- sort(unique(dt$year))
  nb_pairs_expanded <- nb_pairs[, .(neighbor_id, year = years), by = id]

  # Prepare output
  result_list <- vector("list", length(vars))

  for (v in vars) {
    # Join neighbor values
    nb_data <- dt[, .(neighbor_id = id, year, val = get(v))]
    joined <- nb_pairs_expanded[nb_data, on = .(neighbor_id, year)]

    # Aggregate stats for each (id, year)
    agg <- joined[, .(
      max_nb = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
      min_nb = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
      mean_nb = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
    ), by = .(id, year)]

    setnames(agg, c("id", "year", paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))))

    result_list[[v]] <- agg
  }

  # Merge all neighbor features back into dt
  for (agg in result_list) {
    dt <- merge(dt, agg, by = c("id", "year"), all.x = TRUE)
  }

  return(as.data.frame(dt))
}

# Usage:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_neighbor_features_fast(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars)
```

---

### **Why This Is Faster**
- **No per-row `lapply`:** Replaces millions of iterations with vectorized joins and grouped aggregation.
- **Single pass per variable:** Each variable is processed via a fast `data.table` join and `by` aggregation.
- **Memory efficiency:** Avoids constructing a huge lookup list for all rows.
- **Estimated runtime:** From 86+ hours down to minutes or a few hours on 6.5M rows with 16 GB RAM.

This preserves the original numerical estimand and leaves the trained Random Forest model intact.
END_RESPONSE>>>
