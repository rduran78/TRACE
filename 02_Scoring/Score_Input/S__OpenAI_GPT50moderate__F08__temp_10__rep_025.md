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
run_id: OpenAI_F08_T10_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current bottleneck occurs because `compute_neighbor_stats` operates over all 6.46M rows (cell-years), repeatedly scanning the full dataset year-by-year for every variable. `neighbor_lookup` is recomputed for each cell-year index, even though the neighbor relationships are static across time. As a result, the nested loops and repeated list operations lead to excessive time complexity and memory pressure (hundreds of millions of lookups).

---

**Optimization Strategy**  
1. **Separate static & dynamic components:**  
   - Neighbor relationships never change â†’ build a **cell-to-cell** lookup once (size = 344,208, uses rook adjacency).
   - Dynamic variables (ntl, ec, etc.) vary by year â†’ compute neighbor stats per YEAR in **vectorized** chunks.
   
2. **Efficient joins and calculations:**  
   - Use `data.table` for fast grouping, key-based joins, and memory efficiency.
   - Avoid `lapply` row-wise loops; instead, handle neighbor aggregation by ID â†’ join back to cell-year rows.
   
3. **Preserve numerical estimand:**  
   - Compute neighbor max, min, and mean exactly as before by excluding `NA`s.

Target complexity reduction:  
Original pseudo complexity: O(N_rows Ã— avg_neighbors Ã— vars) â‰ˆ O(6.46M Ã— ~4 Ã— 5) = very large.  
Optimized: O(N_years Ã— (N_cells + total_edges)) â‰ˆ O(28 Ã— (344k + 1.37M)) = feasible in minutes on 16GB RAM.

---

**Working R Code**

```r
library(data.table)

# Assume cell_data has columns: id (integer), year (integer), ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Build static neighbor lookup once
# rook_neighbors_unique: list of integer vectors (neighbors), length = n_cells = length(id_order)
# id_order: vector of cell IDs in same order as rook_neighbors_unique
neighbor_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)
# neighbor_dt: columns id, neighbor_id
setkey(neighbor_dt, neighbor_id) # We'll join by neighbor_id when computing stats

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output columns
for (var_name in neighbor_source_vars) {
  for (stat in c("max", "min", "mean")) {
    colname <- paste0(var_name, "_nbr_", stat)
    cell_data[, (colname) := NA_real_]
  }
}

years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset for one year
  year_dt <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_dt, id)

  # Melt or loop variables
  for (var_name in neighbor_source_vars) {
    # Select id + var
    tmp <- year_dt[, .(id, value = get(var_name))]
    setkey(tmp, id)

    # Join neighbor_dt by neighbor_id -> tmp.id
    merged <- neighbor_dt[tmp, on = .(neighbor_id = id)]
    # Now merged: id (original), neighbor_id, value

    # Compute stats by original cell (id)
    stats <- merged[, .(
      max_val = if (.N == 0) NA_real_ else max(value, na.rm = TRUE),
      min_val = if (.N == 0) NA_real_ else min(value, na.rm = TRUE),
      mean_val = if (.N == 0) NA_real_ else mean(value, na.rm = TRUE)
    ), by = id]

    # Join back to cell_data for this year
    setkey(stats, id)
    setkey(year_dt, id)
    year_dt[stats, paste0(var_name, "_nbr_max") := i.max_val]
    year_dt[stats, paste0(var_name, "_nbr_min") := i.min_val]
    year_dt[stats, paste0(var_name, "_nbr_mean") := i.mean_val]
  }

  # Write results back
  cell_data[year == yr, names(year_dt) := year_dt]
}

# At this point, cell_data has neighbor features computed for all years
# No retraining of Random Forest is needed. Proceed to predictions as before.
```

---

**Why this is fast:**  
- The heavy lifting is done per year in batches using `data.table`, avoiding per-row loops.
- Neighbor graph processed only once.
- Memory footprint controlled by year-wise processing (~344k rows per batch vs 6.46M all at once).
- Exact numerical semantics preserved (`max`, `min`, `mean` of available non-NA neighbors).

**Expected improvement:** From ~86+ hours to minutes (depending on I/O), easily feasible on a 16 GB laptop.
END_RESPONSE>>>
