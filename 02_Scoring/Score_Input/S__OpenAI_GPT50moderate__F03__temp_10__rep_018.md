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
run_id: OpenAI_F03_T10_R018
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 18
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F03_OpenAI_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**

The bottleneck is in the neighbor feature computation, not in Random Forest inference itself. The current approach repeatedly uses `lapply` with row-by-row indexing, creating millions of intermediate vectors/lists. This is extremely inefficient for 6.46M rows because:

- `build_neighbor_lookup` creates a full list for every cell-year row (6.46M entries), with repeated calls to `paste()` and index lookups.
- `compute_neighbor_stats` uses `lapply` over these millions of entries, triggering huge memory overhead and repeated list allocations.
- Each neighbor-source variable recomputes `rbind` from many small chunks, multiplying cost by 5.

The RF prediction call is likely fast; the preparation loop dominates runtime and memory footprint.

---

**Optimization Strategy:**

1. **Avoid creating a per-row neighbor lookup.** Instead use a vectorized approach and join on IDs/years.
2. **Leverage `data.table` keyed joins or `dplyr` group operations** to compute neighbor stats without gigabyte-scale lists.
3. **Compute neighbor features in a single grouped aggregation rather than iterating over rows.**
4. Keep data in `data.table` throughout for efficient memory usage and joins.
5. Preserve model as-is; only optimize preprocessing.

---

**Working R Code (Vectorized with `data.table`):**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Ensure keys for fast join
setkey(cell_data id year)

# Expand neighbor relationships for all years
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
neighbors_dt <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    data.table(
      src_id = id_order[i],
      dst_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Cartesian join to replicate neighbors across years
neighbor_full <- neighbors_dt[cell_data, on = .(src_id = id), allow.cartesian = TRUE]
neighbor_full <- neighbor_full[, .(id = id, year = year, neighbor_id = dst_id)]

# Join neighbor values
for (var_name in neighbor_source_vars) {
  neighbor_vals <- cell_data[, .(neighbor_id = id, year, val = get(var_name))]
  neighbor_joined <- neighbor_full[neighbor_vals, on = .(neighbor_id, year)]
  
  # Compute aggregated stats by cell-year
  feats <- neighbor_joined[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Add columns back to cell_data
  setkey(feats, id, year)
  setkey(cell_data, id, year)
  cell_data[feats,
            paste0(var_name, "_nb_max") := i.max_val]
  cell_data[feats,
            paste0(var_name, "_nb_min") := i.min_val]
  cell_data[feats,
            paste0(var_name, "_nb_mean") := i.mean_val]
}

# Ensure NA handling consistent with original code
# Random Forest prediction stays the same:
pred <- predict(trained_rf_model, newdata = cell_data)
```

---

**Why this is fast and memory-efficient:**
- No huge `lapply` lists: aggregation happens via grouped operations in `data.table` C-optimized routines.
- Neighbor replication done once, re-used for all variables.
- Avoids repeated `paste()` keys and repeated `rbind` calls.
- Scales well for millions of rows with available RAM on 16 GB machine.

Estimated runtime drops from ~86 hours to a few hours or less, depending on disk I/O and CPU.
END_RESPONSE>>>
