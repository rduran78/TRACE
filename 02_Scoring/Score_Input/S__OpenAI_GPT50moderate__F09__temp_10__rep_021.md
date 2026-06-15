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
run_id: OpenAI_F09_T10_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is repeatedly constructing and traversing neighbor relationships for 6.46M rows in R using lists and `lapply`. The current approach scales poorly because:
- `build_neighbor_lookup` repeats `paste()` and map lookups for all 6.46M rows, creating massive intermediate objects.
- For each variable, `compute_neighbor_stats` iterates over 6.46M entries again, performing repeated random-access retrievals.
- This results in ~5 Ã— 6.46M row loops and heavy memory pressure.
- The Random Forest part is not the issue; preprocessing dominates.

---

**Optimization Strategy**  
1. **Build a reusable adjacency table once**:
   - Create a data frame with columns: `id`, `year`, `neighbor_id`.
   - Join yearly variable attributes directly via fast joins (e.g., `data.table`), avoiding recomputation and large lists.
2. **Use `data.table` for joins & grouping** for efficiency and vectorization.
3. Compute neighbor stats (`max`, `min`, `mean`) by grouped aggregation on `(id, year)` after joining neighbor values.
4. Add results back to main table via fast merge.
5. Preserve the trained RF model and estimand by only changing the feature generation step.

---

**Optimized R Code** (using `data.table`)  

```r
library(data.table)

# Assume: cell_data (id, year, vars...), id_order, rook_neighbors_unique loaded
setDT(cell_data)

# --- Build reusable adjacency table ---
# From rook_neighbors_unique (length = number of cells)
adj_list <- rookie_neighbors_unique  # assumed loaded
adj_dt <- rbindlist(
  lapply(seq_along(adj_list), function(i) {
    if (length(adj_list[[i]]) == 0) return(NULL)
    data.table(id = id_order[i], neighbor_id = id_order[adj_list[[i]]])
  })
)

# Cartesian join neighbors with years to get full panel neighbor rows
years <- unique(cell_data$year)
adj_panel <- adj_dt[CJ(year = years), on = .(dummy = NULL)]
setnames(adj_panel, c("id", "neighbor_id", "year"))

# Key for fast joins
setkey(adj_panel, neighbor_id, year)
setkey(cell_data, id, year)

# --- Compute neighbor stats for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor var values
  adj_panel[, neighbor_val := cell_data[.SD, on = .(id = neighbor_id, year), get(var_name)]]
  
  # Compute stats by (id, year)
  stats <- adj_panel[!is.na(neighbor_val),
                     .(max = max(neighbor_val),
                       min = min(neighbor_val),
                       mean = mean(neighbor_val)),
                     by = .(id, year)]
  
  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data <- stats[cell_data]
  
  # Rename new columns
  setnames(cell_data,
           c("max", "min", "mean"),
           paste0(var_name, c("_nmax", "_nmin", "_nmean")))
}

# cell_data now has new neighbor features; Random Forest prediction can proceed
```

---

**Expected Gains**  
- Reuses adjacency structure for all vars.
- Vectorized aggregation drastically reduces overhead compared to 6.46M Ã— 5 loops.
- Memory efficient: avoids massive lists, uses `data.table` grouping.
- Estimated runtime: hours â†’ minutes (depending on disk I/O and available RAM).

This preserves original estimand and trained model while making preprocessing highly efficient.
END_RESPONSE>>>
