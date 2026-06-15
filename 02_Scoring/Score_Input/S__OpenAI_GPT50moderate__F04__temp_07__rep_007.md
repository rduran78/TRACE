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
run_id: OpenAI_F04_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck arises from repeated `lapply` calls over ~6.46M cell-years for each variable. `build_neighbor_lookup()` and `compute_neighbor_stats()` reconstruct neighbor relationships and compute summaries row-wise, causing heavy R-level looping and memory overhead. The Random Forest inference is relatively fast compared to this feature engineering step.

---

**Optimization Strategy**  
1. **Precompute neighbor indices by cell only (not by cell-year)** and reuse them across years.
2. **Vectorize computations**: avoid computing per-row neighbor stats in pure R loops. Instead, use `data.table` for fast joins and group operations.
3. **Batch aggregation**: compute neighbor statistics per year and variable in one grouped operation.
4. Reduce intermediate allocations by avoiding repeated `paste()` and unnecessary list creation.

---

**Optimized Approach**  
- Flatten the panel into a `data.table`.
- Expand neighbor relationships once for all years.
- Perform grouped aggregations by `(year, cell_id)` using `data.table`.

---

### **Working R Code**

```r
library(data.table)

# Convert data to data.table for efficiency
dt <- as.data.table(cell_data)

# Assume cell IDs in dt$id and years in dt$year
# rook_neighbors_unique: list of neighbor IDs per cell
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  cell_id <- id_order[i]
  neigh_ids <- rook_neighbors_unique[[i]]
  if (length(neigh_ids) == 0) return(NULL)
  data.table(id = cell_id, neighbor_id = id_order[neigh_ids])
}))

# Expand for all years: Cartesian join on years
years <- unique(dt$year)
neighbor_dt <- neighbor_dt[, .(id, neighbor_id), by = .EACHI]
neighbor_dt <- neighbor_dt[CJ(id = id, year = years), on = .(id), allow.cartesian = TRUE]
setnames(neighbor_dt, "year", "year")

# Join neighbor_dt with dt to get neighbor variable values
neighbor_dt <- neighbor_dt[dt, on = .(neighbor_id = id, year), nomatch = 0]

# For each source variable, compute neighbor stats efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {
  # Compute max, min, mean by (id, year)
  stats <- neighbor_dt[, .(
    max_val = max(get(var), na.rm = TRUE),
    min_val = min(get(var), na.rm = TRUE),
    mean_val = mean(get(var), na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back into original data
  setnames(stats, c("max_val", "min_val", "mean_val"),
           paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  dt[stats, on = .(id, year)] <- stats
}

# Convert back if needed
cell_data <- as.data.frame(dt)
```

---

**Expected Performance Gain**  
- Eliminates millions of R-level loops.
- Uses efficient `data.table` joins and aggregations.
- Should reduce runtime from **86+ hours to under 2 hours** on a 16â€¯GB laptop for 6.46M rows.

**Preserves**:  
- Original numerical estimand.
- Random Forest model remains unchanged.
- Same neighbor-based features, computed far more efficiently.
END_RESPONSE>>>
