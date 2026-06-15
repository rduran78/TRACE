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
run_id: OpenAI_F04_T07_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck is the repeated R-level iteration over 6.46M rows in pure R (`lapply` inside `build_neighbor_lookup` and `compute_neighbor_stats`). This approach scales poorly because:  
- `build_neighbor_lookup` creates ~6.46M neighbor lists via nested lookups.
- `compute_neighbor_stats` repeats costly list traversals for each variable (5 times), causing redundant computation.
- Memory overhead from storing large lists of indices.

The Random Forest inference is not the issue; the feature construction dominates runtime.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once per (cell, year)** â†’ already done, but store as a flat vector-friendly structure.
2. **Vectorize neighbor statistics computation** by converting to a long format and using `data.table` grouping instead of per-row `lapply`.
3. **Avoid repeated passes**: compute all neighbor stats for all variables in one grouped operation.
4. **Leverage `data.table` joins and aggregation**, which are optimized in C, to reduce runtime from days to hours or minutes.
5. Use **parallelization** if available (optional), but the main gain is from vectorized aggregation.

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor relationships as long table: (cell_id, year, neighbor_id)
# id_order and rook_neighbors_unique used to build mapping
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    cell_id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand across years efficiently using CJ
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[CJ(year = years, cell_id = cell_id, unique = TRUE), on = .(cell_id)]
setnames(neighbor_dt, "cell_id", "id")  # align with cell_data naming

# Join neighbor values for all variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Long format for neighbors with values
cell_data_long <- cell_data[, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
setkey(cell_data_long, id, year)

neighbor_dt <- neighbor_dt[cell_data_long, on = .(neighbor_id = id, year), nomatch = 0]

# Compute stats grouped by (id, year) over all neighbors
agg <- neighbor_dt[, .(
  ntl_max = max(ntl, na.rm = TRUE),
  ntl_min = min(ntl, na.rm = TRUE),
  ntl_mean = mean(ntl, na.rm = TRUE),
  ec_max = max(ec, na.rm = TRUE),
  ec_min = min(ec, na.rm = TRUE),
  ec_mean = mean(ec, na.rm = TRUE),
  pop_density_max = max(pop_density, na.rm = TRUE),
  pop_density_min = min(pop_density, na.rm = TRUE),
  pop_density_mean = mean(pop_density, na.rm = TRUE),
  def_max = max(def, na.rm = TRUE),
  def_min = min(def, na.rm = TRUE),
  def_mean = mean(def, na.rm = TRUE),
  usd_est_n2_max = max(usd_est_n2, na.rm = TRUE),
  usd_est_n2_min = min(usd_est_n2, na.rm = TRUE),
  usd_est_n2_mean = mean(usd_est_n2, na.rm = TRUE)
), by = .(id, year)]

# Merge back to original
cell_data <- cell_data[agg, on = .(id, year)]
```

---

**Key Advantages**  
- **Single pass aggregation** for all 5 variables instead of 5 Ã— 6.46M iterations.
- `data.table` performs group-by aggregation in C, massively reducing runtime and memory.
- Avoids large intermediate lists; uses efficient joins and grouping.
- Preserves original numerical estimand and trained model.

**Expected runtime reduction**: from 86+ hours to under 1 hour on a standard laptop.
END_RESPONSE>>>
