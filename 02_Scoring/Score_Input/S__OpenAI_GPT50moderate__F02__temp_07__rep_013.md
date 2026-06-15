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
run_id: OpenAI_F02_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The slowness and memory overhead stem from:  
- **Inefficient lapply loops** over 6.46M rows; heavy R-level overhead.  
- **Redundant lookups** for each row/year combination using string concatenation and hash maps.  
- **Repeated computation**: `neighbor_lookup` is built once, but `compute_neighbor_stats` still iterates row-wise.  
- **No vectorization** and no use of data.table/dplyr for grouped or joined operations.  
- **Memory churn**: `lapply` + `do.call(rbind, ...)` creates large intermediate lists.  

---

### **Optimization Strategy**
1. **Precompute neighbor lookups at the cell level only** (not per cell-year) and reuse them.
2. **Vectorize computations** using `data.table` joins instead of row-wise loops.
3. **Compute neighbor stats by joining neighbor rows of the same year**, then aggregate (max, min, mean) in bulk.
4. Use **efficient key-based joins** with `data.table` to avoid string concatenation.
5. **Chunk processing** if memory is tight (e.g., by year).
6. Avoid copying large data frames repeatedly. Use `:=` in place.

---

### **Working R Code (Optimized)**
```r
library(data.table)

# Convert to data.table and set keys
setDT(cell_data)
setkey(cell_data, id, year)

# Prepare neighbor table (cell-level)
neighbor_dt <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  nb_id = unlist(rook_neighbors_unique)
)

# Key neighbor table for fast join
setkey(neighbor_dt, nb_id)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  # Select only needed columns
  vals <- cell_data[, .(id, year, value = get(var_name))]
  
  # Join neighbor_dt to vals on nb_id -> id
  joined <- neighbor_dt[vals, on = .(nb_id = id), allow.cartesian = TRUE]
  # joined has: id (cell), nb_id, i.id (neighbor), year, value
  
  # Now join back to add the year of the target cell
  setkey(joined, id)
  joined <- cell_data[joined, on = .(id), allow.cartesian = TRUE]
  # joined now has: id (target), year (target), nb_id, value (neighbor)
  
  # Filter so neighbor year == target year
  joined <- joined[year == i.year]
  
  # Aggregate stats by target id + year
  stats <- joined[, .(
    nb_max = max(value, na.rm = TRUE),
    nb_min = min(value, na.rm = TRUE),
    nb_mean = mean(value, na.rm = TRUE)
  ), by = .(id, year)]
  
  # Merge back into cell_data
  setkey(stats, id, year)
  cell_data[stats, paste0(var_name, "_nb_max") := nb_max, on = .(id, year)]
  cell_data[stats, paste0(var_name, "_nb_min") := nb_min, on = .(id, year)]
  cell_data[stats, paste0(var_name, "_nb_mean") := nb_mean, on = .(id, year)]
}

# Apply for each neighbor source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_dt(var_name)
}
```

---

### **Why This Works**
- **Single pass joins + grouped aggregation** replace millions of per-row loops.
- **data.table** provides in-memory efficient joins and aggregation.
- Filters neighbors by same `year` before aggregation â†’ same numerical result.
- No retraining of the model; features computed exactly as before.

---

**Expected Performance**:  
- From 86+ hours â†’ likely under 1â€“2 hours on a 16â€¯GB laptop (depending on disk speed and CPU).  
- Memory footprint manageable because joins and aggregations are done in chunks internally.

If memory is still tight, **process by year**:
```r
for (yr in unique(cell_data$year)) {
  # subset cell_data[year == yr], compute, then rbind
}
```  

This preserves the original estimand and Random Forest model while making the pipeline practical.
END_RESPONSE>>>
