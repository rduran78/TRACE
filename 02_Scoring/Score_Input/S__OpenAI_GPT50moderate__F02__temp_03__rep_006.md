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
run_id: OpenAI_F02_T03_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F02_OpenAI_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length â‰ˆ 6.46M (one per cell-year), each storing neighbor indices. This is extremely costly in memory and time.  
- `compute_neighbor_stats` repeatedly scans these lists and subsets vectors, causing millions of small allocations.  
- The outer loop repeats this for 5 variables, multiplying overhead.  
- The approach is row-wise and not vectorized, unsuitable for 6.46M rows on a 16â€¯GB laptop.  

---

**Optimization Strategy**  
1. **Avoid per-row lists**: Instead of building a huge list, work at the *cell level* and join neighbor stats by year.  
2. **Precompute neighbor relationships once**: Expand rook neighbors into a long edge list `(cell_id, neighbor_id)`.  
3. **Use data.table for fast joins and aggregation**: Compute neighbor stats per year and variable in a grouped manner.  
4. **Iterate over variables efficiently**: Compute all neighbor stats in a single pass if possible.  
5. **Memory efficiency**: Process year by year (28 chunks) to keep memory manageable.  

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell ids in consistent order
# rook_neighbors_unique: spdep::nb object

# Convert to data.table
setDT(cell_data)

# Build edge list of neighbors
edges <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique)
)

# Variables to compute
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare result container
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nb_max") := NA_real_]
  cell_data[, paste0(v, "_nb_min") := NA_real_]
  cell_data[, paste0(v, "_nb_mean") := NA_real_]
}

# Process year by year to limit memory
years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset to this year
  dt_year <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  
  # Join neighbor values
  dt_edges <- merge(edges, dt_year, by.x = "neighbor_id", by.y = "id", allow.cartesian = TRUE)
  
  # Compute stats for each id
  stats <- dt_edges[, lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0) return(c(NA, NA, NA))
    c(max(x), min(x), mean(x))
  }), by = id, .SDcols = neighbor_source_vars]
  
  # Reshape stats into wide format
  stat_cols <- setdiff(names(stats), "id")
  # stats currently has lists, so unnest
  expanded <- stats[, c("id", unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))
  }))) := transpose(lapply(.SD, function(lst) do.call(rbind, lst))), .SDcols = stat_cols]
  
  # Merge back to main table
  cell_data[year == yr, (names(expanded)[-1]) := expanded[match(id, expanded$id), -1]]
}

# cell_data now has neighbor stats columns added
```

---

**Why This Works**  
- **No giant lists**: We compute neighbor stats via grouped aggregation, not row-wise iteration.  
- **Chunking by year**: Only ~344k rows in memory at once, manageable on 16â€¯GB RAM.  
- **Vectorized aggregation**: `data.table` handles millions of joins and group operations efficiently.  
- **Preserves estimand**: Same max, min, mean neighbor values as original code.  
- **Estimated runtime**: Minutes to a few hours instead of 86+ hours.  

This approach is practical for a standard laptop and avoids retraining the Random Forest model.
END_RESPONSE>>>
