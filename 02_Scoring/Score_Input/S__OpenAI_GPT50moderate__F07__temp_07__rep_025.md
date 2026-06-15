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
run_id: OpenAI_F07_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F07_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- **Inefficient per-row processing**: Both `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M rows, creating millions of small R objects and repeated list indexing.  
- **Repeated computations per variable**: For each of 5 variables, the neighbor stats are recomputed from scratch rather than leveraging vectorized or compiled operations.  
- **Memory overhead**: Storing neighbor lookups as large lists and repeatedly subsetting vectors incurs significant RAM and GC pressure.  
- **Lack of compiled/parallelized routines**: Pure R loops on millions of rows are inherently slow.  

---

**Optimization Strategy**  
1. **Precompute neighbor relationships as integer vectors**: Flatten the neighbor lookup into two integer vectors (`from`, `to`) representing directed edges across all cell-year instances.  
2. **Join-based aggregation**: Use `data.table` or `dplyr` to compute `max`, `min`, `mean` for each origin cell-year by joining neighbor values and aggregating in a single pass per variable.  
3. **Leverage `data.table` for speed**: `data.table` is memory-efficient and optimized for grouped aggregation on large datasets.  
4. **One-pass per variable**: Compute all neighbor stats for a variable in one grouped aggregation instead of millions of small operations.  
5. **Preserve estimands**: Ensure the neighbor mapping respects exact rook topology and years.  

---

**Working R Code (Optimized)**  
```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Flatten neighbor relationships for all years
# id_order: vector of unique cell ids in rook_neighbors_unique order
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build pair list (cell_id, neighbor_id) once
neighbor_pairs <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(from = id_order[i], to = id_order[rook_neighbors_unique[[i]]])
}))

# Expand to cell-year level using join on years
years <- unique(cell_data$year)
neighbor_dt <- CJ(year = years, from = neighbor_pairs$from)[
  , to := neighbor_pairs$to[.I %% nrow(neighbor_pairs) + 1L]] # replicate mapping
# Above uses Cartesian join carefully; better approach:
neighbor_dt <- merge(
  CJ(year = years, from = unique(neighbor_pairs$from)),
  neighbor_pairs,
  by = "from",
  allow.cartesian = TRUE
)

# Add keys for fast join
setkey(cell_data, id, year)
setkey(neighbor_dt, to, year)

# Compute neighbor stats for each variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  neighbor_vals <- cell_data[, .(to = id, year, val = get(var_name))]
  setkey(neighbor_vals, to, year)
  
  merged <- neighbor_dt[neighbor_vals, on = .(to, year)]
  
  # Aggregate by (from, year)
  stats <- merged[!is.na(val),
                  .(max_val = max(val), min_val = min(val), mean_val = mean(val)),
                  by = .(id = from, year)]
  
  # Merge back to cell_data
  setkey(stats, id, year)
  cell_data <- stats[cell_data]
  
  # Rename columns
  setnames(cell_data,
           old = c("max_val", "min_val", "mean_val"),
           new = paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
}

# cell_data now contains neighbor features for all 5 variables
```

---

**Why This is Fast**  
- Avoids per-row `lapply`; uses vectorized joins and group aggregation.  
- Single pass per variable instead of millions of inner loops.  
- Efficient memory handling with `data.table`.  

**Expected Runtime**  
On a 16 GB laptop, this approach should complete in **minutes to a few hours**, not 86 hours, given efficient joins and aggregation in C.  

**Preserves**  
- Exact rook-neighbor relationships  
- Original numerical estimands (max, min, mean)  
- Trained Random Forest model remains untouched
END_RESPONSE>>>
