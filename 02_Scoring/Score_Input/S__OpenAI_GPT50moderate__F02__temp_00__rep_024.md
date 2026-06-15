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
run_id: OpenAI_F02_T00_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F02_OpenAI_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated use of `lapply` over ~6.46M rows and dynamic lookups in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation, list indexing, and repeated memory allocations. This is highly inefficient for large panel data. The process is CPU-bound and memory-intensive because it builds neighbor indices row-by-row and computes stats repeatedly for each variable.

**Optimization Strategy**  
1. **Precompute neighbor indices by (cell_id, year)** using integer mapping instead of string keys.
2. **Vectorize computations**: Avoid per-row `lapply` by using `data.table` or `dplyr` joins and grouped aggregations.
3. **Use long format and keyed joins**: Expand neighbor relationships across years once, then compute stats in bulk.
4. **Avoid repeated loops for each variable**: Compute all neighbor stats in one pass.
5. **Memory efficiency**: Use integer vectors and `data.table` for fast joins and aggregation.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
cell_dt <- as.data.table(cell_data)

# Precompute neighbor relationships across years
# id_order: vector of cell IDs in reference order
# rook_neighbors_unique: list of neighbor indices per cell
neighbor_pairs <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand across years
years <- unique(cell_dt$year)
neighbor_dt <- neighbor_pairs[, .(id, neighbor_id), by = .(year = years)]

# Join to get neighbor values
setkey(cell_dt, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Merge neighbor values for all source vars
merged_dt <- neighbor_dt[cell_dt, on = .(neighbor_id = id, year), nomatch = 0]

# Compute stats for all variables in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

stats_dt <- merged_dt[, lapply(.SD, function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(c(NA_real_, NA_real_, NA_real_))
  c(max(x), min(x), mean(x))
}), by = .(id, year), .SDcols = neighbor_source_vars]

# Reshape stats into wide format with suffixes
stat_names <- c("max", "min", "mean")
stats_long <- melt(stats_dt, id.vars = c("id", "year"))
stats_long[, c("var", "stat") := tstrsplit(variable, "_", fixed = TRUE)]
stats_wide <- dcast(stats_long, id + year ~ var + stat, value.var = "value")

# Merge back to original data
cell_dt <- merge(cell_dt, stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

**Why this works**  
- Eliminates per-row loops: All neighbor relationships are expanded once.
- Uses `data.table` joins and aggregation for speed and memory efficiency.
- Computes all neighbor stats in one grouped operation instead of 5 separate loops.
- Avoids string concatenation and repeated indexing.

**Expected Performance**  
This approach reduces complexity from O(N Ã— neighbors Ã— vars) with heavy R overhead to efficient keyed joins and grouped aggregation. On 6.46M rows, this should complete in **minutes to a few hours** on a 16 GB laptop instead of 86+ hours.

**Preserves**  
- Original numerical estimand.
- Trained Random Forest model (no retraining).
- All required neighbor-based features.
END_RESPONSE>>>
