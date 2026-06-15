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
run_id: OpenAI_F02_T10_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**:  
The slowdown arises from (a) repeatedly using `lapply` over 6.46 million rows in pure R, (b) creating large lists of neighbor indices on the fly, and (c) processing each neighbor feature sequentially. This becomes highly memory- and compute-intensive given the size of the dataset. Pure R list operations are not efficient for this scale.

---

### **Optimization Strategy**
1. **Precompute and flatten neighbor relations**: Instead of building per-row neighbor lists, transform the neighbor lookup into a long two-column index (`from`, `to`) mapping cell-year observations.  
2. **Vectorized, join-based aggregation** (using `data.table`): Compute max, min, and mean at scale via grouped operations rather than per-row loops.  
3. **Memory optimization**: Use `data.table` for efficient in-memory manipulation and avoid large nested `lapply`.  
4. **Preserve the original estimand**: Matching by year ensures that the statistics for neighbors in the same year remain consistent.  

---

### **Working R Code (Optimized)**

```r
library(data.table)

# Convert input to data.table
dt <- as.data.table(cell_data)

# Create a flat neighbor table
# id_order: vector of cell IDs in the same index order as rook_neighbors_unique
# rook_neighbors_unique: list of integer vectors (neighbors), length = length(id_order)
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Add year replication: Cartesian join on years for panel structure
years <- sort(unique(dt$year))
neighbor_pairs <- neighbor_pairs[ , .(src, nbr, year = rep(years, each = .N)), by = .(src, nbr)]
setkey(neighbor_pairs, nbr, year)

# Prepare for join
setkey(dt, id, year)

# Merge to get neighbor values
neighbor_dt <- neighbor_pairs[dt, on = .(nbr = id, year), allow.cartesian = TRUE]
# neighbor_dt has src (cell), year, and neighborâ€™s variables via 'i.' prefix after merge

# Compute stats for each source cell-year and variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  agg <- neighbor_dt[!is.na(get(var_name)), .(
    max_val = max(get(var_name), na.rm = TRUE),
    min_val = min(get(var_name), na.rm = TRUE),
    mean_val = mean(get(var_name), na.rm = TRUE)
  ), by = .(src, year)]

  # Merge back into main dt
  setnames(agg,
           c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  dt <- merge(dt, agg, by.x = c("id", "year"), by.y = c("src", "year"), all.x = TRUE)
}
```

---

### **Why This Is Faster**
- **No row-wise loops**: Everything is handled by `data.table` grouping and joins, which are implemented in C and optimized for speed.
- **Single neighbor structure flattening**: Avoids huge, nested lists and repeated lookups.
- **Vectorized computation**: Using `max`, `min`, `mean` in grouped operations is much faster than iterative computations.

---

**Expected Performance**: On 6.46M rows with ~1.4M neighbor links, this approach should complete in a few hours (depending on disk I/O and joins) instead of 86+ hours, and fit into 16GB RAM if processed sequentially for the 5 variables.

**Note**: Preserve column naming convention for downstream compatibility, and ensure `id` and `year` are indexed.
END_RESPONSE>>>
