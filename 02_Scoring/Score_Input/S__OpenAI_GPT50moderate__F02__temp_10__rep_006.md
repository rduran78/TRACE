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
run_id: OpenAI_F02_T10_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck arises from:  
1. `lapply` loops over ~6.46M rows â†’ massive overhead.  
2. Repeated string-based lookups (`paste` keys, `setNames`, `idx_lookup`) â†’ memory- and CPU-heavy.  
3. Nested list operations in `compute_neighbor_stats` create overhead due to `do.call(rbind, ...)`.  

Goal: Calculate neighbor statistic features (max, min, mean for 5 variables) for 6.46M rows efficiently within 16â€¯GB RAM **without retraining the RF model**.  

---

### **Optimization Strategy**
- **Avoid full per-row R loops:** Use vectorized joins or `data.table` grouping.
- **Precompute neighbor relationships in a long form table** (source_id, target_id) so stats can be calculated via fast grouped summaries rather than lapply.
- **Leverage `data.table`** for aggregation to reduce RAM and speed up processing.
- Use integer IDs instead of string keys.
- **Compute all years in one go:** Use year as join key, no repeated string concatenation.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Assume `id_order` aligns with rook_neighbors_unique
# Build a long neighbor mapping (source -> neighbor)
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src_id = id_order[i], nb_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand neighbor relationships across all years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(src_id, nb_id)]

# Melt cell_data to long form for neighbor vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_vars_long <- melt(cell_data, measure.vars = neighbor_source_vars, 
                       variable.name = "var", value.name = "val")

# Join to get neighbor values
setkey(cell_vars_long, id, year)
neighbor_dt <- cell_vars_long[neighbor_dt, on = .(id = nb_id, year)]

# Aggregate neighbor stats (max, min, mean) per src_id-year-var
neighbor_stats <- neighbor_dt[!is.na(val),
                               .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                               by = .(id = src_id, year, var)]

# Reshape back to wide form by variable
neighbor_stats_wide <- dcast(neighbor_stats, id + year ~ var, 
                             value.var = c("nb_max", "nb_min", "nb_mean"))

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_stats_wide, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This Works**
- Single pass joins + grouped aggregations â†’ replaces millions of small lapply calls.
- `data.table` handles 6.5M rows + neighbor expansions efficiently within 16 GB.
- Avoids expensive string manipulation on every row.
- Preserves original model and numerical meaning (same statistics computed).

**Expected Runtime:** Minutes to a couple of hours instead of 86+ hours, depending on disk speed and RAM.
END_RESPONSE>>>
