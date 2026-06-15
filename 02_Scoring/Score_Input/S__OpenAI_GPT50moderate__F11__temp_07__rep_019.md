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
run_id: OpenAI_F11_T07_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F11_OpenAI_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleagueâ€™s claim that *Random Forest inference* is the main bottleneck is incorrect. Random Forest prediction on 6.46M rows with 110 features is heavy but not 86+ hours on a laptopâ€”it would typically take minutes to a couple of hours with `ranger` or parallelized `randomForest`.  

The real bottleneck is the neighbor feature computation:
- `build_neighbor_lookup` creates a list of length = number of rows (â‰ˆ 6.46M), each element being a vector of neighbor indices. This is enormous in memory and slow.
- `compute_neighbor_stats` repeatedly scans through these lists for **each variable**, causing repeated large `lapply` and `do.call(rbind, ...)` operations.  
This is `O(N * V * K)` operations (N = 6.46M rows, V = 5 variables, K = avg neighbor count), which is prohibitively expensive.

**Optimization Strategy:**  
- **Avoid per-row `lapply` loops** and expensive list manipulations.
- Use a **long-format join-based approach**: expand neighbor relationships once into a data.table and compute aggregates with fast group-by.
- Compute all neighbor stats in one pass rather than looping over variables.
- Leverage `data.table` for speed and memory efficiency.

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Create unique key for each cell-year
cell_data[, cell_year := paste(id, year, sep = "_")]

# Prepare neighbor relationships in long format
# rook_neighbors_unique is a list of neighbors by cell ID order
id_order_dt <- data.table(id = id_order, ref_idx = seq_along(id_order))
neighbor_pairs <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
  })
)

# Expand neighbor pairs across all years
years <- unique(cell_data$year)
neighbor_pairs_expanded <- neighbor_pairs[
  , .(id = cell_id, neighbor = neighbor_id), by = 1:nrow(neighbor_pairs)
][, .(id, neighbor, year = rep(years, each = .N)), by = .(id, neighbor)]

# Add keys for join
neighbor_pairs_expanded[, id_year := paste(id, year, sep = "_")]
neighbor_pairs_expanded[, neighbor_year := paste(neighbor, year, sep = "_")]

# Join to get neighbor values
setkey(cell_data, cell_year)
setkey(neighbor_pairs_expanded, neighbor_year)

# For memory efficiency, select only needed columns
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_vals <- neighbor_pairs_expanded[cell_data, on = .(neighbor_year = cell_year), nomatch = 0]
# neighbor_vals: columns id, neighbor, year, id_year, and value columns from cell_data

# Melt to long for aggregation
neighbor_long <- melt(neighbor_vals,
                      id.vars = c("id_year"),
                      measure.vars = vars,
                      variable.name = "var_name",
                      value.name = "val")

# Remove NA values
neighbor_long <- neighbor_long[!is.na(val)]

# Compute max, min, mean per id_year and var_name
agg_stats <- neighbor_long[, .(
  max_val = max(val),
  min_val = min(val),
  mean_val = mean(val)
), by = .(id_year, var_name)]

# Reshape wide to join back
agg_wide <- dcast(agg_stats, id_year ~ var_name, value.var = c("max_val", "min_val", "mean_val"))

# Merge back into cell_data
cell_data <- merge(cell_data, agg_wide, by.x = "cell_year", by.y = "id_year", all.x = TRUE)
```

### **Why This Works**
- We **materialize neighbor relationships once** in a long data.table instead of per-row lists.
- We compute all stats in **bulk group-by operations**, which are highly optimized in `data.table`.
- This reduces complexity from `O(N * V * K)` explicit loops to efficient joins and aggregations.

**Expected runtime:** From 86+ hours down to well under 1 hour on 6.46M rows with `data.table` (depending on I/O and RAM).

**Preserves:**  
- Original trained Random Forest model (unchanged).  
- Numerical estimands (stats computed identically: max, min, mean).  

**Next step:** After this optimization, call `predict()` once on the enriched `cell_data` table.
END_RESPONSE>>>
