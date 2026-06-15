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
run_id: OpenAI_F12_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` is the main bottleneck is incorrect. While `rbind` on a large list (â‰ˆ6.46M rows) is costly, the dominant bottleneck lies in **recomputing neighbor statistics via nested `lapply` over millions of rows for each variable** (5 passes over 6.46M rows = ~32M neighbor lookups), performing repeated indexing and filtering for NA values. This is an **O(n Ã— k)** pattern (n = 6.46M, k = mean neighbor count), and the cost grows multiplicatively with the number of variables.  

The expensive part is:  
```r
lapply(neighbor_lookup, function(idx) { ... vals[idx] ... })
```
repeated for each variable. This performs billions of random-access lookups and multiple allocations, far outweighing the cost of the final `rbind`.

---

### Correct Optimization Strategy  
- **Pre-flatten neighbor relationships into an edge list** (cell-year â†’ neighbor cell-year) once, avoiding repeated per-row neighbor discovery.
- **Compute all neighbor-derived stats in a single grouped aggregation** using `data.table` or `dplyr`, taking advantage of vectorized operations and grouping instead of millions of small loops.
- This avoids repeated passes over 6.46M rows and replaces nested `lapply` with efficient joins and aggregations.

---

### Optimized Approach (data.table)

```r
library(data.table)

# Convert data to data.table
setDT(cell_data)

# Create a unique key for each cell-year
cell_data[, cell_year := paste(id, year, sep = "_")]

# Flatten neighbor relationships into an edge list for all years
# rook_neighbors_unique: list of neighbor IDs for each id in id_order
id_to_neighbors <- rook_neighbors_unique
edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(id_to_neighbors[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[id_to_neighbors[[i]]]
  )
}))

# Expand edge list for all years and join indices
years <- unique(cell_data$year)
edge_list <- edge_list[, .(id = rep(id, length(years)),
                            neighbor_id = rep(neighbor_id, length(years)),
                            year = rep(years, each = .N))]
# Join to get cell_year keys
edge_list[, cell_year := paste(id, year, sep = "_")]
edge_list[, neighbor_cell_year := paste(neighbor_id, year, sep = "_")]

# Map neighbor_cell_year to data rows
lookup <- cell_data[, .(cell_year, row_id = .I)]
edge_list <- merge(edge_list, lookup, by.x = "neighbor_cell_year", by.y = "cell_year", all.x = TRUE, allow.cartesian = TRUE)
edge_list <- merge(edge_list, lookup, by.x = "cell_year", by.y = "cell_year", suffixes = c("_neighbor", "_origin"), all.x = TRUE)

# Drop missing neighbors
edge_list <- edge_list[!is.na(row_id_neighbor)]

# Compute all required stats in one pass
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Gather neighbor values for each variable and compute stats
for (var in neighbor_vars) {
  edge_list[, (var) := cell_data[[var]][row_id_neighbor]]
}

# Compute summary stats per origin row
agg <- edge_list[, .(
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
), by = row_id_origin]

# Merge back into cell_data
setnames(agg, "row_id_origin", "row_id")
cell_data <- merge(cell_data, agg, by.x = ".I", by.y = "row_id", all.x = TRUE)
```

---

### Why This Works  
- **Single pass aggregation** instead of 5 Ã— 6.46M nested loops.
- Avoids repeated indexing and repeated neighbor lookups.
- `data.table` optimizes joins and group operations in C.
- Memory footprint minimized by edge list representation and vectorized stats.

---

**Expected Performance Gain**: Reduces runtime from 86+ hours to a few hours or less on a 16 GB RAM laptop, depending on I/O and join size, without changing the trained Random Forest model or the estimand.
END_RESPONSE>>>
