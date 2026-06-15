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
run_id: OpenAI_F12_T07_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F12_OpenAI_temp_07_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The colleague's claim that `do.call(rbind, result)` and repeated list-binding inside `compute_neighbor_stats()` dominate runtime is incorrect. While `rbind` on millions of rows has cost, the primary bottleneck lies in **`build_neighbor_lookup()`**. It does intensive string operations (`paste`) and repeated hash lookups for **6.46M rows Ã— multiple neighbors**, causing extreme overhead before any stats are computed.

`compute_neighbor_stats()` is relatively light: it iterates over the already-built lookup and computes simple aggregations. The repeated binding via `do.call(rbind, ...)` is minor compared to the O(N Ã— neighbors) string concatenations in `build_neighbor_lookup()`.

**Root Cause:**  
`build_neighbor_lookup()` creates neighbor index lists by dynamically generating composite keys (`paste(id, year, ...)`) for every row, then mapping through `idx_lookup`. This is repeated for 6.46M observations, making it the dominant cost.

---

### **Correct Optimization Strategy**
- **Eliminate string concatenation and large hash lookups**: Precompute numeric indices for `(id, year)` pairs and neighbors.
- Build neighbor indices using vectorized joins or integer arithmetic instead of `paste` and hashed lookups.
- Use `data.table` for efficient joins and grouping.
- Compute all neighbor statistics in a single pass over the long table instead of looping per variable.

---

### **Optimized Approach**
1. Represent `cell_data` as `data.table` for fast keyed operations.
2. Precompute a neighbor edge list for all years: `(source_idx, neighbor_idx)`.
3. Join values for all variables and compute aggregates via grouped operations.
4. Avoid building a massive `neighbor_lookup` list.

---

### **Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of ids in consistent order
# rook_neighbors_unique: list of neighbor ids per id

# 1. Build static edge list for spatial neighbors
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0L) return(NULL)
  data.table(src_id = id_order[i], nb_id = id_order[rook_neighbors_unique[[i]]])
}))

# 2. Expand edge list for all years (cross join with unique years)
years <- unique(cell_data$year)
edges_year <- edges[CJ(year = years), on = .(dummy = NULL)]
setnames(edges_year, c("src_id","nb_id","year"))

# 3. Map neighbor-year pairs to row indices via join instead of paste
cell_data_keyed <- copy(cell_data)
setkey(cell_data_keyed, id, year)

edges_year[, src_idx := cell_data_keyed[.(src_id, year), which = TRUE]]
edges_year[, nb_idx  := cell_data_keyed[.(nb_id, year), which = TRUE]]

# Drop any missing indices (no data for that year)
edges_year <- edges_year[!is.na(src_idx) & !is.na(nb_idx)]

# 4. Build long table of neighbor values for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

long_vals <- melt(
  cell_data_keyed[edges_year$nb_idx, ..neighbor_source_vars],
  measure.vars = neighbor_source_vars,
  variable.name = "var",
  value.name = "nb_value"
)[, src_idx := rep(edges_year$src_idx, times = length(neighbor_source_vars))]

# 5. Compute stats per (src_idx, var)
neighbor_stats <- long_vals[
  !is.na(nb_value),
  .(max_nb = max(nb_value), min_nb = min(nb_value), mean_nb = mean(nb_value)),
  by = .(src_idx, var)
]

# 6. Reshape and merge back into cell_data
neighbor_stats_wide <- dcast(
  neighbor_stats,
  src_idx ~ var,
  value.var = c("max_nb", "min_nb", "mean_nb")
)

cell_data <- cbind(cell_data, neighbor_stats_wide[match(seq_len(nrow(cell_data)), src_idx), -1])
```

---

### **Why This Fix Works**
- Removes costly `paste()` and hash lookups in `build_neighbor_lookup()`.
- Vectorizes operations with `data.table` joins and grouped aggregations.
- Processes all variables in one grouped pass rather than looping.
- Scales linearly with rows and neighbors, reducing pipeline from ~86 hours to hours or less on a 16 GB laptop.

**Preserves:**  
- Original estimand.
- Trained Random Forest model (we only update feature engineering step).  
- Numerical equivalence for neighbor stats.  

**Expected speedup:** Orders of magnitude faster due to elimination of O(N Ã— neighbors Ã— string ops) overhead and fully vectorized aggregation.
END_RESPONSE>>>
