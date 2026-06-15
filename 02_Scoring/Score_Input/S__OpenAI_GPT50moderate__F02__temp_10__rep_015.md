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
run_id: OpenAI_F02_T10_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

- The current approach uses `lapply` for ~6.46â€¯M rows, causing heavy memory allocations and R object overhead.
- Neighbor index lookup repeatedly constructs vectors and does list-based aggregation for every row, producing extreme inefficiency in both CPU cycles and memory thrashing.
- The process is non-vectorized and not parallelized.
- Each iteration recomputes joins and subsets on a flattened panel instead of pre-computing an indexed mapping.

**Optimization Strategy**

1. **Precompute a flattened neighbor table**: Convert cell-level neighbor relationships into a long `data.table` with `(cell_id, year, neighbor_id, neighbor_index)` for quick joins.
2. **Use `data.table` joins and aggregation**: Move from `lapply` to fully vectorized grouped computations; compute max, min, mean in one pass with `by=`.
3. **Remove repeated character concatenations**: Avoid generating lookup keys for each row by using numeric IDs and joining on integer columns.
4. **Parallelize** where possible: `data.table` operations can run in parallel with `setDTthreads()`.
5. **Compute all neighbor-source vars in a single long-format joinâ€“aggregateâ€“wide pivot cycle instead of looping**.

---

### **Working R Code**

```r
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

#---------------------------------------------------------
# 1. Precompute neighbor pairs (id, neighbor_id)
# rook_neighbors_unique: list of integer vectors (neighbors), names(id_order)
neighbor_pairs <- data.table(
  id = rep(id_order, lengths(rook_neighbors_unique)),
  neighbor_id = unlist(rook_neighbors_unique, use.names = FALSE)
)

#---------------------------------------------------------
# 2. Create all (id, year, neighbor_id) combos by expanding across years
years_vec <- unique(cell_data$year)
neighbor_pairs <- neighbor_pairs[, .(year = years_vec), by = .(id, neighbor_id)]

#---------------------------------------------------------
# 3. Join to get neighbor values
long_data <- melt(cell_data,
                  id.vars = c("id", "year"),
                  measure.vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
                  variable.name = "var_name",
                  value.name = "value")

# Key the tables for fast joins
setkey(neighbor_pairs, neighbor_id, year)
setkey(long_data, id, year)

neighbor_values <- neighbor_pairs[long_data,
                                  on = .(neighbor_id = id, year),
                                  allow.cartesian = TRUE]

# Rename for clarity
neighbor_values[, focal_id := id]
neighbor_values[, id := NULL]

#---------------------------------------------------------
# 4. Aggregate: compute max, min, mean by (focal_id, year, var_name)
agg <- neighbor_values[, .(
  neigh_max = max(value, na.rm = TRUE),
  neigh_min = min(value, na.rm = TRUE),
  neigh_mean = mean(value, na.rm = TRUE)
), by = .(focal_id, year, var_name)]

# Handle case when all NA
agg[is.infinite(neigh_max), neigh_max := NA]
agg[is.infinite(neigh_min), neigh_min := NA]
agg[is.nan(neigh_mean), neigh_mean := NA]

#---------------------------------------------------------
# 5. Reshape back to wide format
agg_wide <- dcast(agg, focal_id + year ~ var_name,
                  value.var = c("neigh_max", "neigh_min", "neigh_mean"))

# Merge back to cell_data
setkey(cell_data, id, year)
setkey(agg_wide, focal_id, year)
cell_data <- agg_wide[cell_data, on = .(focal_id = id, year)]
```

---

### **Key Improvements**

- Switched from per-row `lapply` loops to **vectorized `data.table` joins and aggregation**.
- Avoided repeatedly building lookup keys; used numeric joins.
- Computed all 5 variables in **one pass**, reducing repeated scans.
- Scales efficiently under 16â€¯GB by streaming joins and aggregated grouping.
- Compatible with existing trained Random Forest model: preserves all IDs, years, and numeric estimands.

**Expected runtime drop**: From 86+ hours to about **minutes on a laptop** (depending on disk I/O) due to fully vectorized operations and single-pass aggregation.
END_RESPONSE>>>
