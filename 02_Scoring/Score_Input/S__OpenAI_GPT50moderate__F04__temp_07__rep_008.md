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
run_id: OpenAI_F04_T07_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The main bottleneck is the repeated use of `lapply` over 6.46M rows and repeated lookups with string concatenation (`paste`) and name-based indexing inside `build_neighbor_lookup` and `compute_neighbor_stats`. This is extremely inefficient for a large panel because it performs millions of character-based hash lookups and list allocations. Random Forest prediction is fast in comparison.

---

**Optimization Strategy**  
1. **Precompute integer-based neighbor indices for all rows**:
   - Avoid string concatenations in loops.
   - Use matrix-based indexing instead of lists of character keys.
2. **Vectorize neighbor statistics computation**:
   - Flatten neighbor relations into a long table (row_id â†’ neighbor_id).
   - Use `data.table` grouping to compute `max`, `min`, `mean` for each variable.
3. **Process all neighbor source variables in one pass** instead of looping over them repeatedly.

This approach reduces overhead from ~6.46M Ã— N operations to a few grouped aggregations, leveraging fast C-level `data.table` methods.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# Flatten neighbor relationships
# rook_neighbors_unique: list of neighbor ids per cell (length = number of unique cells)
# id_order: vector of cell ids in same order as rook_neighbors_unique
neighbors_dt <- rbindlist(
  lapply(seq_along(id_order), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      id = id_order[i],
      neighbor_id = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Expand for all years (28 years)
years <- unique(dt$year)
neighbors_dt <- neighbors_dt[, .(id = rep(id, each = length(years)),
                                 neighbor_id = rep(neighbor_id, each = length(years)),
                                 year = rep(years, times = .N))]

# Add row ids for fast join
setkey(neighbors_dt, neighbor_id, year)
setkey(dt, id, year)

# Join to bring neighbor values for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Melt neighbor values for all source variables
neighbor_vals <- dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_vals, "id", "neighbor_id")
setkey(neighbor_vals, neighbor_id, year)

neighbors_dt <- neighbor_vals[neighbors_dt]

# Compute stats grouped by (id, year) for each variable
neighbors_long <- melt(neighbors_dt,
                       id.vars = c("id", "year"),
                       measure.vars = neighbor_source_vars,
                       variable.name = "var",
                       value.name = "val")

stats_dt <- neighbors_long[!is.na(val),
                            .(max_val = max(val),
                              min_val = min(val),
                              mean_val = mean(val)),
                            by = .(id, year, var)]

# Reshape back to wide format and merge into main data
stats_wide <- dcast(stats_dt, id + year ~ var, value.var = c("max_val", "min_val", "mean_val"))
dt <- merge(dt, stats_wide, by = c("id", "year"), all.x = TRUE)

cell_data <- as.data.frame(dt)
```

---

**Expected Gains**  
- Eliminates per-row `lapply` and string-based lookups.
- Uses efficient `data.table` joins and aggregations.
- Processes all variables in a single grouped operation.
- On a standard laptop, this approach should reduce runtime from **86+ hours to under 1 hour** for 6.46M rows, assuming enough memory for intermediate tables (~16 GB should suffice with `data.table` efficiency).

This preserves the original Random Forest model and the numerical estimand while dramatically improving performance.
END_RESPONSE>>>
