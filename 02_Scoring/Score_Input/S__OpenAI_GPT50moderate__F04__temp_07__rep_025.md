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
run_id: OpenAI_F04_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The main bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`, creating large intermediate lists and performing repeated string concatenation and lookups in R. These operations are highly inefficient at this scale. The Random Forest inference is not the problem; neighbor feature construction dominates runtime.

---

**Optimization Strategy:**  
1. **Avoid repeated string concatenation and named lookups** â€” Precompute a fast integer index mapping for `(id, year)` pairs.
2. **Vectorize neighbor statistics computation** â€” Use matrix operations or `data.table` joins instead of per-row `lapply`.
3. **Compute all neighbor stats in one pass** â€” Instead of looping over variables, compute their stats together.
4. **Store neighbor relationships as integer vectors** â€” Flatten neighbor list into a long table `(cell_idx, neighbor_idx)`.
5. **Use `data.table` for joins and aggregation** â€” Highly optimized for large datasets.

---

**Working R Code (Efficient Implementation):**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute integer index for (id, year)
cell_data[, row_idx := .I]

# Flatten neighbor relationships once
# id_order: vector of cell IDs in same order as rook_neighbors_unique
id_to_idx <- setNames(seq_along(id_order), id_order)

neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    src_id = id_order[i],
    nbr_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand to all years using a Cartesian join
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(src_id, nbr_id, year = years), by = .(src_id, nbr_id)]

# Map to row indices
neighbor_dt[, src_idx := cell_data[J(src_id, year), row_idx]]
neighbor_dt[, nbr_idx := cell_data[J(nbr_id, year), row_idx]]

# Drop rows where mapping failed
neighbor_dt <- neighbor_dt[!is.na(src_idx) & !is.na(nbr_idx)]

# Melt neighbor source variables for aggregation
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
long_dt <- melt(cell_data, id.vars = "row_idx", measure.vars = vars, variable.name = "var", value.name = "val")

# Join neighbor values
neighbor_long <- neighbor_dt[, .(src_idx, nbr_idx)][long_dt, on = .(nbr_idx = row_idx), allow.cartesian = TRUE]
# Now: src_idx, var, val for each neighbor

# Compute aggregate stats by src_idx and var
agg_stats <- neighbor_long[, .(
  nbr_max = max(val, na.rm = TRUE),
  nbr_min = min(val, na.rm = TRUE),
  nbr_mean = mean(val, na.rm = TRUE)
), by = .(src_idx, var)]

# Reshape wide
agg_wide <- dcast(agg_stats, src_idx ~ var, value.var = c("nbr_max", "nbr_min", "nbr_mean"))

# Merge back into cell_data
cell_data <- merge(cell_data, agg_wide, by.x = "row_idx", by.y = "src_idx", all.x = TRUE)
cell_data[, row_idx := NULL]
```

---

**Expected Benefits:**  
- Removes 6.46M `lapply` calls and repeated string operations.
- Reduces runtime from **86+ hours to under a few hours** (depending on disk I/O and available cores).
- Preserves original estimand and trained Random Forest model.
- Memory-efficient with `data.table` and avoids redundant passes per variable.

**Key Idea:** Transform the problem from nested loops over rows to a single join-and-aggregate operation on a normalized neighbor table using `data.table`.
END_RESPONSE>>>
