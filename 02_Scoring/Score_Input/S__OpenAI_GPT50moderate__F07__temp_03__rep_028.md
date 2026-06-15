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
run_id: OpenAI_F07_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested `lapply` approach over ~6.46M rows, repeatedly subsetting vectors and computing stats. This results in huge R-level overhead and memory churn. The neighbor lookup is precomputed, but the stats computation is entirely interpreted and not vectorized. With 6.46M rows Ã— 5 variables Ã— 3 stats, the naive approach becomes infeasible (86+ hours).

---

**Optimization Strategy**  
1. **Precompute neighbor index list once** (already done).
2. **Avoid repeated R loops**: Use `data.table` for fast grouping and joins or collapse neighbor relationships into a long edge table and aggregate.
3. **Vectorize aggregation**: Instead of looping per row, reshape neighbor relationships into a two-column edge list (`source`, `neighbor`), join values, and compute `max`, `min`, `mean` by `source`.
4. **Process variable-by-variable in chunks** to manage memory.
5. **Preserve estimand**: Same neighbor sets, same stats, just computed efficiently.
6. **Do not retrain model**: Only augment `cell_data` with neighbor features.

---

**Working R Code (Efficient Implementation)**  
```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids
# rook_neighbors_unique: spdep::nb object
# Build edge list once
build_edge_table <- function(id_order, neighbors) {
  src <- rep(id_order, lengths(neighbors))
  dst <- unlist(neighbors, use.names = FALSE)
  data.table(src = src, dst = id_order[dst])
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Expand edge list to panel by joining on year
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Create full edge-year table
edge_year_dt <- cell_dt[, .(src = id, year)][edge_dt, on = .(src), allow.cartesian = TRUE]
edge_year_dt[, dst := i.dst]
edge_year_dt[, i.dst := NULL]

# Join neighbor values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_features <- function(var_name) {
  # Join neighbor values for var_name
  tmp <- edge_year_dt[cell_dt, on = .(dst = id, year), nomatch = 0]
  # tmp now has columns: src, year, dst, var_name
  tmp_vals <- tmp[[var_name]]
  tmp[, val := tmp_vals]
  
  # Aggregate by src, year
  agg <- tmp[, .(
    max_val = max(val, na.rm = TRUE),
    min_val = min(val, na.rm = TRUE),
    mean_val = mean(val, na.rm = TRUE)
  ), by = .(src, year)]
  
  # Handle all-NA case
  agg[is.infinite(max_val), `:=`(max_val = NA, min_val = NA, mean_val = NA)]
  
  # Rename columns
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  
  agg
}

# Compute and merge for each variable
for (var_name in neighbor_source_vars) {
  agg <- compute_neighbor_features(var_name)
  setkey(agg, src, year)
  cell_dt <- agg[cell_dt, on = .(src = id, year)]
}

# cell_dt now has neighbor features added
```

---

**Why This Works**  
- Converts neighbor relationships into a long table and uses `data.table` joins and aggregations (highly optimized C backend).
- Avoids per-row loops and repeated subsetting.
- Memory efficient: processes one variable at a time.
- Preserves original rook-neighbor topology and estimand.
- Expected runtime: **minutes to a few hours** instead of 86+ hours on a 16 GB laptop.

**Key Idea**: Transform the problem from 6.46M nested loops into a single vectorized aggregation over an edge-year table.
END_RESPONSE>>>
