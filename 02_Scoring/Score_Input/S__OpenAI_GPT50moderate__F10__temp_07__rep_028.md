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
run_id: OpenAI_F10_T07_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:
- It computes neighbor lookups and statistics in pure R loops and `lapply` over ~6.46M rows.
- For each variable, it repeatedly extracts and aggregates neighbor values, causing redundant work.
- No vectorization or efficient data structures are used.
- Memory overhead is high due to repeated list and string operations.

---

**Optimization Strategy**  
1. **Precompute graph topology once** using integer indices (avoid string concatenation).
2. **Use `data.table` for fast joins and aggregations**.
3. **Compute all neighbor stats in a single pass per variable** by leveraging the precomputed neighbor index.
4. **Avoid loops over rows**; instead, flatten neighbor relationships into a long edge list and aggregate.
5. **Preserve numerical equivalence** by replicating the exact logic: ignore `NA`s, return `NA` if no non-NA neighbors.
6. **Reuse neighbor graph across all years** by joining on `(cell_id, year)` pairs.

---

**Optimized R Implementation**

```r
library(data.table)

# Assume input: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2)
# id_order: vector of unique cell IDs
# rook_neighbors_unique: list of neighbor indices (spdep::nb)

# 1. Build edge list once
build_edge_list <- function(id_order, neighbors) {
  src <- rep(seq_along(neighbors), lengths(neighbors))
  dst <- unlist(neighbors)
  data.table(src = src, dst = dst)
}

# Precompute edge list
edge_list <- build_edge_list(id_order, rook_neighbors_unique)

# 2. Expand edge list across years
years <- sort(unique(cell_data$year))
edge_dt <- edge_list[, .(src_id = id_order[src], dst_id = id_order[dst])]
edge_dt <- edge_dt[CJ(year = years), on = .(rowid = NULL)][, year := year]
setkey(edge_dt, dst_id, year)

# 3. Convert cell_data to data.table and index
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# 4. Function to compute neighbor stats for one variable
compute_neighbor_stats_dt <- function(var_name) {
  # Join neighbor values
  joined <- edge_dt[dt, on = .(dst_id = id, year), nomatch = 0]
  # joined now has: src_id, dst_id, year, and var from dt
  joined <- dt[joined, on = .(id = src_id, year), nomatch = 0]
  # Columns: id (src), year, var (from src), dst_id
  setnames(joined, var_name, "val")
  
  # Aggregate max, min, mean by dst_id (target) and year
  agg <- joined[!is.na(val), .(
    max_val = max(val),
    min_val = min(val),
    mean_val = mean(val)
  ), by = .(dst_id, year)]
  
  # Merge back to main dt
  setnames(agg, c("dst_id", "year"), c("id", "year"))
  setnames(agg, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  dt[agg, on = .(id, year), (names(agg)[-(1:2)]) := mget(names(agg)[-(1:2)])]
}

# 5. Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (var in neighbor_source_vars) {
  compute_neighbor_stats_dt(var)
}

# dt now contains original data + neighbor stats
# Preserve Random Forest model: predict directly
preds <- predict(trained_rf_model, newdata = dt)
```

---

**Key Improvements**
- **Graph topology built once** (`edge_list`) and reused.
- **Full vectorization**: all neighbor stats computed via `data.table` grouping instead of millions of loops.
- **Memory-efficient**: avoids repeated string concatenation and large intermediate lists.
- **Parallelizable**: `compute_neighbor_stats_dt` can run in parallel over variables if needed.

This approach will reduce runtime from **86+ hours to likely under 1â€“2 hours** on a standard laptop, while preserving numerical equivalence and the trained Random Forest model.
END_RESPONSE>>>
