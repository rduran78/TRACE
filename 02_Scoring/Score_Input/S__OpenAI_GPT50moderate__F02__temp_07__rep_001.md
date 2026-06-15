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
run_id: OpenAI_F02_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F02_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The current pipeline is prohibitively slow and memory-heavy because:

- **`build_neighbor_lookup` and `compute_neighbor_stats` rely on `lapply` over 6.46M rows**, creating millions of small vectors and lists, which is inefficient in R.
- Neighbor lookups are recomputed redundantly for each variable.
- Heavy use of dynamic lookup (`setNames`, list indexing, repeated `paste`) inflates memory and CPU.
- The entire process is single-threaded and not vectorized.

  
**Optimization Strategy**

1. **Precompute neighbor relationships once in an efficient structure**: Instead of storing a list for each cell-year, create a long-format table mapping each observation to its neighbors.
2. **Vectorize neighbor stats computation** using `data.table` or `dplyr` joins and grouped aggregations instead of `lapply`.
3. **Process in chunks if memory becomes an issue**.
4. **Leverage fast aggregation** (`data.table` is highly recommended for this size).
5. Preserve the trained model: we only optimize feature engineering.

  
**Optimized Approach**

- Build a long table with columns: `id`, `year`, `neighbor_id`, `neighbor_year` (same year), then join and aggregate in one pass.
- Compute `max`, `min`, `mean` by group for all variables using `data.table`.

  
**Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor relationships in long format
# id_order and rook_neighbors_unique assumed available
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Expand for all years (cross join years with neighbor pairs)
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(year = years), by = .(id, neighbor_id)]

# Merge to get neighbor values
# Create a key for fast joins
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)

# Join neighbor_dt with cell_data to get neighbor values
neighbor_dt <- neighbor_dt[cell_data, on = .(neighbor_id = id, year), nomatch = 0]
# neighbor_dt now has: id, neighbor_id, year, [neighbor vars]

# Compute neighbor stats for selected vars
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_exprs <- lapply(neighbor_source_vars, function(var) {
  list(
    (function(x) max(x, na.rm = TRUE)) = as.name(var),
    (function(x) min(x, na.rm = TRUE)) = as.name(var),
    (function(x) mean(x, na.rm = TRUE)) = as.name(var)
  )
})

# data.table way: melt + aggregate or loop through vars
results_list <- lapply(neighbor_source_vars, function(var) {
  neighbor_dt[, .(
    max_val = max(get(var), na.rm = TRUE),
    min_val = min(get(var), na.rm = TRUE),
    mean_val = mean(get(var), na.rm = TRUE)
  ), by = .(id, year)]
})

# Combine results
# Rename columns appropriately and cbind to cell_data
for (i in seq_along(neighbor_source_vars)) {
  vars <- results_list[[i]]
  setnames(vars,
           c("max_val", "min_val", "mean_val"),
           paste0(neighbor_source_vars[i], c("_nb_max", "_nb_min", "_nb_mean")))
  cell_data <- vars[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features efficiently added
```

  
**Why This is Faster and Feasible on 16 GB RAM**

- Only one join and one aggregation per variable instead of millions of list operations.
- `data.table` uses memory-efficient in-place updates and multi-threaded aggregation (if enabled).
- Eliminates repeated string concatenation and dynamic lookups.
- Scales linearly and can be chunked if needed.

**Expected Runtime Reduction:** From ~86 hours to under an hour on a laptop using `data.table` due to full vectorization and efficient joins.
END_RESPONSE>>>
