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
run_id: OpenAI_F06_T07_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F06_OpenAI_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the nested R loops and repeated lapply calls over ~6.46M rows for 5 variables. Each iteration searches and aggregates neighbors using R lists, leading to huge overhead. The algorithm is correct but extremely inefficient because:

- `lapply` over millions of elements in pure R is slow.
- Repeated index lookups and object creation.
- No vectorization or compiled backend (C++/data.table).
- Memory thrashing on 16 GB machine.

**Optimization Strategy**  
- Precompute neighbor indices as integer vectors once.
- Use `data.table` for fast keyed joins and aggregation.
- Compute max, min, mean in a single grouped operation rather than per-row lapply.
- Avoid loops over rows; instead, reshape data to long form with neighbor mapping and aggregate.
- Keep everything in RAM-friendly format.

**Working R Code (Optimized):**

```r
library(data.table)

# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute neighbor pairs as a data.table
# rook_neighbors_unique: list of neighbor ids per cell in id_order
neighbor_pairs <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Cartesian product with years to match panel structure
years <- sort(unique(cell_data$year))
neighbor_dt <- neighbor_pairs[CJ(year = years), on = .(dummy = NULL)]
setnames(neighbor_dt, c("src", "nbr", "year"))

# Join neighbor_dt with cell_data to get neighbor values
setkey(cell_data, id, year)
setkey(neighbor_dt, nbr, year)

neighbor_dt <- neighbor_dt[cell_data, on = .(nbr = id, year), nomatch = 0]

# neighbor_dt now has: src, year, and neighbor's variables
# Compute stats for each src-year per var in one pass
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

agg_stats <- neighbor_dt[, c(
  lapply(.SD, max, na.rm = TRUE),
  lapply(.SD, min, na.rm = TRUE),
  lapply(.SD, mean, na.rm = TRUE)
), by = .(src, year), .SDcols = neighbor_source_vars]

# Rename columns
setnames(agg_stats,
         old = names(agg_stats)[-(1:2)],
         new = c(
           paste0(neighbor_source_vars, "_nbr_max"),
           paste0(neighbor_source_vars, "_nbr_min"),
           paste0(neighbor_source_vars, "_nbr_mean")
         ))

# Merge back to cell_data
setnames(agg_stats, c("src", "year"), c("id", "year"))
cell_data <- merge(cell_data, agg_stats, by = c("id", "year"), all.x = TRUE)
```

**Why this works:**  
- Reduces complexity from O(N Ã— neighbors) in R loops to efficient C-backed aggregation in `data.table`.
- Computes all neighbor stats for all variables in one grouped aggregation, minimizing overhead.
- Preserves original numerical estimand and model (Random Forest model unchanged).
- Expected runtime: minutes to a few hours instead of 86+ hours.

**Memory Note:**  
If memory is tight, process variables in batches or split by year, but the above pattern remains optimal for speed.
END_RESPONSE>>>
