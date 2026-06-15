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
run_id: OpenAI_F12_T10_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F12_OpenAI_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The colleagueâ€™s claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottlenecks is incorrect. The true bottleneck lies in repeatedly performing millions of neighbor lookups and recomputing statistics in pure R loops across **6.46 million** rows. This results in heavy interpreter overhead and memory allocations. The join operations and neighbor aggregation should be vectorized or delegated to efficient backends.  

**Optimization Strategy**  
- Precompute all neighbor relationships as a sparse matrix or edge list.
- Avoid repeated neighbor traversals by converting the panel data into a `data.table` and using fast keyed joins.
- Aggregate neighbor statistics using highly optimized grouping operations (`data.table` or matrix operations) instead of millions of small `lapply` calls.
- Compute all 5 variables in a single pass rather than looping over them.

**Optimized Approach in R (data.table)**  
```r
library(data.table)

# Convert cell_data to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Build neighbor edge list: each row = (source_id, year, neighbor_id)
lookup_dt <- build_neighbor_lookup_dt(id_order, rook_neighbors_unique)

# Expand to cell-year pairs and join to get neighbor rows
edge_dt <- lookup_dt[cell_data, on = .(id_source = id)] # id_source-year aligned
edge_dt[, year := i.year]
edge_dt <- edge_dt[cell_data, on = .(id_neighbor = id, year), nomatch = 0]

# Compute all stats in one grouped operation
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
agg_dt <- edge_dt[, lapply(.SD, function(x) {
    list(max = max(x, na.rm = TRUE),
         min = min(x, na.rm = TRUE),
         mean = mean(x, na.rm = TRUE))
}), by = .(id_source, year), .SDcols = neighbor_vars]

# Reshape and merge back
long_agg <- melt(agg_dt, id.vars = c("id_source", "year"),
                 variable.name = "var", value.name = "stats")
long_agg <- dcast(long_agg, id_source + year ~ var,
                  value.var = "stats", fun.aggregate = identity)

setnames(long_agg, "id_source", "id")
cell_data <- long_agg[cell_data, on = .(id, year)]
```

**Why This is Faster**  
- Eliminates millions of `lapply` calls.
- Leverages `data.table` grouping and joins (highly optimized in C).
- Handles all neighbor source variables in one pass.
- Reduces intermediate allocations.

This approach should cut runtime from **86+ hours to a few hours or less** on a standard laptop while preserving the trained Random Forest model and original estimands.
END_RESPONSE>>>
