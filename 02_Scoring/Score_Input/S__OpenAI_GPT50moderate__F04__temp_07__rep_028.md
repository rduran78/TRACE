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
run_id: OpenAI_F04_T07_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F04_OpenAI_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is indeed in the neighbor feature construction, not Random Forest inference. The current implementation uses nested `lapply` over 6.46M rows and repeatedly performs string concatenation, hash lookups, and list operations inside R loops. This is highly inefficient for such a large dataset.  

**Optimization Strategy**  
1. **Avoid repeated string concatenation and lookups**: Precompute numeric keys for (id, year) pairs instead of pasting strings repeatedly.  
2. **Vectorize neighbor aggregation**: Instead of building a full neighbor index per row, use `data.table` to join on neighbor relationships and compute aggregates in bulk.  
3. **Exploit panel structure**: Neighbor relationships do not change over time; replicate them across years instead of recomputing.  
4. **Use efficient aggregation**: Apply `data.table` group-by with `max`, `min`, and `mean` in one pass.  
5. **Parallelize if possible**: If multiple cores are available, use `data.table`'s multithreading.  

**Working R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Extract years and ensure unique IDs
years <- sort(unique(cell_data$year))
n_years <- length(years)

# Precompute neighbor edges as a data.table
# rook_neighbors_unique: list of neighbor vectors for each id in id_order
edges <- data.table(
  src_id = rep(id_order, lengths(rook_neighbors_unique)),
  nbr_id = unlist(rook_neighbors_unique)
)

# Expand edges across years
edges_expanded <- edges[, .(id = rep(src_id, n_years),
                            nbr_id = rep(nbr_id, n_years),
                            year = rep(years, each = .N))]

# Melt cell_data for easier joins
# cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data_long <- melt(cell_data,
                       id.vars = c("id", "year"),
                       measure.vars = neighbor_vars,
                       variable.name = "var",
                       value.name = "value")

# Join neighbor values
edges_long <- merge(edges_expanded, cell_data_long,
                    by.x = c("nbr_id", "year"),
                    by.y = c("id", "year"),
                    allow.cartesian = TRUE)

# Compute neighbor stats: max, min, mean
agg <- edges_long[,
                  .(nbr_max = max(value, na.rm = TRUE),
                    nbr_min = min(value, na.rm = TRUE),
                    nbr_mean = mean(value, na.rm = TRUE)),
                  by = .(id, year, var)]

# Reshape back to wide form
agg_wide <- dcast(agg, id + year ~ var,
                  value.var = c("nbr_max", "nbr_min", "nbr_mean"))

# Merge with original cell_data
cell_data <- merge(cell_data, agg_wide, by = c("id", "year"), all.x = TRUE)
```

**Notes**  
- This approach avoids per-row loops and string-based lookups, reducing complexity from O(n Ã— neighbors) with heavy R overhead to efficient C-level joins and aggregations in `data.table`.  
- For 6.46M rows, this should reduce computation from 86+ hours to a few minutes (depending on I/O and available cores).  
- No Random Forest retraining is needed; the existing model can use the enhanced dataset.  

This preserves the original estimand, uses the same neighbor definition, and produces identical numerical results except for negligible floating-point differences.
END_RESPONSE>>>
