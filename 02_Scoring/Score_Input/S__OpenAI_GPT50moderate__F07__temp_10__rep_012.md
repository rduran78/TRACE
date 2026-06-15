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
run_id: OpenAI_F07_T10_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F07_OpenAI_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current approach is extremely slow (86+ hours) because it repeatedly scans large vectors and performs millions of list/lookup operations in Râ€™s interpreted environment. The complexity issue stems from:  
- `lapply` over ~6.46M rows to resolve neighbor indices per observation.  
- Nested lookups using `setNames`-based maps (string concatenation and hashing for every row).  
- For each variable, `compute_neighbor_stats` iterates again through all rows.  
- The algorithm is inherently O(n Ã— avg_degree Ã— vars) in pure R lists and string operations, which is very inefficient for millions of observations.  

**Optimization Strategy:**  
- **Precompute everything in integer space**, avoiding repeated string concatenation and hashing.  
- **Vectorize computations** via data.table or matrix operations.  
- **Unnest neighbor relationships** into a long table and use grouped aggregation (`max`, `min`, `mean`) with fast joins.  
- Compute all neighbor statistics for all variables in a single grouped operation instead of looping per variable.  
- Maintain exact rook adjacency, avoid approximating with naive raster focal ops.  
- Exploit `data.table` for memory-efficient grouping, since 16 GB RAM suffices for ~50â€“100M rows if designed carefully.  

### **Optimized Pipeline**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
cell_data[, cell_year_id := .I]  # unique row id for join later

# Convert rook_neighbors_unique (spdep::nb) adjacency into a long edge list
# id_order: vector of original cell ids in order for rook_neighbors_unique
edge_list <- rbindlist(
  lapply(seq_along(rook_neighbors_unique), function(i) {
    if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
    data.table(
      src = id_order[i],
      nbr = id_order[rook_neighbors_unique[[i]]]
    )
  })
)

# Cartesian join edges with all years to create cell-year neighbor pairs
years <- unique(cell_data$year)
edge_list_expanded <- edge_list[, .(id = src, nbr_id = nbr), ][
  , .(id, nbr_id, year = years), by = .(id, nbr_id)
]

# Map id-year to row index for fast join
cell_index <- cell_data[, .(id, year, cell_year_id)]
edge_list_expanded <- edge_list_expanded[cell_index, on = .(nbr_id = id, year), nomatch=0]
setnames(edge_list_expanded, "cell_year_id", "nbr_row_id")

# Also map focal cellâ€™s row index
edge_list_expanded <- edge_list_expanded[cell_index, on = .(id, year), nomatch=0]
setnames(edge_list_expanded, "cell_year_id", "src_row_id")

# Now edge_list_expanded has src_row_id, nbr_row_id for all valid neighbor pairs
# Join back neighbor values for all 5 variables
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
neighbor_vals <- cell_data[, c("cell_year_id", vars), with = FALSE]
edge_list_expanded <- edge_list_expanded[neighbor_vals, on = .(nbr_row_id = cell_year_id)]

# Compute grouped stats for source rows
agg_stats <- edge_list_expanded[
  , as.list(unlist(lapply(.SD, function(x) c(max = max(x, na.rm=TRUE),
                                             min = min(x, na.rm=TRUE),
                                             mean = mean(x, na.rm=TRUE))))),
  by = src_row_id,
  .SDcols = vars
]

# Clean NA from groups with no neighbors
agg_stats[is.infinite(ntl.max), (names(agg_stats)) := NA]

# Merge back into cell_data
cell_data <- cell_data[agg_stats, on = .(cell_year_id = src_row_id)]
```

**Complexity Improvement:**  
- Previous: per-row neighbor matching + string ops Ã— 6.46M Ã— 5 variables.  
- New: one-time long table join + single grouped aggregation over ~edges Ã— years (â‰ˆ 1.37M edges Ã— 28 yrs â‰ˆ 38M pairs). On a modern laptop, `data.table` can handle 40â€“50M rows group aggregation in under a few hours, often minutes.  
- Memory fits in 16GB because intermediate tables are numeric and integer (avoid strings).  

**Result:**  
- Preserves exact rook neighborhood topology across cells and years.  
- Computes `max`, `min`, and `mean` for all neighbor variables in one pass.  
- No modification to the trained Random Forest model.  
- Same numerical estimand, just faster generation of features.  

**Expected runtime reduction:** From 86+ hours to a few tens of minutes (or ~1â€“2 hours worst case).
END_RESPONSE>>>
