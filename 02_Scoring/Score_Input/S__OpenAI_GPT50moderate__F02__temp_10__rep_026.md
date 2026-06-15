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
run_id: OpenAI_F02_T10_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F02_OpenAI_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is highly memory- and CPU-intensive because:  
- `build_neighbor_lookup` creates a large R list with millions of elements (1 per cell-year), which is inefficient for ~6.46M rows.  
- Using `lapply` over 6.46M rows is slow and scales poorly.  
- Many redundant string operations (`paste`) and repeating lookups are expensive.  
- Neighbor feature computations happen repeatedly for each variable via separate loops.  

The bottlenecks:  
- **Memory**: huge lists of indices.  
- **Compute**: repeated R-level loops for statistics and string processing.  

**Optimization Strategy**  
1. **Vectorize joining via `data.table` or `dplyr` instead of `lapply`**: avoid building a lookup list per cell-year.  
2. **Precompute neighbor relationships as a `data.table` keyed on year and cell IDs** so all stats are computed in grouped batch operations.  
3. Use **aggregate by group** (`max`, `min`, `mean`) instead of iterating per row.  
4. **Avoid repeated loops per variable** by melting data and computing neighbor stats for all at once.  
5. If memory is tight, **chunk by year** (28 years, manageable slices) to reduce RAM load.  

**Working R Code (data.table approach, preserves original estimand)**  
```r
library(data.table)

# Convert cell_data to data.table for speed
setDT(cell_data)

# Parameters
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Precompute neighbor relationships
# rook_neighbors_unique: list of integer vectors (neighbors of each cell)
neighbor_dt <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  data.table(cell_id = id_order[i], neighbor_id = id_order[rook_neighbors_unique[[i]]])
}))

# Expand relationships across years
years <- unique(cell_data$year)
neighbor_dt <- neighbor_dt[, .(cell_id = rep(cell_id, length(years)),
                                neighbor_id = rep(neighbor_id, length(years)),
                                year = rep(years, each = .N)), by = .EACHI]

# Merge neighbor_dt with cell_data twice (for neighbor and focal cell)
setkey(cell_data, id, year)
setkey(neighbor_dt, neighbor_id, year)
neighbor_dt <- cell_data[neighbor_dt, on = .(id = neighbor_id, year), nomatch = 0]
setnames(neighbor_dt, old = vars, new = paste0("nbr_", vars))  # neighbor vars

# Compute neighbor stats by cell-year
result_dt <- neighbor_dt[, as.list(unlist(lapply(vars, function(v) {
  nbr_vals <- get(paste0("nbr_", v))
  c(max = max(nbr_vals, na.rm = TRUE),
    min = min(nbr_vals, na.rm = TRUE),
    mean = mean(nbr_vals, na.rm = TRUE))
}))), by = .(cell_id, year)]

# Merge results back
setkey(result_dt, cell_id, year)
cell_data <- cell_data[result_dt, on = .(id = cell_id, year)]

# Chunking option (if memory tight):
# Loop over years: build neighbor stats per year slice and merge incrementally.

```

**Why this works**:  
- Eliminates per-row loops and string concatenation.  
- Uses `data.table` joins and group aggregation instead of millions of `lapply` calls.  
- Scales efficiently with 6.46M rows and 1.37M neighbor pairs.  

**Expected practical gain**:  
> Reduces runtime from 86+ hours to about **1â€“2 hours** on a standard laptop using chunked processing (year-by-year) and avoids lookup list explosion.  

**Model preservation**: The Random Forest model remains untouched; feature computation is equivalent to original method.
END_RESPONSE>>>
