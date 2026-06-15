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
run_id: OpenAI_F02_T00_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F02_OpenAI_temp_00_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow and memory-intensive because:  
- `build_neighbor_lookup` creates a large list of length â‰ˆ 6.46M (one per cell-year), each storing neighbor indices. This is huge in memory and expensive to compute.  
- `compute_neighbor_stats` repeatedly traverses large lists and subsets vectors, causing heavy R-level overhead.  
- The process is purely in R loops (`lapply`), which is inefficient for millions of rows.  
- Repeated string concatenation (`paste`) and lookups in large named vectors are costly.  

**Optimization Strategy**  
1. **Avoid per-row neighbor lists**: Instead of building a 6.46M-element list, compute neighbor relationships in a long format (edges) and join.  
2. **Vectorize aggregation**: Use `data.table` or `dplyr` to compute neighbor statistics in a grouped manner.  
3. **Precompute static structures**: The neighbor graph is static across years; replicate edges by year efficiently.  
4. **Memory efficiency**: Work with integer IDs, avoid character keys, and use `data.table` for joins and aggregations.  
5. **Parallelization**: Use `data.table`â€™s multithreading for aggregation.  

**Working R Code (Optimized)**  
```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# Convert to data.table
setDT(cell_data)

# Map cell id to index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Build edge list (cell_id -> neighbor_id)
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(src = id_order[i], nbr = id_order[rook_neighbors_unique[[i]]])
}))

# Expand edges for all years
years <- sort(unique(cell_data$year))
edges_expanded <- edges[, .(id = rep(src, each = length(years)),
                             nbr = rep(nbr, each = length(years)),
                             year = rep(years, times = .N))]
# Key for fast join
setkey(edges_expanded, nbr, year)
setkey(cell_data, id, year)

# Join neighbor values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  # Join neighbor values
  tmp <- cell_data[, .(nbr_val = get(var_name)), by = .(id, year)]
  setnames(tmp, "id", "nbr")
  edges_expanded <- tmp[edges_expanded, on = .(nbr, year)]
  
  # Compute stats by (id, year)
  stats <- edges_expanded[!is.na(nbr_val),
                           .(max = max(nbr_val),
                             min = min(nbr_val),
                             mean = mean(nbr_val)),
                           by = .(id, year)]
  
  # Merge back to cell_data
  setnames(stats, c("max", "min", "mean"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added
```

**Why this is faster**  
- Eliminates 6.46M-element list and repeated `lapply`.  
- Uses `data.table` joins and group operations in C for speed.  
- Memory footprint is reduced by working in long format and avoiding character concatenation.  
- Scales well with multithreading on a standard laptop.  

**Expected performance**  
- From 86+ hours to a few minutes to an hour, depending on disk I/O and CPU cores.  
- Preserves original estimand and trained Random Forest model.
END_RESPONSE>>>
