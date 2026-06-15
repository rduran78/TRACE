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
run_id: OpenAI_F10_T03_R030
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 30
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind, ...)`, which is inefficient in R.  
- Neighbor lookups are recomputed per row and per variable.  
- No vectorization; operations are row-wise and loop-heavy.  
- Memory overhead from repeated list-to-matrix conversions.  

**Optimization Strategy**  
- Build the neighbor index **once** and reuse it across all years and variables.  
- Represent the graph as an adjacency list or sparse matrix for efficient aggregation.  
- Use **matrix operations** or **data.table** for grouped computations instead of nested loops.  
- Compute all neighbor stats in a single pass per variable using fast aggregation (e.g., `rowsum`, `data.table` joins).  
- Avoid repeated NA filtering inside loops; handle NA logic in vectorized form.  

**Efficient Implementation in R**  
Below is a fully optimized approach using `data.table` and precomputed adjacency:

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object

# 1. Build adjacency once
build_adjacency <- function(id_order, rook_neighbors_unique) {
  src <- rep(id_order, lengths(rook_neighbors_unique))
  dst <- unlist(rook_neighbors_unique, use.names = FALSE)
  data.table(src = src, dst = id_order[dst])
}

adj_dt <- build_adjacency(id_order, rook_neighbors_unique)

# 2. Convert cell_data to data.table and set keys
setDT(cell_data)
setkey(cell_data, id, year)

# 3. Expand adjacency across years (cartesian join)
years <- unique(cell_data$year)
adj_year <- adj_dt[, .(id = src, neighbor_id = dst)][, year := rep(years, each = .N)]

# 4. Join neighbor values
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_neighbor_stats_dt <- function(dt, adj_year, var_name) {
  # Join neighbor values
  tmp <- adj_year[dt, on = .(neighbor_id = id, year), nomatch = 0]
  setnames(tmp, var_name, "neighbor_val")
  
  # Aggregate max, min, mean by (id, year)
  tmp[, .(
    max_val = if (.N > 0) max(neighbor_val, na.rm = TRUE) else NA_real_,
    min_val = if (.N > 0) min(neighbor_val, na.rm = TRUE) else NA_real_,
    mean_val = if (.N > 0) mean(neighbor_val, na.rm = TRUE) else NA_real_
  ), by = .(id, year)]
}

# 5. Compute and merge all neighbor features
for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_dt(cell_data, adj_year, var_name)
  setnames(stats_dt, c("max_val", "min_val", "mean_val"),
           paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
  cell_data <- stats_dt[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added
```

**Why this is efficient:**  
- Adjacency is built once and reused.  
- Uses `data.table` joins and grouped aggregation (highly optimized in C).  
- Avoids per-row loops and repeated NA filtering inside loops.  
- Scales well for millions of rows and millions of edges on 16 GB RAM.  

**Expected performance:**  
- Orders of magnitude faster than 86 hours (likely a few hours or less depending on disk I/O).  
- Preserves numerical equivalence with original pipeline.  
- Random Forest model remains unchanged; predictions can be applied immediately after feature augmentation.
END_RESPONSE>>>
