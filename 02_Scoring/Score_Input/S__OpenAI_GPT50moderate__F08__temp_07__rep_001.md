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
run_id: OpenAI_F08_T07_R001
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 1
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor statistics (max, min, mean) for each of the 6.46 million cell-year rows by iterating through all rows and accessing neighbor values repeatedly. This is extremely inefficient because:  
- The neighbor structure is static across years, yet `compute_neighbor_stats` recalculates neighbor indices for every row and every variable.  
- For 28 years and 344k cells, the cost of repeatedly scanning neighbors is huge (O(N * neighbors * years * variables)).  
- Using `lapply` and row-by-row operations on millions of elements adds overhead.  

**Optimization Strategy**  
- Precompute `neighbor_lookup` **once per cell** (not per cell-year).  
- Reshape data into a list or array grouped by year, so we can compute neighbor stats per year in vectorized loops rather than row-by-row.  
- For each year, fetch relevant variable vectors, then compute neighbor summaries using matrix operations.  
- Bind results back efficiently with `rbindlist` or `data.table`.  
- Avoid redundant lookups and expensive string concatenation for keys.  

**Optimized R Code**  

```r
library(data.table)

# Assume: cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of all cell IDs in spatial order
# rook_neighbors_unique: nb object from spdep

# 1. Precompute static neighbor lookup by cell index (no year)
build_neighbor_lookup_static <- function(id_order, neighbors) {
  # Return a list: each element is an integer vector of neighbor indices (by id_order position)
  lapply(seq_along(id_order), function(i) neighbors[[i]])
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)
cell_index_map <- setNames(seq_along(id_order), as.character(id_order))

# 2. Convert to data.table and set keys for fast join
setDT(cell_data)
setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))

# 3. Function to compute neighbor stats for one variable across all years
compute_neighbor_stats_fast <- function(dt, var_name, neighbor_lookup, cell_index_map, years) {
  n_cells <- length(cell_index_map)
  result_list <- vector("list", length(years))
  
  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Extract values for this year in id_order
    vals <- dt[year == yr][match(id_order, id), get(var_name)]
    # Initialize matrix for stats: rows = cells, cols = 3 (max, min, mean)
    stats_mat <- matrix(NA_real_, nrow = n_cells, ncol = 3)
    for (i in seq_len(n_cells)) {
      nbr_idx <- neighbor_lookup[[i]]
      if (length(nbr_idx) > 0) {
        nvals <- vals[nbr_idx]
        nvals <- nvals[!is.na(nvals)]
        if (length(nvals) > 0) {
          stats_mat[i, ] <- c(max(nvals), min(nvals), mean(nvals))
        }
      }
    }
    result_list[[yi]] <- data.table(
      id = id_order,
      year = yr,
      max = stats_mat[, 1],
      min = stats_mat[, 2],
      mean = stats_mat[, 3]
    )
  }
  
  rbindlist(result_list)
}

# 4. Loop over source variables and compute stats, merging back
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_stats_fast(cell_data, var_name, neighbor_lookup_static, cell_index_map, years)
  setnames(stats_dt, c("max", "min", "mean"), paste0(var_name, "_nbr_", c("max", "min", "mean")))
  cell_data <- stats_dt[cell_data, on = .(id, year)]
}

# cell_data now has neighbor features added efficiently
```

**Why this is faster**  
- `neighbor_lookup_static` is computed once (not per row-year).  
- For each year (28 iterations) and for each variable (5 iterations), we iterate over 344k cellsâ€”but without string concat or hash lookups.  
- Memory footprint is lower and vectorization reduces overhead.  
- Expected runtime goes from 86+ hours to a few hours on a 16 GB laptop.  

**Key properties preserved**  
- Random Forest model remains unchanged (we only recompute features).  
- Numerical estimand is identical (same max/min/mean logic).  
- Handles NAs and missing neighbors as before.
END_RESPONSE>>>
