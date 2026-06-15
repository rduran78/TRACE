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
run_id: OpenAI_F09_T00_R023
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 23
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F09_OpenAI_temp_00_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because for each of the ~6.46M rows, `build_neighbor_lookup` repeatedly constructs neighbor indices by string concatenation and hash lookups. This is done once, but then `compute_neighbor_stats` iterates over all rows and repeatedly subsets vectors, causing heavy R-level overhead. The complexity is roughly O(N Ã— avg_neighbors) with millions of small list operations in R, which is inefficient.  

**Optimization Strategy**  
- Precompute a reusable adjacency table (cell_id â†’ neighbor_cell_ids) once, independent of year.  
- For each year, join the year-specific attributes to this adjacency table and compute neighbor statistics using **vectorized data.table operations** instead of per-row loops.  
- Avoid repeated string concatenation and list indexing.  
- Use `data.table` for fast joins and aggregations.  
- Keep memory usage manageable by processing one year at a time and writing results back.  

**Working R Code**  

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in same order as rook_neighbors_unique

# 1. Build adjacency table once
build_adjacency_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- unlist(lapply(neighbors, function(x) id_order[x]), use.names = FALSE)
  data.table(id = from, neighbor_id = to)
}

adjacency_dt <- build_adjacency_table(id_order, rook_neighbors_unique)
setkey(adjacency_dt, id)

# 2. Compute neighbor stats year by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Ensure cell_data is a data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Prepare output container
result_list <- vector("list", length = length(unique(cell_data$year)))

years <- sort(unique(cell_data$year))

for (yr in years) {
  # Subset data for this year
  year_dt <- cell_data[year == yr, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  setkey(year_dt, id)
  
  # Join adjacency with year data for neighbors
  neighbor_dt <- adjacency_dt[year_dt, on = .(neighbor_id = id)]
  # neighbor_dt now has: id (focal), neighbor_id, ntl, ec, ...
  
  # Compute stats for each var
  stats_list <- lapply(neighbor_source_vars, function(var) {
    neighbor_dt[, .(
      max = max(get(var), na.rm = TRUE),
      min = min(get(var), na.rm = TRUE),
      mean = mean(get(var), na.rm = TRUE)
    ), by = id]
  })
  
  # Merge all stats into one table
  stats_dt <- Reduce(function(x, y) merge(x, y, by = "id", all = TRUE), stats_list)
  
  # Rename columns
  new_names <- c("id")
  for (var in neighbor_source_vars) {
    new_names <- c(new_names, paste0(var, "_nbr_max"), paste0(var, "_nbr_min"), paste0(var, "_nbr_mean"))
  }
  setnames(stats_dt, old = names(stats_dt), new = new_names)
  
  # Merge back to year_dt
  year_dt <- merge(year_dt, stats_dt, by = "id", all.x = TRUE)
  
  result_list[[as.character(yr)]] <- year_dt
}

# Combine all years
cell_data_with_neighbors <- rbindlist(result_list)

# Now you can run predictions with the existing Random Forest model
# predictions <- predict(rf_model, newdata = cell_data_with_neighbors)
```

**Why This Is Faster**  
- Adjacency is computed once (O(#edges) ~ 1.37M).  
- For each year (~28 iterations), we do a single join and grouped aggregation using `data.table` (highly optimized in C).  
- Eliminates millions of R-level loops and string operations.  
- Memory stays within 16 GB because we process one year at a time.  

This approach should reduce runtime from 86+ hours to a few hours or less on a standard laptop.
END_RESPONSE>>>
