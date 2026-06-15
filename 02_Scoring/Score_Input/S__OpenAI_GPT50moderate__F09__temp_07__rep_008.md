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
run_id: OpenAI_F09_T07_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation rebuilds neighbor relationships for every row in the dataset, repeatedly creating and processing large intermediate structures across ~6.46M rows. This leads to severe redundancy and memory overhead. The adjacency between cells does not change across years, so computing neighbor lookups per cell-year is inefficient. The bottleneck is the nested loop logic inside `compute_neighbor_stats` and repeated string concatenations, which scale poorly when applied millions of times.

---

**Optimization Strategy**  
1. **Precompute and cache a reusable neighbor index table** at the cell level (not per cell-year).  
2. **Join yearly attributes onto this static neighbor table**:  
   - For each year, extract the relevant columns and compute neighbor stats by joining to the static adjacency.  
3. Use **vectorized operations and data.table** to avoid millions of small list operations.  
4. Keep everything in memory-efficient structures, process year-by-year to avoid blowing up RAM.  
5. Preserve the trained Random Forest model and numeric estimands by keeping variable names and scales unchanged.

---

**Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in the same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object (list of integer neighbor indices)

# 1. Build reusable neighbor table (static across years)
build_neighbor_table <- function(id_order, neighbors) {
  from <- rep(id_order, lengths(neighbors))
  to   <- id_order[unlist(neighbors)]
  data.table(id = from, neighbor_id = to)
}

neighbor_dt <- build_neighbor_table(id_order, rook_neighbors_unique)

# 2. Convert main data to data.table and set keys
setDT(cell_data)
setkey(cell_data, id, year)

# 3. Function to compute neighbor stats for one variable, one year
compute_neighbor_stats_year <- function(dt_year, neighbor_dt, var_name) {
  # Join neighbors with current year's data
  joined <- neighbor_dt[dt_year, on = .(neighbor_id = id)]
  # joined now has: id (original cell), neighbor_id, year, var_name
  stats <- joined[, .(
    max = max(get(var_name), na.rm = TRUE),
    min = min(get(var_name), na.rm = TRUE),
    mean = mean(get(var_name), na.rm = TRUE)
  ), by = id]
  # Handle case where all are NA
  stats[is.infinite(max), c("max","min","mean") := NA]
  setnames(stats, c("id", paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
  stats
}

# 4. Process all years and variables in a loop
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))
names(result_list) <- years

for (yr in years) {
  dt_year <- cell_data[year == yr]
  stats_list <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_year(dt_year, neighbor_dt, v))
  year_stats <- Reduce(function(x,y) merge(x,y, by = "id"), stats_list)
  year_stats[, year := yr]
  result_list[[as.character(yr)]] <- year_stats
}

neighbor_features <- rbindlist(result_list)
setkey(neighbor_features, id, year)

# 5. Merge back into main data
cell_data <- merge(cell_data, neighbor_features, by = c("id","year"), all.x = TRUE)

# At this point, cell_data has the original variables plus computed neighbor stats
# Use with the already trained Random Forest model as before
```

---

**Why This Is Faster**  
- The neighbor table (â‰ˆ1.37M rows) is built once instead of reconstructing millions of lookups.  
- Year-by-year processing reduces memory footprint and leverages efficient `data.table` grouping instead of millions of small `lapply` calls.  
- Eliminates repeated string concatenation and large list indexing operations.  

**Expected Speed-Up**: From 86+ hours to a few hours (or less) on a 16 GB laptop, depending on I/O and CPU speed.
END_RESPONSE>>>
