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
run_id: OpenAI_F09_T10_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F09_OpenAI_temp_10_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is repetitive neighbor lookups across 6.46M rows (344,208 cells Ã— 28 years) for five variables. Current code repeatedly computes neighbor stats by scanning indices for every cell-year combination, which is highly inefficient (O(n Ã— k Ã— vars)). The neighbor relationships do not change over time, so rebuilding neighbor indices repeatedly is wasted effort. Building a reusable adjacency mapping of cell IDs once, then working with a vectorized join-based approach or matrix aggregation per year will significantly reduce computation time and memory pressure.

---

**Optimization Strategy**  
1. **Precompute adjacency table once**:
   - Expand `rook_neighbors_unique` into a long data frame with columns: `cell_id`, `neighbor_id`.
2. **Use join-based aggregation for each year**:
   - For each year, join cell attributes to the adjacency table (for current year only).
   - Group by `cell_id` to compute `max`, `min`, `mean` for each neighbor variable.
3. **Avoid lapply over 6.46M rows**: Use `data.table` for speed and memory efficiency.
4. **Iterate over years instead of all rows**: Compute yearly neighbor stats and append to data.

This reduces complexity from per-row computation to per-year grouped aggregation over neighbor edges (~1.37M edges), repeated 28 times (â‰ˆ38.4M rows to process) rather than per observation (6.46M Ã— neighbor count).

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list from spdep representing neighbor IDs corresponding to id_order
# id_order: vector of cell_id in same order as rook_neighbors_unique

# 1. Build adjacency table once
adj_list <- rook_neighbors_unique
adj_dt <- data.table(
  cell_id    = rep(id_order, lengths(adj_list)),
  neighbor_id = unlist(lapply(adj_list, function(x) id_order[x]), use.names = FALSE)
)

setkey(adj_dt, neighbor_id)  # for fast joins
setkey(cell_data, id, year)

# 2. Convert cell_data to data.table if not already
cell_data <- as.data.table(cell_data)

# 3. Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 4. Compute yearly neighbor stats efficiently
results_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

for (y in years) {
  message("Processing year: ", y)
  year_dt <- cell_data[year == y, .(id, year, ntl, ec, pop_density, def, usd_est_n2)]
  
  # Join attributes with neighbors: neighbor_id -> neighbor values
  join_dt <- adj_dt[year_dt, on = .(neighbor_id = id)]
  # Now join_dt has: cell_id, neighbor_id, year, ntl, ec, ...

  # Compute stats by cell_id for each var
  agg_list <- list()
  for (var_name in neighbor_source_vars) {
    agg_stats <- join_dt[, .(
      max_val = max(get(var_name), na.rm = TRUE),
      min_val = min(get(var_name), na.rm = TRUE),
      mean_val = mean(get(var_name), na.rm = TRUE)
    ), by = cell_id]
    
    setnames(agg_stats, c("cell_id", 
                          paste0(var_name, "_nbr_max"), 
                          paste0(var_name, "_nbr_min"), 
                          paste0(var_name, "_nbr_mean")))
    if (length(agg_list) == 0) {
      agg_list[[1]] <- agg_stats
    } else {
      agg_list[[length(agg_list)+1]] <- agg_stats
    }
  }
  
  # Combine stats for all vars
  merged_stats <- Reduce(function(x, y) merge(x, y, by = "cell_id"), agg_list)
  merged_stats[, year := y]
  
  results_list[[which(years == y)]] <- merged_stats
}

neighbor_stats_all <- rbindlist(results_list)

# 5. Merge back to full data
setkey(neighbor_stats_all, cell_id, year)
setkey(cell_data, id, year)
cell_data <- merge(cell_data, neighbor_stats_all, by.x = c("id", "year"), by.y = c("cell_id", "year"), all.x = TRUE)

# Random Forest prediction (model is pre-trained)
predictions <- predict(rf_model, newdata = cell_data)
```

---

**Why faster?**
- Eliminates per-row lookup and repeated adjacency computation.
- Joins and aggregations leverage vectorized `data.table` operations.
- Scales with ~38M operations instead of ~6.46M Ã— neighbor_count Ã— vars.
- Fits into standard laptop memory (16 GB) since processing happens per year.

**Expected speedup**: From 86+ hours to a few hours or less, depending on I/O and CPU.  
**Model and estimand preserved**: No retraining; numerical operations unchanged.
END_RESPONSE>>>
