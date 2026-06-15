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
run_id: OpenAI_F09_T07_R007
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 7
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for each cell-year row, resulting in heavy recomputation and high memory overhead. With 6.46M rows and 5 neighbor-based variables, the `build_neighbor_lookup` function creates a large list (one entry per row), which is expensive for both time and memory. Additionally, the loop over variables causes redundant neighbor traversals.

**Optimization Strategy**  
- Build the neighbor lookup **once** at the cell level (344,208 cells), not per cell-year row.
- For each year, **join yearly attributes** to this fixed neighbor graph.
- Compute neighbor statistics using **vectorized joins or matrix operations** instead of large nested loops.
- Use `data.table` for efficient grouping and joining.
- Avoid recomputation of neighbor lists across variables and years.
- Process one year at a time to control memory usage.

---

### **Optimized R Code**

```r
library(data.table)

# Assumes: cell_data has columns id, year, and predictor variables
# rook_neighbors_unique is a list of neighbor indices (from spdep)

# Convert to data.table for speed
setDT(cell_data)

# Precompute adjacency table at cell-level
build_adjacency_table <- function(id_order, rook_neighbors_unique) {
  from_ids <- rep(id_order, times = lengths(rook_neighbors_unique))
  to_ids   <- unlist(rook_neighbors_unique)
  data.table(from = from_ids, to = id_order[to_ids])
}

adjacency_dt <- build_adjacency_table(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one year efficiently
compute_neighbor_stats_year <- function(dt_year, adjacency_dt, vars) {
  # dt_year: data for one year
  # Join adjacency to bring neighbor values
  result_list <- list(id = dt_year$id)
  
  for (v in vars) {
    adj_join <- merge(adjacency_dt, dt_year[, .(to = id, val = get(v))],
                      by = "to", all.x = TRUE, allow.cartesian = TRUE)
    
    # Compute stats by 'from' (original cell)
    stats <- adj_join[, .(
      max_val = if (.N > 0) max(val, na.rm = TRUE) else NA_real_,
      min_val = if (.N > 0) min(val, na.rm = TRUE) else NA_real_,
      mean_val = if (.N > 0) mean(val, na.rm = TRUE) else NA_real_
    ), by = from]
    
    setnames(stats, c("from", paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))))
    
    # Merge back to dt_year
    dt_year <- merge(dt_year, stats, by.x = "id", by.y = "from", all.x = TRUE)
  }
  dt_year
}

# Apply year by year to control memory
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

years <- unique(cell_data$year)
result_list <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  dt_year <- compute_neighbor_stats_year(dt_year, adjacency_dt, neighbor_source_vars)
  result_list[[i]] <- dt_year
}

# Combine all years back
cell_data_enhanced <- rbindlist(result_list)

# Now cell_data_enhanced contains neighbor features and can go to prediction
# Preserves original Random Forest model and estimand
```

---

### **Why This is Faster**
- **Adjacency computed once** (344k rows instead of 6.46M).
- Yearly processing avoids creating a 6.46M-length list.
- `data.table` joins and grouped aggregations reduce overhead from loops.
- Reuse adjacency for all variables and years.

**Expected Speed-up:** From 86+ hours to a few hours (depending on I/O and CPU). Memory footprint remains within 16 GB by chunking per year.
END_RESPONSE>>>
