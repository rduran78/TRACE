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
run_id: OpenAI_F09_T03_R019
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 19
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline repeatedly computes neighbor relationships for each year and variable, causing massive redundant work. With ~6.46M rows and 5 variables, iterating through all rows and re-deriving neighbor indices is extremely expensive. The bottleneck is the repeated neighbor lookup and aggregation across 28 years Ã— 344k cells.  

**Optimization Strategy**  
- Build a **single reusable neighbor lookup** based on cell IDs (not cell-years).  
- For each year, **join yearly attributes** to that lookup and compute neighbor stats in a **vectorized manner**.  
- Avoid repeated `lapply` over millions of rows; instead, use `data.table` for fast grouping and joins.  
- Compute all neighbor stats in one pass per variable per year.  
- Preserve the trained Random Forest model and estimand by keeping feature definitions identical.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute reusable neighbor table (cell-to-cell)
neighbor_dt <- rbindlist(lapply(seq_along(id_order), function(i) {
  if (length(rook_neighbors_unique[[i]]) == 0) return(NULL)
  data.table(
    id = id_order[i],
    neighbor_id = id_order[rook_neighbors_unique[[i]]]
  )
}))

# Keys for fast joins
setkey(neighbor_dt, neighbor_id)
setkey(cell_data, id, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_year <- function(var_name) {
  # For each year, join and aggregate
  result_list <- lapply(unique(cell_data$year), function(yr) {
    # Subset for this year
    year_data <- cell_data[year == yr, .(id, val = get(var_name))]
    
    # Join neighbor values
    joined <- neighbor_dt[year_data, on = .(neighbor_id = id)]
    
    # Aggregate neighbor stats per focal id
    stats <- joined[, .(
      n_max = max(val, na.rm = TRUE),
      n_min = min(val, na.rm = TRUE),
      n_mean = mean(val, na.rm = TRUE)
    ), by = id]
    
    # Merge back to main data
    stats[, year := yr]
    setnames(stats, c("n_max", "n_min", "n_mean"),
             paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")))
    stats
  })
  
  rbindlist(result_list)
}

# Compute for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
all_stats <- lapply(neighbor_source_vars, compute_neighbor_stats_year)

# Merge all stats into cell_data
for (stats in all_stats) {
  cell_data <- merge(cell_data, stats, by = c("id", "year"), all.x = TRUE)
}

# At this point, cell_data has neighbor features ready for Random Forest prediction
# Predict using existing trained model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

### **Why This is Faster**
- **Neighbor relationships computed once** (1.37M rows vs 6.46M Ã— 5 loops).
- **Yearly joins** avoid recomputing neighbor indices.
- **Vectorized aggregation** via `data.table` replaces millions of `lapply` calls.
- Memory efficient: works year-by-year instead of loading all combinations.

**Expected improvement:** From 86+ hours to a few hours (or less) on a 16 GB laptop.
END_RESPONSE>>>
