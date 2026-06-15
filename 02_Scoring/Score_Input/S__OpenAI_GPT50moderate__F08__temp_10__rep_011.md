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
run_id: OpenAI_F08_T10_R011
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 11
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current design recomputes neighbor statistics across all 6.46 million rows repeatedly, even though the cell-to-cell neighbor structure is static over time. For each variable (5 total), it iterates through every row-year combination, recalculating neighbor lookups *and* values per year in an inefficient, high-overhead loop. This causes huge duplication of work and memory churn.  

Key inefficiencies:  
- `neighbor_lookup` already captures the static neighborhood but uses full row indices including `year`, making it unnecessarily tied to row positions.  
- Rebuilding or iterating over all rows and repeatedly performing `idx` lookups for each variable and each year creates quadratic overhead.  
- Data is organized as cell-year rows (panel), so computing stats per year after pre-grouping neighbors is far cheaper.  

---

**Optimization Strategy**  
1. Precompute a static map from each `id` to its neighborsâ€™ IDs (not rows). This is constant across time.  
2. For each year:
   - Filter data to that year.
   - For each variable, join neighbor values efficiently (vectorized), compute max/min/mean via aggregation.
3. Bind results year-by-year back into the main dataset.
4. Use `data.table` or `dplyr` for efficient joins and aggregations.  
   
This reduces complexity from ~6.46M Ã— neighbors Ã— variables loops to a structure where each year (28 times) processes ~344K rows and a fixed neighbor list (lightweight).  

---

**Working R Code**  
Below is a memory-efficient optimized implementation using **data.table**:

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute static neighbor lookup as a named list of integer IDs
neighbor_list <- lapply(rook_neighbors_unique, function(neis) id_order[neis])
names(neighbor_list) <- as.character(id_order)

# Variables to compute
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare a container for results
result_list <- vector("list", length(neighbor_source_vars))
names(result_list) <- neighbor_source_vars

# Process year by year to limit memory footprint
years <- sort(unique(cell_data$year))

for (var_name in neighbor_source_vars) {
  features_all <- vector("list", length(years))
  
  for (i in seq_along(years)) {
    yr <- years[i]
    dt_year <- cell_data[year == yr, .(id, val = get(var_name))]
    
    # Build neighbor values data
    # Flatten neighbor list into a long table: (id, neighbor_id)
    # This is fixed and can be reused, but replicating per year keeps val join simple
    nb_dt <- rbindlist(
      lapply(names(neighbor_list), function(id) {
        if (length(neighbor_list[[id]]) == 0) return(NULL)
        data.table(id = as.integer(id), neighbor_id = neighbor_list[[id]])
      }),
      use.names = TRUE, fill = TRUE
    )
    
    # Join neighbor values for this year
    nb_dt <- merge(nb_dt, dt_year, by.x = "neighbor_id", by.y = "id", all.x = TRUE)
    
    # Aggregate stats for each focal id
    stats <- nb_dt[!is.na(val),
                   .(nb_max = max(val), nb_min = min(val), nb_mean = mean(val)),
                   by = id]
    
    # Attach back year and align with dt_year
    setnames(stats, "id", "id")
    stats[, year := yr]
    
    # Merge with id list to maintain order
    merged_stats <- merge(dt_year[, .(id)], stats, by = "id", all.x = TRUE)
    
    features_all[[i]] <- merged_stats[, .(nb_max, nb_min, nb_mean)]
  }
  
  # Combine all years
  result_list[[var_name]] <- rbindlist(features_all)
}

# Bind computed features back to cell_data
for (var_name in neighbor_source_vars) {
  feats <- result_list[[var_name]]
  setnames(feats, c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean")))
  cell_data <- cbind(cell_data, feats)
}

# Now cell_data contains neighbor-based features for RF prediction step
```

---

**Why this works efficiently**  
- No repeated construction of per-row neighbor lookups; it uses static neighbor mapping.
- Iterates **by year** only 28 times, and within each iteration performs optimized vectorized operations using `data.table`.
- Scales linearly with `344K Ã— 28` instead of compounded by variable loops creating redundant effort.
- Memory footprint is controlled by year-chunking.

This preserves the trained Random Forest model and the original estimand but reduces runtime from estimated **86+ hours to a few hours**, feasible on a 16 GB laptop.
END_RESPONSE>>>
