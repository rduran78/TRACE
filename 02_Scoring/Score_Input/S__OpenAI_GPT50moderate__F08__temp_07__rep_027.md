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
run_id: OpenAI_F08_T07_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor stats (max, min, mean) by iterating over all 6.46 million rows and repeatedly scanning neighbor indices. This results in roughly `O(N * k)` operations per variable (where `N`â‰ˆ6.46M and `k`â‰ˆ4), multiplied by 5 variables, causing **~86+ hours** runtime. The core inefficiency is that neighbor relationships are static by cell, but the pipeline recomputes neighbors for every cell-year row.

**Optimization Strategy**  
- **Exploit static neighbor structure**: Compute neighbor stats year-by-year, not row-by-row.  
- Precompute a mapping from `cell_id â†’ neighbor_ids` once, then for each year slice, apply vectorized operations on matrices.  
- Use `data.table` or matrix operations to avoid repeated list traversals.  
- Memory-conscious: operate by year to avoid loading all 6.46M rows at once.  
- Preserve model and estimand by producing identical aggregated features.  

**Optimized Approach**  
1. Precompute `neighbor_map` as a named list: `cell_id â†’ vector of neighbor_ids`.  
2. For each year:
   - Subset data for that year.
   - For each variable, build a numeric vector aligned to `id_order`.
   - Compute neighbor stats in a **vectorized way** using apply over `neighbor_map`.  
3. Append results back to the main dataset.  

---

### **Working R Code**

```r
library(data.table)

# Assume cell_data: data.table with columns id, year, and variables
setDT(cell_data)

# Precompute static neighbor map keyed by cell id
neighbor_map <- setNames(rook_neighbors_unique, id_order)

# Function to compute stats for one variable in one year
compute_neighbor_stats_year <- function(vals, neighbor_map) {
  # vals is a named vector: names(vals) = cell ids
  sapply(neighbor_map, function(neigh) {
    if (length(neigh) == 0) return(c(NA, NA, NA))
    nv <- vals[neigh]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0) return(c(NA, NA, NA))
    c(max(nv), min(nv), mean(nv))
  })
}

# Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare result containers
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nb_max") := NA_real_]
  cell_data[, paste0(v, "_nb_min") := NA_real_]
  cell_data[, paste0(v, "_nb_mean") := NA_real_]
}

# Process by year
years <- sort(unique(cell_data$year))
for (yr in years) {
  cat("Processing year", yr, "\n")
  subset_idx <- which(cell_data$year == yr)
  year_dt <- cell_data[subset_idx]
  
  # Create lookup by id for this year
  vals_by_id <- setNames(seq_len(nrow(year_dt)), year_dt$id)
  
  for (var_name in neighbor_source_vars) {
    vals <- setNames(year_dt[[var_name]], year_dt$id)
    stats_mat <- compute_neighbor_stats_year(vals, neighbor_map)
    # stats_mat has 3 rows: max, min, mean; columns = cells in id_order
    # Transpose to align
    stats_df <- t(stats_mat)
    colnames(stats_df) <- c("max", "min", "mean")
    
    # Match back to rows
    cell_data[subset_idx, paste0(var_name, "_nb_max") := stats_df[as.character(id), "max"]]
    cell_data[subset_idx, paste0(var_name, "_nb_min") := stats_df[as.character(id), "min"]]
    cell_data[subset_idx, paste0(var_name, "_nb_mean") := stats_df[as.character(id), "mean"]]
  }
}

# At this point cell_data has all neighbor features computed efficiently
```

---

### **Why This is Fast**
- **Neighbor list computed once** (static).
- For each year, process ~344k rows, not 6.46M at once.
- Vectorized `sapply` over 344k cells (not nested loops over 6.46M).
- Memory footprint low: only one year in memory during processing.
- Expected runtime: hours â†’ **minutes** on a 16 GB laptop.

**Preserves:**
- Original Random Forest model (unchanged).
- Numerical estimand (identical neighbor-based aggregations).
- Data integrity and panel structure.
END_RESPONSE>>>
