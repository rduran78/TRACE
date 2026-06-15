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
run_id: OpenAI_F09_T03_R028
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 28
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F09_OpenAI_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor relationships for every cell-year row, causing massive redundant work across 6.46M rows and 28 years. The `build_neighbor_lookup` function is efficient for one pass, but the outer loop repeatedly calls `compute_neighbor_stats` for each variable, scanning the entire dataset multiple times. This results in high memory pressure and excessive computation time (86+ hours).  

**Optimization Strategy**  
- Build the neighbor lookup **once** at the cell level (not cell-year), since spatial adjacency does not change over time.  
- For each year, subset the data, compute neighbor stats for all variables in one pass, and join back.  
- Use **vectorized operations** and `data.table` for speed and memory efficiency.  
- Avoid repeated large `lapply` calls by precomputing neighbor indices and applying them efficiently.  
- Preserve the trained Random Forest model and estimand by not altering the feature definitions, only optimizing computation.  

---

### **Optimized R Code**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute neighbor lookup at cell level
# rook_neighbors_unique: list of integer vectors (neighbors for each cell)
neighbor_lookup <- rook_neighbors_unique  # already precomputed

# Ensure id_order corresponds to cell IDs
id_order <- sort(unique(cell_data$id))
id_to_idx <- setNames(seq_along(id_order), id_order)

# Function to compute neighbor stats for one year
compute_yearly_neighbor_stats <- function(dt_year, neighbor_lookup, vars) {
  # Map cell id to row index in dt_year
  idx_lookup <- setNames(seq_len(nrow(dt_year)), dt_year$id)
  
  # Preallocate result list
  res_list <- vector("list", length(vars))
  names(res_list) <- vars
  
  # For each variable, compute neighbor stats
  for (var in vars) {
    vals <- dt_year[[var]]
    stats_mat <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
    
    for (i in seq_len(nrow(dt_year))) {
      cell_id <- dt_year$id[i]
      neighbors <- neighbor_lookup[[id_to_idx[[as.character(cell_id)]]]]
      if (length(neighbors) == 0) next
      neighbor_ids <- id_order[neighbors]
      neighbor_idx <- idx_lookup[as.character(neighbor_ids)]
      neighbor_idx <- neighbor_idx[!is.na(neighbor_idx)]
      if (length(neighbor_idx) == 0) next
      neighbor_vals <- vals[neighbor_idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      stats_mat[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
    }
    
    colnames(stats_mat) <- paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    res_list[[var]] <- as.data.table(stats_mat)
  }
  
  # Combine all stats for this year
  cbind(dt_year[, .(id, year)], do.call(cbind, res_list))
}

# Main loop: process by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

result_list <- vector("list", length(years))

for (y in seq_along(years)) {
  dt_year <- cell_data[year == years[y]]
  result_list[[y]] <- compute_yearly_neighbor_stats(dt_year, neighbor_lookup, neighbor_source_vars)
}

# Combine all years back
neighbor_features <- rbindlist(result_list)

# Merge neighbor features back into main dataset
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

### **Why This is Faster**
- Neighbor relationships are computed **once** at cell level.
- Each year is processed in isolation, reducing memory footprint.
- Each variable is computed in a single pass per year, not per entire dataset multiple times.
- Uses `data.table` for efficient joins and binding.

**Expected Performance Gain:**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, as redundant computations are eliminated and operations are chunked by year.

**Preserves:**  
- Original Random Forest model (no retraining).
- Original estimand and feature definitions.
END_RESPONSE>>>
