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
run_id: OpenAI_F08_T10_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The major bottleneck stems from *recomputing neighbor stats for every row in the 6.46M panel*, using `lapply` calls that scale with row count. This ignores the static nature of neighbor relationships across years. For each of 28 years, the same neighbor topology is redundantly processed, multiplying overhead by ~28. Memory and CPU thrash on your laptop is unavoidable with current design.

---

**Optimization Strategy**  
1. Cache the neighbor indices once per unique cell (344,208 entities).
2. Reshape data so computations occur per-year across cells:
   - Split panel into yearly slices.
   - Apply neighbor max/min/mean using precomputed static neighbor topology.
3. Use *vectorized row/column operations* rather than one-by-one `lapply`.
4. Avoid repeated joins by merging results back by year after chunk computation.
5. Reduce memory footprint via `data.table` for slicing and fast assignment.

This reduces complexity from `O(n_rows)` neighbor traversals to `O(n_cells * years)`, but each yearly slice runs over 344K cells instead of 6.46M individually with redundant lookups.

---

### Working R Code

```r
library(data.table)

# Convert to data.table for efficiency
setDT(cell_data)

# Precompute static neighbors per cell_id (not per row-year)
neighbor_lookup_static <- setNames(
  rook_neighbors_unique,
  as.character(id_order) # ensure cell IDs align
)

# Function: compute neighbor stats per year slice (vectorized)
compute_neighbor_stats_year <- function(year_dt, var_name) {
  vals <- year_dt[[var_name]]
  n_cells <- nrow(year_dt)
  
  # Initialize result matrix
  result <- matrix(NA_real_, nrow = n_cells, ncol = 3)
  
  for (i in seq_len(n_cells)) {
    neighbors <- neighbor_lookup_static[[as.character(year_dt$id[i])]]
    if (length(neighbors) == 0) next
    
    neighbor_vals <- vals[neighbors]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) next
    
    result[i, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }
  # Return as data.table with named columns
  as.data.table(result)[, .(max = V1, min = V2, mean = V3)]
}

# Compute and bind features for all years efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
setkey(cell_data, year)

for (var_name in neighbor_source_vars) {
  res_list <- vector("list", length(unique(cell_data$year)))
  yrs <- sort(unique(cell_data$year))
  
  for (k in seq_along(yrs)) {
    yr <- yrs[k]
    yr_dt <- cell_data[year == yr]
    
    stats_dt <- compute_neighbor_stats_year(yr_dt, var_name)
    # Combine with cell_id and year for merge
    stats_dt[, id := yr_dt$id]
    stats_dt[, year := yr]
    res_list[[k]] <- stats_dt
  }
  
  res_all <- rbindlist(res_list)
  setnames(res_all, c("id","year","max","min","mean"),
           c("id","year",
             paste0(var_name,"_nmax"),
             paste0(var_name,"_nmin"),
             paste0(var_name,"_nmean")))
  
  # Merge back efficiently
  cell_data <- res_all[cell_data, on = .(id, year)]
}

# The Random Forest prediction step remains unchanged
```

---

**Expected Impact**  
- Removes redundant neighbor recomputation for 6.46M rows.
- Uses yearly splits + precomputed static lookup for topology.
- Memory-safe with `data.table` and chunking.
- Estimated runtime reduction: from 86+ hours to a few hours, possibly minutes depending on SSD and CPU.

The pipeline preserves:
- Trained Random Forest model.
- Original target estimand.
- Correct neighbor max/min/mean definitions.
END_RESPONSE>>>
