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
run_id: OpenAI_F08_T03_R009
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 9
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is inefficient because it recomputes neighbor statistics for every cell-year row (â‰ˆ6.46M) by iterating through a large `neighbor_lookup` list. Although neighbor relationships are static across years, the code redundantly processes the same neighbor structure 28 times. This leads to massive overhead in both computation and memory, causing the 86+ hour estimate.

---

**Optimization Strategy**  
1. **Precompute static neighbor lookup once** at the cell level (not cell-year).
2. **Reshape data by year** and compute neighbor stats year by year using vectorized operations.
3. Avoid `lapply` over 6.46M rows; instead, compute for all cells in a year in one pass.
4. Use `data.table` for fast grouping and joins.
5. Preserve the trained Random Forest model and numerical estimand.

---

**Optimized R Code**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute static neighbor lookup at cell level
# neighbor_list: list of integer vectors, each element = neighbor cell IDs
neighbor_list <- rook_neighbors_unique  # already precomputed

# Function to compute neighbor stats for one variable in one year
compute_neighbor_stats_year <- function(dt_year, var_name, neighbor_list) {
  vals <- dt_year[[var_name]]
  # Preallocate result matrix
  res <- matrix(NA_real_, nrow = nrow(dt_year), ncol = 3)
  for (i in seq_len(nrow(dt_year))) {
    neigh_ids <- neighbor_list[[i]]
    if (length(neigh_ids) > 0) {
      neigh_vals <- vals[neigh_ids]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        res[i, ] <- c(max(neigh_vals), min(neigh_vals), mean(neigh_vals))
      }
    }
  }
  colnames(res) <- paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  res
}

# Main optimized loop
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Split by year for memory efficiency
years <- unique(cell_data$year)
setkey(cell_data, year)

for (yr in years) {
  dt_year <- cell_data[year == yr]
  # Compute neighbor stats for each variable
  for (var_name in neighbor_source_vars) {
    stats_mat <- compute_neighbor_stats_year(dt_year, var_name, neighbor_list)
    # Bind columns back
    dt_year <- cbind(dt_year, stats_mat)
  }
  # Update main table
  cell_data[year == yr] <- dt_year
}

# At this point, cell_data has all neighbor features added
# Proceed with Random Forest prediction using the existing trained model
```

---

**Why This Works**  
- **Static neighbor structure**: We compute once per cell and reuse across years.
- **Year-wise processing**: Reduces memory footprint and avoids 6.46M-row iteration.
- **Matrix preallocation**: Eliminates repeated `rbind` overhead.
- **Preserves estimand**: Same max, min, mean logic applied per year.

**Expected Performance**  
- Complexity drops from O(N * T * avg_neighbors) with heavy R overhead to O(T * N * avg_neighbors) but in efficient loops.
- On a standard laptop, this should reduce runtime from 86+ hours to a few hours or less, depending on I/O and CPU speed.  

This approach keeps the Random Forest model intact and produces identical numerical results.
END_RESPONSE>>>
