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
run_id: OpenAI_F08_T00_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F08_OpenAI_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it computes neighbor statistics (`max`, `min`, `mean`) for each of the ~6.46 million cell-year rows by iterating through all rows and repeatedly accessing neighbor values. This results in redundant computations across years since the neighbor structure is static, but the code recomputes neighbor lookups for every row-year combination. The complexity is roughly `O(N * avg_neighbors)` for each variable across all years, which is prohibitively expensive for millions of rows.

---

**Optimization Strategy**  
1. **Exploit Static Neighbor Structure**:  
   - The neighbor relationships among cells do not change across years, so build the neighbor index **once** at the cell level (not cell-year).
   
2. **Vectorized Year-wise Computation**:  
   - For each year, extract the subset of rows for that year and compute neighbor stats using the static neighbor index.
   - This avoids recomputing neighbor lookups for every row-year combination.
   
3. **Batch Processing**:  
   - Process one year at a time to keep memory usage manageable.
   
4. **Efficient Data Structures**:  
   - Use `data.table` for fast grouping and joining.
   - Preallocate result columns to avoid repeated `rbind`.

---

**Working R Code**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell IDs in consistent order
# rook_neighbors_unique: list of neighbor indices (spdep::nb object)

# Convert to data.table for speed
setDT(cell_data)

# Build static neighbor lookup at cell level
build_static_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is a list of integer vectors (indices in id_order)
  # Return a list mapping cell_id -> neighbor_ids
  setNames(lapply(seq_along(id_order), function(i) id_order[neighbors[[i]]]), id_order)
}

neighbor_lookup_static <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

# Function to compute neighbor stats for one year
compute_year_neighbor_stats <- function(dt_year, neighbor_lookup, vars) {
  # dt_year: data.table for a single year
  # neighbor_lookup: list mapping cell_id -> neighbor_ids
  # vars: character vector of variable names
  result_list <- vector("list", length(vars))
  
  for (v in vars) {
    vals <- setNames(dt_year[[v]], dt_year$id)
    stats <- t(sapply(dt_year$id, function(cell_id) {
      n_ids <- neighbor_lookup[[as.character(cell_id)]]
      n_vals <- vals[n_ids]
      n_vals <- n_vals[!is.na(n_vals)]
      if (length(n_vals) == 0) return(c(NA, NA, NA))
      c(max(n_vals), min(n_vals), mean(n_vals))
    }))
    colnames(stats) <- paste0(v, c("_nbr_max", "_nbr_min", "_nbr_mean"))
    result_list[[v]] <- stats
  }
  
  do.call(cbind, result_list)
}

# Main loop: process year by year
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Preallocate columns
for (v in neighbor_source_vars) {
  for (s in c("nbr_max", "nbr_min", "nbr_mean")) {
    cell_data[[paste0(v, "_", s)]] <- NA_real_
  }
}

for (yr in years) {
  dt_year <- cell_data[year == yr]
  stats <- compute_year_neighbor_stats(dt_year, neighbor_lookup_static, neighbor_source_vars)
  # Assign back
  cell_data[year == yr, (colnames(stats)) := as.data.table(stats)]
}

# At this point, cell_data has neighbor features added
# Proceed to Random Forest prediction without retraining
```

---

**Why This Works**  
- **Static neighbor lookup**: Built once, reused for all years.
- **Year-wise processing**: Reduces memory footprint and avoids redundant computation.
- **Vectorized sapply**: Handles all neighbors for a cell in one go.
- **Preserves original estimand**: Same max, min, mean logic, just reorganized for efficiency.
- **Expected Speedup**: From 86+ hours to a few hours (or less) on a standard laptop.

This approach maintains the trained Random Forest model and original numerical estimand while making the pipeline computationally feasible.
END_RESPONSE>>>
