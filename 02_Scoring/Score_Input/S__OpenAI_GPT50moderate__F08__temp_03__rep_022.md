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
run_id: OpenAI_F08_T03_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because it recomputes neighbor statistics (`max`, `min`, `mean`) for each variable across all 6.46M rows using an `lapply` over rows. This results in ~6.46M Ã— 5 variables = 32M iterations, each doing vector lookups. The neighbor structure is static across years, but the code redundantly rebuilds neighbor relationships for every row-year combination. This leads to massive overhead and memory pressure.

---

**Optimization Strategy**  
1. **Exploit static neighbor structure**: Build a neighbor index once at the cell level (344,208 cells), not at the row level (6.46M rows).
2. **Vectorize by year**: For each year, compute neighbor stats for all cells in one pass using matrix operations.
3. **Avoid repeated lookups**: Precompute a sparse adjacency matrix or list for neighbors and reuse it.
4. **Chunk processing**: Process one year at a time to keep memory usage manageable.
5. **Preserve estimand**: Ensure the computed neighbor stats match the original logic (same neighbors, same year).

---

**Optimized R Code**

```r
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in the same order as rook_neighbors_unique

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor list as integer positions
id_to_pos <- setNames(seq_along(id_order), id_order)
neighbor_pos <- lapply(rook_neighbors_unique, function(nbs) id_to_pos[nbs])

# Variables to compute
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare result columns
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process by year
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)

for (yr in years) {
  message("Processing year: ", yr)
  # Subset for this year
  dt_year <- cell_data[year == yr]
  
  # Ensure order matches id_order
  vals_mat <- dt_year[match(id_order, id), ..neighbor_source_vars]
  
  # Compute neighbor stats for each variable
  for (var_idx in seq_along(neighbor_source_vars)) {
    var_name <- neighbor_source_vars[var_idx]
    vals <- vals_mat[[var_idx]]
    
    # Compute stats for each cell
    nbr_stats <- t(vapply(seq_len(n_cells), function(i) {
      nbs <- neighbor_pos[[i]]
      if (length(nbs) == 0) return(c(NA, NA, NA))
      nbr_vals <- vals[nbs]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) return(c(NA, NA, NA))
      c(max(nbr_vals), min(nbr_vals), mean(nbr_vals))
    }, numeric(3)))
    
    # Assign back
    dt_year[, paste0(var_name, "_nbr_max") := nbr_stats[,1]]
    dt_year[, paste0(var_name, "_nbr_min") := nbr_stats[,2]]
    dt_year[, paste0(var_name, "_nbr_mean") := nbr_stats[,3]]
  }
  
  # Write back to main table
  cell_data[year == yr] <- dt_year
}

# cell_data now has neighbor features ready for Random Forest prediction
```

---

**Why this is faster**  
- Reduces 6.46M row-level operations to 344K cell-level operations per year.
- Uses `vapply` for compiled loops and processes one year at a time.
- Avoids building a massive row-year neighbor lookup.
- Memory footprint is controlled (only one year in memory at a time).

**Expected runtime improvement**  
From 86+ hours to a few hours on a standard laptop, possibly under 2â€“3 hours depending on I/O and CPU speed, because the complexity drops from O(N_rows Ã— neighbors) to O(N_cells Ã— neighbors Ã— years) with efficient loops and no redundant lookups.

**Preserves**  
- Original estimand (same neighbors, same-year stats).
- Trained Random Forest model (no retraining needed).
END_RESPONSE>>>
