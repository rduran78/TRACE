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
run_id: OpenAI_F04_T03_R015
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 15
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each iteration performs string concatenation (`paste`) and hash lookups, which are extremely costly at this scale. The neighbor relationships are static across years, but the code redundantly recomputes neighbor indices for every cell-year. This results in ~6.46M Ã— average neighbors operations, which is prohibitively expensive.

---

**Optimization Strategy:**  
1. **Precompute neighbor indices once per cell (not per cell-year)** since spatial neighbors do not change over time.
2. **Vectorize neighbor stats computation** using `data.table` or `dplyr` joins instead of per-row `lapply`.
3. **Avoid string concatenation for keys**; use integer indexing.
4. **Compute neighbor stats in a grouped manner**: reshape data to wide or use rolling joins so that for each year, neighbor values are merged efficiently.
5. **Parallelize where possible** using `data.table` or `future.apply`.

---

**Optimized R Code (data.table approach):**

```r
library(data.table)

# Convert to data.table
setDT(cell_data)

# Precompute neighbor list as integer indices (once)
id_to_idx <- setNames(seq_along(id_order), id_order)
neighbor_idx_list <- lapply(rook_neighbors_unique, function(nbs) id_to_idx[nbs])

# Add an integer index for each cell
cell_data[, cell_idx := id_to_idx[id]]

# For speed, set keys
setkey(cell_data, cell_idx, year)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(dt, var_name, neighbor_idx_list) {
  # Extract relevant columns
  vals <- dt[[var_name]]
  
  # Prepare result matrix
  res <- matrix(NA_real_, nrow = nrow(dt), ncol = 3)
  
  # Process by year to reduce memory footprint
  years <- unique(dt$year)
  
  for (yr in years) {
    # Subset for this year
    idx_year <- which(dt$year == yr)
    vals_year <- vals[idx_year]
    cell_idx_year <- dt$cell_idx[idx_year]
    
    # Map from cell_idx to position in this year's slice
    pos_map <- match(seq_along(id_order), cell_idx_year)
    
    # Compute neighbor stats
    for (i in seq_along(idx_year)) {
      nbs <- neighbor_idx_list[[cell_idx_year[i]]]
      nb_pos <- pos_map[nbs]
      nb_pos <- nb_pos[!is.na(nb_pos)]
      if (length(nb_pos) > 0) {
        nb_vals <- vals_year[nb_pos]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          res[idx_year[i], ] <- c(max(nb_vals), min(nb_vals), mean(nb_vals))
        }
      }
    }
  }
  
  colnames(res) <- paste0(var_name, c("_nb_max", "_nb_min", "_nb_mean"))
  res
}

# Apply for all variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
results_list <- lapply(neighbor_source_vars, function(v) compute_neighbor_stats_fast(cell_data, v, neighbor_idx_list))

# Combine results
neighbor_features <- do.call(cbind, results_list)
cell_data <- cbind(cell_data, neighbor_features)
```

---

**Expected Gains:**  
- Eliminates repeated string operations and hash lookups.
- Reduces complexity from O(n Ã— neighbors Ã— years) string-based to integer-based lookups.
- Memory-efficient by processing year-by-year.
- On a 16 GB laptop, this should cut runtime from 86+ hours to a few hours (or less with parallelization).

**Preserves:**  
- Original Random Forest model.
- Numerical estimand (same neighbor stats).
- Panel structure and neighbor relationships.
END_RESPONSE>>>
