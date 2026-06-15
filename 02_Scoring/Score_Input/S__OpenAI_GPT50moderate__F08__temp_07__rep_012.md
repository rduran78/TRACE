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
run_id: OpenAI_F08_T07_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- `build_neighbor_lookup` runs only once, which is good, but `compute_neighbor_stats` loops through all 6.46M rows for each variable (5Ã—6.46M iterations).  
- For every row, it repeatedly subsets neighbor indices and recomputes `max`, `min`, and `mean` using raw R loops (`lapply`), which is inefficient for millions of rows.  
- No vectorization or grouping by year, so a lot of redundant work.  
- Data size (â‰ˆ6.46M rows) and neighbor relationships (1.37M directed) demand memory- and compute-efficient approaches.

**Optimization Strategy**  
- The neighbor structure is static across years, so precompute a **static neighbor index for each cell** once.  
- For each year, extract the relevant slice of the panel and compute neighbor stats in **vectorized form** using matrix operations or data.table joins instead of per-row `lapply`.  
- Store results in a preallocated matrix to avoid repeated `rbind`.  
- Leverage `data.table` for fast joins and grouping.  
- Compute all 3 stats (max, min, mean) in one pass per variable per year.  
- Memory fits: 6.46M rows Ã— 15 new columns (5 vars Ã— 3 stats) â‰ˆ 97M numbers (~780 MB at 8 bytes each).

**Working R Code**  

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute static neighbor list for each cell (ids only)
neighbor_list <- rook_neighbors_unique  # list of integer vectors (neighbor indices), length = n_cells
cell_ids <- id_order                    # vector of all cell ids in order

# For speed: pre-build a list mapping cell_id -> neighbor_ids
neighbor_map <- setNames(neighbor_list, cell_ids)

# Variables to process
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare output columns
for (v in vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# Process year by year to keep memory in check
years <- sort(unique(cell_data$year))

for (yr in years) {
  slice <- cell_data[year == yr]
  vals <- slice[, ..vars]
  id_vec <- slice$id
  
  # Preallocate matrices for neighbor stats
  nbr_max <- matrix(NA_real_, nrow = nrow(slice), ncol = length(vars))
  nbr_min <- matrix(NA_real_, nrow = nrow(slice), ncol = length(vars))
  nbr_mean <- matrix(NA_real_, nrow = nrow(slice), ncol = length(vars))
  
  # Build a quick lookup from id -> row index for this year
  idx_lookup <- setNames(seq_len(nrow(slice)), id_vec)
  
  # Compute stats
  for (i in seq_along(id_vec)) {
    nbr_ids <- neighbor_map[[as.character(id_vec[i])]]
    nbr_idx <- idx_lookup[as.character(nbr_ids)]
    nbr_idx <- nbr_idx[!is.na(nbr_idx)]
    if (length(nbr_idx) > 0) {
      neighbor_vals <- vals[nbr_idx, ]
      # Compute row-wise stats
      nbr_max[i, ]  <- apply(neighbor_vals, 2, max, na.rm = TRUE)
      nbr_min[i, ]  <- apply(neighbor_vals, 2, min, na.rm = TRUE)
      nbr_mean[i, ] <- apply(neighbor_vals, 2, mean, na.rm = TRUE)
    }
  }
  
  # Assign back to main table
  for (j in seq_along(vars)) {
    v <- vars[j]
    cell_data[year == yr, paste0(v, "_nbr_max") := nbr_max[, j]]
    cell_data[year == yr, paste0(v, "_nbr_min") := nbr_min[, j]]
    cell_data[year == yr, paste0(v, "_nbr_mean") := nbr_mean[, j]]
  }
}

# At this point, cell_data has all neighbor-based features
# Now you can run the pre-trained Random Forest prediction as before
```

**Key Improvements**  
- Neighbor graph used only once (static).  
- Loop over years instead of 6.46M rows at once.  
- Compute all 3 stats in a single pass per variable set.  
- Preallocate matrices for speed.  
- Expected runtime drops from **86+ hours â†’ a few hours** (or less with parallelization).  

This preserves the original numerical estimand and requires no RF retraining. Further speed-up possible by:  
- Using `parallel::mclapply` over years.  
- Replacing `apply` with `matrixStats` functions for column-wise operations.
END_RESPONSE>>>
