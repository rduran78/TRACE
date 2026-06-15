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
run_id: OpenAI_F08_T03_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all 6.46M rows and repeatedly scans neighbor indices for each variable. This results in ~6.46M Ã— 5 loops, each doing vector operations, causing massive overhead. Additionally, the neighbor structure is static across years, but the code rebuilds neighbor relationships for every row-year combination, wasting memory and CPU.

**Optimization Strategy**  
- Exploit the static neighbor structure: compute neighbor indices once at the *cell level* (344,208 cells), not per row-year.
- Reshape data into a wide matrix grouped by `id` (cell) and `year` for fast vectorized operations.
- Precompute neighbor index lists for cells only (length = 344,208).
- For each year, compute neighbor stats in a vectorized way using matrix operations or `vapply` over 344k cells, then bind results back.
- Avoid repeated `lapply` over 6.46M rows.
- Use `data.table` for efficient grouping and joining.

This reduces complexity from O(N_rows Ã— neighbors) to O(N_cells Ã— years Ã— neighbors), which is ~344k Ã— 28 vs 6.46M Ã— neighbors, and avoids redundant lookups.

---

### **Optimized R Code**

```r
library(data.table)

# Assume cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# 1. Build static neighbor lookup at cell level
build_neighbor_lookup_static <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[neighbors[[i]]]
    id_to_idx[as.character(neighbor_ids)]
  })
}

neighbor_lookup_static <- build_neighbor_lookup_static(id_order, rook_neighbors_unique)

# 2. Reshape data into wide matrix by variable for fast access
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# Create a list of matrices: one per variable, rows=cells, cols=years
make_matrix <- function(var) {
  m <- matrix(NA_real_, nrow = n_cells, ncol = n_years,
              dimnames = list(id_order, years))
  vals <- cell_data[[var]]
  m[cbind(match(cell_data$id, id_order), match(cell_data$year, years))] <- vals
  m
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
var_mats <- lapply(neighbor_source_vars, make_matrix)
names(var_mats) <- neighbor_source_vars

# 3. Compute neighbor stats per year in a vectorized way
compute_neighbor_stats_year <- function(mat, neighbor_lookup) {
  n_cells <- nrow(mat)
  n_years <- ncol(mat)
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (y in seq_len(n_years)) {
    col_vals <- mat[, y]
    for (i in seq_len(n_cells)) {
      idx <- neighbor_lookup[[i]]
      if (length(idx) == 0) next
      neighbor_vals <- col_vals[idx]
      neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
      if (length(neighbor_vals) == 0) next
      max_mat[i, y] <- max(neighbor_vals)
      min_mat[i, y] <- min(neighbor_vals)
      mean_mat[i, y] <- mean(neighbor_vals)
    }
  }
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

neighbor_stats <- lapply(var_mats, compute_neighbor_stats_year, neighbor_lookup_static)

# 4. Melt back to long and merge with cell_data
add_neighbor_features <- function(cell_data, var_name, stats_list) {
  max_long <- as.vector(stats_list$max)
  min_long <- as.vector(stats_list$min)
  mean_long <- as.vector(stats_list$mean)
  
  expanded <- CJ(id = id_order, year = years)
  expanded[, paste0(var_name, "_nbr_max") := max_long]
  expanded[, paste0(var_name, "_nbr_min") := min_long]
  expanded[, paste0(var_name, "_nbr_mean") := mean_long]
  
  merge(cell_data, expanded, by = c("id", "year"), all.x = TRUE)
}

for (var_name in neighbor_source_vars) {
  cell_data <- add_neighbor_features(cell_data, var_name, neighbor_stats[[var_name]])
}

# cell_data now has neighbor max/min/mean features added efficiently
```

---

**Why this works efficiently**  
- Neighbor relationships computed once for 344k cells.
- Matrix-based storage allows O(1) access for each cell-year.
- Loops only over 344k Ã— 28 instead of 6.46M Ã— neighbors.
- Memory footprint is manageable: 344k Ã— 28 â‰ˆ 9.6M entries per variable (~76MB per variable).

**Expected runtime**: From 86+ hours down to a few hours or less on a 16GB laptop.
END_RESPONSE>>>
