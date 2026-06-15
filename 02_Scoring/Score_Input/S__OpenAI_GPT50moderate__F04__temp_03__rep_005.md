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
run_id: OpenAI_F04_T03_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F04_OpenAI_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The bottleneck is the repeated `lapply` over 6.46M rows in `build_neighbor_lookup` and `compute_neighbor_stats`. Each call constructs vectors and performs string concatenation (`paste`) and list indexing, which is extremely inefficient at this scale. The neighbor lookup is recomputed for every row and then repeatedly traversed for each variable. This is an O(N Ã— neighbors) operation done multiple times in pure R lists.

---

**Optimization Strategy**  
1. **Precompute neighbor indices once** as an integer matrix instead of a list of vectors. Avoid string-based lookups (`paste`).
2. **Vectorize neighbor stats computation** using matrix operations or `data.table` joins instead of per-row `lapply`.
3. **Use data.table for fast grouping and joining**.
4. **Avoid repeated NA filtering inside loops** by pre-filtering or using `rowMeans` with `na.rm=TRUE`.
5. **Memory efficiency**: Store neighbor indices in a fixed-size integer matrix (rows = cell-year rows, columns = max neighbors) with `NA` padding.

---

### **Optimized R Implementation**

```r
library(data.table)

# Convert to data.table for speed
setDT(cell_data)

# Precompute mapping: (cell_id -> row indices by year)
cell_data[, row_idx := .I]

# Build neighbor matrix: rows = cell-year rows, cols = max neighbors
build_neighbor_matrix <- function(data, id_order, neighbors) {
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  max_nbrs <- max(lengths(neighbors))
  n <- nrow(data)
  neighbor_mat <- matrix(NA_integer_, nrow = n, ncol = max_nbrs)
  
  # Map cell_id to its reference index
  ref_idx_vec <- id_to_ref[as.character(data$id)]
  
  # Precompute year for each row
  years <- data$year
  
  # Build lookup for (id, year) -> row index
  key_dt <- data[, .(id, year, row_idx)]
  setkey(key_dt, id, year)
  
  for (i in seq_len(n)) {
    nbr_ids <- id_order[neighbors[[ref_idx_vec[i]]]]
    if (length(nbr_ids) > 0) {
      nbr_rows <- key_dt[J(nbr_ids, years[i]), row_idx]
      neighbor_mat[i, seq_along(nbr_rows)] <- nbr_rows
    }
  }
  neighbor_mat
}

neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  n <- nrow(neighbor_mat)
  max_nbrs <- ncol(neighbor_mat)
  
  # Extract neighbor values in a matrix
  nbr_vals <- matrix(NA_real_, nrow = n, ncol = max_nbrs)
  valid_idx <- !is.na(neighbor_mat)
  nbr_vals[valid_idx] <- vals[neighbor_mat[valid_idx]]
  
  # Compute stats row-wise
  max_val <- apply(nbr_vals, 1, max, na.rm = TRUE)
  min_val <- apply(nbr_vals, 1, min, na.rm = TRUE)
  mean_val <- rowMeans(nbr_vals, na.rm = TRUE)
  
  # Replace Inf/-Inf with NA for rows with all NA neighbors
  max_val[!is.finite(max_val)] <- NA
  min_val[!is.finite(min_val)] <- NA
  mean_val[is.nan(mean_val)] <- NA
  
  cbind(max_val, min_val, mean_val)
}

# Apply for all neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  cell_data[, paste0(var_name, "_nbr_max") := stats[, 1]]
  cell_data[, paste0(var_name, "_nbr_min") := stats[, 2]]
  cell_data[, paste0(var_name, "_nbr_mean") := stats[, 3]]
}
```

---

### **Why This Is Faster**
- Eliminates string concatenation and repeated list indexing.
- Uses a fixed-size integer matrix for neighbor indices.
- Vectorized computation of max, min, mean using `apply` and `rowMeans`.
- Single pass for each variable instead of millions of `lapply` calls.
- Expected runtime reduction from **86+ hours â†’ under 2â€“3 hours** on a 16 GB laptop.

**Preserves:**
- Original estimand (same neighbor stats).
- Trained Random Forest model (unchanged).
- Data integrity.

**Optional further speed-up:** Parallelize the `for` loop over variables with `future.apply` or `parallel`.
END_RESPONSE>>>
