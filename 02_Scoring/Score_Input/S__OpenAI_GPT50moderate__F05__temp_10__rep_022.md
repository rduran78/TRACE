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
run_id: OpenAI_F05_T10_R022
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 22
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F05_OpenAI_temp_10_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 The inefficiency is **algorithmic**, not just local:  
`build_neighbor_lookup` constructs neighbor indices by repeatedly creating string keys (`paste(id, year)`) and performing a vectorized name lookup for each of ~6.46M rows. This repeats for every neighbor and every row, creating **O(N Ã— neighbors)** string hashing work. For 6.46M rows and multiple features, this dominates runtime.

### Root Cause
- String concatenation + named vector indexing happens for every observation in `lapply(row_ids, ...)`.
- The same mapping of `(id, year) â†’ row index` is effectively recomputed through string manipulations multiple times.
- Five feature passes reuse the same neighbor structure, but rebuilding lookup each time is wasteful.
- The core issue: current design uses string keys per-row rather than precomputing numeric indices.

### Optimization Strategy
1. Replace string-based keys with **pure integer indexing**.
2. Exploit panel structure:  
   - `id` and `year` ranges are fixed â†’ compute row index from integer math.
3. Precompute a **neighbor index matrix**: rows = observations, columns = neighbor positions, values = indices (or `NA`).
4. Use this matrix for all variable computations without repeated neighbor lookups.

This moves complexity from **O(N Ã— deg Ã— string operations)** â†’ **O(N Ã— deg)** integer operations, computed once.

---

### Efficient Reformulation in R

```r
# Precompute fast mappings
build_neighbor_index_matrix <- function(data, id_order, neighbors) {
  # 1. Map original IDs to contiguous 1..n_id
  id_to_idx <- setNames(seq_along(id_order), id_order)
  n_ids <- length(id_order)
  years <- sort(unique(data$year))
  n_years <- length(years)
  year_to_idx <- setNames(seq_along(years), years)

  # 2. Compute row index as: (year_idx - 1)*n_ids + id_idx
  get_row_idx <- function(id_idx, year_idx) (year_idx - 1L) * n_ids + id_idx
  n_rows <- nrow(data)
  stopifnot(n_rows == n_ids * n_years)

  # 3. Prepare output: matrix [n_rows x max_degree]
  max_deg <- max(sapply(neighbors, length))
  neighbor_mat <- matrix(NA_integer_, nrow = n_rows, ncol = max_deg)

  for (yr_idx in seq_along(years)) {
    yr <- years[yr_idx]
    for (ref_id_idx in seq_along(id_order)) {
      row_id <- get_row_idx(ref_id_idx, yr_idx)
      neigh_ids <- neighbors[[ref_id_idx]]
      if (length(neigh_ids) > 0) {
        # translate neighbor node ids -> their physical row index for this year
        neigh_row_idx <- get_row_idx(neigh_ids, yr_idx)
        neighbor_mat[row_id, seq_along(neigh_row_idx)] <- neigh_row_idx
      }
    }
  }
  neighbor_mat
}

compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  apply(neighbor_mat, 1, function(idx) {
    idx <- idx[!is.na(idx)]
    if (length(idx) == 0) return(c(NA, NA, NA))
    neighbor_vals <- vals[idx]
    neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
    if (length(neighbor_vals) == 0) return(c(NA, NA, NA))
    c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
  }) |> t()
}

# ---- Pipeline ----
neighbor_mat <- build_neighbor_index_matrix(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  colnames(stats) <- paste(var_name, c("max", "min", "mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### Advantages
- **Single precomputation** of `neighbor_mat` replaces costly string operations.
- Memory footprint: `6.46M Ã— max_degree (~8)` â‰ˆ 52M ints (~200 MB) â†’ feasible on 16 GB RAM.
- Runtime drops from **86+ hrs â†’ under a few hours (I/O bound)**.
- Original Random Forest model remains unchanged; feature values identical.

**Summary:** The repeated work was a symptom of larger design inefficiency. Moving to numeric indexing and a precomputed neighbor index matrix eliminates redundant string manipulation and repeated lookups.
END_RESPONSE>>>
