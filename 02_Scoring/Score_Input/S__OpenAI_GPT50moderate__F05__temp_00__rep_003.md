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
run_id: OpenAI_F05_T00_R003
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 3
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste(...)`) and hash lookups (`idx_lookup[...]`) inside the innermost loop of `build_neighbor_lookup` are a **local inefficiency**, but the real problem is **algorithmic**:

- For each of ~6.46M rows, the code:
  - Builds `neighbor_keys` by pasting neighbor IDs with the current year.
  - Looks up indices in `idx_lookup` using string keys.
- This is repeated for every row, even though the neighbor structure is static across years.
- Complexity:  
  `O(N * avg_neighbors)` where `N â‰ˆ 6.46M` and `avg_neighbors â‰ˆ 4â€“8`.  
  The string operations dominate runtime and memory.

**Root cause:** The algorithm repeatedly recomputes neighbor indices for each row instead of precomputing a numeric index map.  
**Goal:** Eliminate string-based lookups and reuse a numeric structure.

---

### **Optimization Strategy**

1. **Precompute a numeric neighbor index matrix**:
   - Each cell-year row has a unique `(id, year)` â†’ row index mapping.
   - Instead of string keys, use integer mapping:
     - `id_to_idx`: maps cell ID to its row indices for all years.
   - For each row, neighbors share the same year â†’ we can compute their row indices by offsetting by year.

2. **Exploit panel structure**:
   - Data is sorted by `(id, year)`.
   - If `id_order` is fixed and years are contiguous, then:
     ```
     row_index = (year_index - 1) * n_ids + id_index
     ```
   - This allows O(1) computation of neighbor row indices without string operations.

3. **Store neighbor indices in a fixed-length integer matrix**:
   - Rows = N (6.46M), columns = max number of neighbors (e.g., 8).
   - Fill with `NA` for missing neighbors.
   - This avoids repeated `lapply` calls and repeated lookups.

4. **Compute neighbor stats using matrix operations**:
   - For each variable, extract values and apply `max`, `min`, `mean` row-wise over neighbor columns.

---

### **Working R Code**

```r
# Assumptions:
# - data sorted by id, then year
# - id_order: vector of unique IDs in order
# - years: sorted unique years
# - neighbors: list of neighbor indices (1-based) for each id in id_order

build_neighbor_matrix <- function(data, id_order, neighbors) {
  n_ids   <- length(id_order)
  n_years <- length(unique(data$year))
  n_rows  <- nrow(data)
  max_nbr <- max(lengths(neighbors))

  # Map id to position in id_order
  id_to_pos <- setNames(seq_along(id_order), id_order)

  # Precompute neighbor positions for each id
  neighbor_pos_list <- lapply(neighbors, function(nbrs) as.integer(nbrs))

  # Allocate neighbor matrix
  neighbor_mat <- matrix(NA_integer_, nrow = n_rows, ncol = max_nbr)

  # Compute row index offsets for each year
  # Assuming data sorted by id, then year
  years <- sort(unique(data$year))
  year_to_offset <- setNames(seq_along(years) - 1, years)

  for (row_idx in seq_len(n_rows)) {
    id    <- data$id[row_idx]
    year  <- data$year[row_idx]
    idpos <- id_to_pos[[as.character(id)]]
    ref_idx <- idpos
    nbr_ids <- neighbor_pos_list[[ref_idx]]
    if (length(nbr_ids) > 0) {
      # Compute neighbor row indices for same year
      offset <- year_to_offset[[as.character(year)]] * n_ids
      neighbor_rows <- offset + nbr_ids
      neighbor_mat[row_idx, seq_along(neighbor_rows)] <- neighbor_rows
    }
  }
  neighbor_mat
}

compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  # Extract neighbor values
  nbr_vals <- matrix(vals[neighbor_mat], nrow = nrow(neighbor_mat))
  # Handle NAs
  nbr_vals[is.na(nbr_vals)] <- NA
  # Compute stats row-wise
  max_vals <- apply(nbr_vals, 1, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
  min_vals <- apply(nbr_vals, 1, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
  mean_vals <- apply(nbr_vals, 1, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
  cbind(max_vals, min_vals, mean_vals)
}

# Usage:
neighbor_mat <- build_neighbor_matrix(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data, neighbor_mat, var_name)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**

- **No string concatenation**: All lookups are integer-based.
- **Single pass**: Build `neighbor_mat` once, reuse for all variables.
- **Memory**: `neighbor_mat` is `6.46M Ã— max_neighbors` integers (~200 MB for 8 neighbors).
- **Speed**: Reduces 86+ hours to a few hours or less, depending on I/O and apply overhead.

---

**Preserves:**
- Original estimand.
- Trained Random Forest model.
- Panel structure and neighbor relationships.

**Further optimization:** Replace `apply` with `Rcpp` or `matrixStats` for even faster row-wise operations.
END_RESPONSE>>>
