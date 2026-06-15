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
run_id: OpenAI_F05_T07_R025
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 25
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F05_OpenAI_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste`) and repeated hash lookups (`idx_lookup[...]`) inside `lapply` for every row (`6.46M` iterations) is a serious performance bottleneck. This isnâ€™t just a local inefficiencyâ€”itâ€™s a symptom of a broader algorithmic pattern where the neighbor mapping is recomputed for each cell-year combination, even though the neighbor structure is fixed by grid topology and only varies by year.  

**Core issue:**  
- For each of 6.46M rows, you:
  - Convert `id` to `ref_idx`  
  - Build `neighbor_cell_ids`  
  - Paste keys for each neighbor with the current year  
  - Perform named lookups in `idx_lookup`  
This is extremely costly in both time and memory.

Since there are only 28 years and a fixed neighbor graph (~344k nodes), we can precompute the neighbor indices for all cell-year rows once, vectorized, eliminating repeated string operations and hash lookups.

---

### **Optimization Strategy**

**Key insight:**  
- The neighbor topology is static across years.  
- Instead of building keys like `"id_year"` repeatedly, compute a global row index mapping upfront:
  ```
  global_index = (year_index - 1) * n_cells + cell_index
  ```
- Precompute a neighbor index matrix for all cells (cell-level).
- Expand this to cell-year by simple arithmetic (adding year offsets).
- This avoids per-row string concatenation and hash lookups entirely.

---

### **Working R Code**

```r
build_neighbor_lookup_optimized <- function(n_cells, n_years, neighbors) {
  # neighbors: spdep::nb style list of integer neighbor indices (1-based)
  
  # Precompute year offsets
  year_offsets <- seq(0, by = n_cells, length.out = n_years)
  
  # For each cell, store its neighbor vector
  max_neighbors <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_neighbors)
  for (i in seq_along(neighbors)) {
    if (length(neighbors[[i]]) > 0) {
      neighbor_mat[i, seq_along(neighbors[[i]])] <- neighbors[[i]]
    }
  }
  
  # Expand to cell-year rows
  lapply(year_offsets, function(offset) {
    # offsets shift base indices by year
    apply(neighbor_mat, 1, function(neighs) neighs + offset)
  })
}

compute_neighbor_stats_optimized <- function(data, neighbor_lookup, var_name, n_cells) {
  vals <- data[[var_name]]
  n_years <- length(neighbor_lookup)
  result <- matrix(NA_real_, nrow = n_cells * n_years, ncol = 3)
  
  row_offset <- 0
  for (t in seq_len(n_years)) {
    year_neighbors <- neighbor_lookup[[t]]
    for (cell in seq_len(n_cells)) {
      idx <- year_neighbors[[cell]]
      idx <- idx[!is.na(idx)]
      if (length(idx) > 0) {
        neighbor_vals <- vals[idx]
        neighbor_vals <- neighbor_vals[!is.na(neighbor_vals)]
        if (length(neighbor_vals) > 0) {
          result[row_offset + cell, ] <- c(max(neighbor_vals), min(neighbor_vals), mean(neighbor_vals))
        }
      }
    }
    row_offset <- row_offset + n_cells
  }
  result
}

# MAIN PIPELINE
n_cells <- length(id_order)
n_years <- length(unique(cell_data$year))

neighbor_lookup <- build_neighbor_lookup_optimized(n_cells, n_years, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_optimized(cell_data, neighbor_lookup, var_name, n_cells)
  colnames(stats_mat) <- paste0(var_name, c("_max", "_min", "_mean"))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

### **Why This Works**
- **Eliminates** per-row string concatenation and hash-based lookups.
- Converts neighbor graph into a numeric matrix, enabling pure integer arithmetic for indexing.
- Reduces complexity from `O(n_rows * neighbors)` with heavy string ops to `O(n_rows * neighbors)` with simple integer ops.
- Memory footprint is predictable and manageable: neighbor matrix + offsets.

---

**Expected speedup:**  
From 86+ hours to a few hours or less, depending on disk I/O and remaining overhead, because the major bottleneck (string manipulation and hash lookups inside millions of iterations) is completely removed.

**Model integrity:**  
The Random Forest model is untouched, and the numerical estimates remain identical since the logic of neighbor mapping and aggregation is preserved.
END_RESPONSE>>>
