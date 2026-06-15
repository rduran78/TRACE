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
run_id: OpenAI_F05_T00_R024
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 24
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F05_OpenAI_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**

The repeated string concatenation (`paste`) and hash lookups inside the `lapply` loop in `build_neighbor_lookup` are a **local inefficiency**, but the real issue is **algorithmic**:  
- For each of ~6.46M rows, the code constructs neighbor keys and performs repeated hash lookups.  
- This is done once per row in `build_neighbor_lookup`, and then the neighbor stats are computed repeatedly for each of 5 variables.  
- The neighbor relationships are static across years, so recomputing them for every row is unnecessary.  
- The current approach scales as `O(N * avg_neighbors)` for building the lookup and then repeats similar work for each variable.

**Optimization Strategy**

1. **Precompute a global neighbor index matrix**:  
   - Each cell has a fixed set of neighbors (rook adjacency).  
   - For each cell-year row, neighbors are the same cell IDs but in the same year.  
   - Instead of string keys, use integer indexing:  
     - Sort `data` by `(id, year)` so that rows for each year are contiguous.  
     - Build a matrix of neighbor row indices for all rows in one pass.  

2. **Vectorize neighbor stats computation**:  
   - Once you have an integer matrix of neighbor indices, you can compute max/min/mean for each variable using `apply` or `matrixStats` without repeated lookups.  

3. **Memory considerations**:  
   - With ~6.46M rows and ~4â€“8 neighbors per cell, the neighbor index matrix will have about 6.46M Ã— 8 integers (~200 MB), which fits in 16 GB RAM.  

---

### **Reformulated Approach**

```r
library(data.table)
library(matrixStats)

# Assume data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute mapping from (id, year) to row index
row_index <- seq_len(nrow(cell_data))
id_year_to_idx <- matrix(row_index, nrow = length(unique(cell_data$id)), byrow = FALSE)

# Build neighbor index matrix
# rook_neighbors_unique: list of neighbor IDs for each cell ID
n_cells <- length(id_order)
n_years <- length(unique(cell_data$year))
max_neighbors <- max(lengths(rook_neighbors_unique))

neighbor_idx_mat <- matrix(NA_integer_, nrow = nrow(cell_data), ncol = max_neighbors)

for (year_idx in seq_len(n_years)) {
  year_rows_start <- (year_idx - 1) * n_cells + 1
  year_rows_end   <- year_idx * n_cells
  year_rows       <- year_rows_start:year_rows_end
  
  for (cell_idx in seq_len(n_cells)) {
    row_pos <- year_rows_start + cell_idx - 1
    neigh_ids <- rook_neighbors_unique[[cell_idx]]
    if (length(neigh_ids) > 0) {
      neigh_rows <- year_rows_start + neigh_ids - 1
      neighbor_idx_mat[row_pos, seq_along(neigh_rows)] <- neigh_rows
    }
  }
}

# Compute neighbor stats for each variable
compute_neighbor_stats_fast <- function(vals, neighbor_idx_mat) {
  # vals: numeric vector of length nrow(cell_data)
  # neighbor_idx_mat: integer matrix (rows = observations, cols = neighbors)
  neighbor_vals <- matrix(vals[neighbor_idx_mat], nrow = nrow(neighbor_idx_mat))
  # Replace NA for missing neighbors
  neighbor_vals[is.na(neighbor_vals)] <- NA_real_
  
  max_vals <- rowMaxs(neighbor_vals, na.rm = TRUE)
  min_vals <- rowMins(neighbor_vals, na.rm = TRUE)
  mean_vals <- rowMeans(neighbor_vals, na.rm = TRUE)
  
  cbind(max_vals, min_vals, mean_vals)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_idx_mat)
  colnames(stats) <- paste(var_name, c("nbr_max", "nbr_min", "nbr_mean"), sep = "_")
  cell_data <- cbind(cell_data, stats)
}
```

---

### **Why This Works**
- **No repeated string concatenation or hash lookups**: All indexing is integer-based.
- **Single neighbor matrix reused for all variables**: Avoids recomputation.
- **Vectorized stats computation**: Uses `matrixStats` for speed.
- **Preserves original estimand**: Same neighbor relationships, same summary stats.

---

**Expected Performance Gain**  
- Original: 86+ hours due to repeated string operations and lookups.  
- New: Precompute neighbor matrix once (~minutes), then compute stats for 5 variables in a few more minutes.  
- Overall: Likely **>100Ã— speedup** on a 16 GB laptop.
END_RESPONSE>>>
