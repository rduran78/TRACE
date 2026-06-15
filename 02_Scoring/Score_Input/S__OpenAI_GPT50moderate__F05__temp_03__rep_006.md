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
run_id: OpenAI_F05_T03_R006
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 6
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F05_OpenAI_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The repeated string-key construction (`paste(id, year, sep="_")`) and repeated hash lookups inside the inner `lapply` loop are only a *symptom* of a larger inefficiency: the algorithm repeatedly performs expensive string concatenation and name-based indexing for every row (â‰ˆ6.46M times). This is not just local overhead; itâ€™s an algorithmic bottleneck because the neighbor relationships are static across years, yet the code rebuilds keys and performs lookups per row and per variable.

**Optimization Strategy**  
- Precompute a numeric matrix of neighbor indices for all rows once, avoiding string concatenation entirely.
- Use integer-based indexing instead of name-based lookups.
- Exploit the fact that neighbors are constant across years: replicate neighbor relationships across time in a vectorized manner.
- Compute neighbor statistics in a fully vectorized or batched way rather than looping through rows repeatedly.

**Reformulated Approach**  
1. Map `(id, year)` to a single integer index once.
2. Build an integer matrix `neighbor_idx` where each row corresponds to a row in `data` and contains neighbor row indices (or `NA` if fewer neighbors).
3. Use `matrixStats` or `apply` on slices to compute max, min, mean efficiently.

---

### **Working R Code**

```r
library(matrixStats)

# Assume: data has columns id, year, and is sorted by (id, year)
# id_order: vector of unique ids in desired order
# neighbors: list of neighbor indices (spdep::nb style)

build_neighbor_matrix <- function(data, id_order, neighbors) {
  n_rows <- nrow(data)
  n_ids  <- length(id_order)
  years  <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map id -> position in id_order
  id_pos <- match(data$id, id_order)
  
  # Precompute neighbor ids for each id
  max_deg <- max(lengths(neighbors))
  neighbor_mat <- matrix(NA_integer_, n_rows, max_deg)
  
  # For each row in data, fill neighbor indices
  # Since data is sorted by (id, year), we can compute row index as:
  # row_index = (id_pos - 1) * n_years + year_pos
  year_pos <- match(data$year, years)
  
  # Precompute a lookup: (id_pos, year_pos) -> row index
  # This is just seq_len(n_rows) because of sorting
  row_index <- seq_len(n_rows)
  
  # For each row, find neighbor ids and map to row indices
  for (i in seq_len(n_rows)) {
    ref_idx <- id_pos[i]
    neigh_ids <- id_order[neighbors[[ref_idx]]]
    neigh_pos <- match(neigh_ids, id_order)
    # Compute neighbor row indices for same year
    neigh_rows <- (neigh_pos - 1) * n_years + year_pos[i]
    neighbor_mat[i, seq_along(neigh_rows)] <- neigh_rows
  }
  
  neighbor_mat
}

compute_neighbor_stats_fast <- function(data, neighbor_mat, var_name) {
  vals <- data[[var_name]]
  # Replace invalid indices with NA
  idx_mat <- neighbor_mat
  idx_mat[idx_mat == 0] <- NA_integer_
  
  # Extract neighbor values
  neighbor_vals <- matrix(vals[idx_mat], nrow(idx_mat), ncol(idx_mat))
  
  # Compute row-wise max, min, mean ignoring NA
  max_vals <- rowMaxs(neighbor_vals, na.rm = TRUE)
  min_vals <- rowMins(neighbor_vals, na.rm = TRUE)
  mean_vals <- rowMeans(neighbor_vals, na.rm = TRUE)
  
  # If a row has all NA neighbors, set to NA
  all_na <- rowAlls(is.na(neighbor_vals))
  max_vals[all_na] <- NA
  min_vals[all_na] <- NA
  mean_vals[all_na] <- NA
  
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

### **Why This Is Faster**
- Eliminates repeated string concatenation and hash lookups (O(n) â†’ O(1) per neighbor).
- Builds neighbor index matrix once (O(n * avg_deg)) instead of per variable.
- Uses vectorized operations (`matrixStats`) for summary stats.
- Memory footprint is manageable: `neighbor_mat` â‰ˆ 6.46M Ã— max_deg (e.g., 4â€“8) integers â†’ fits in 16 GB.

**Expected Speedup:** From 86+ hours to a few hours or less, depending on I/O and CPU.
END_RESPONSE>>>
