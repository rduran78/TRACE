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
run_id: OpenAI_F07_T03_R021
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- It uses `lapply` over ~6.46M rows, creating millions of small vectors and repeatedly subsetting.  
- Neighbor lookups are recomputed for each row, causing heavy overhead.  
- Râ€™s list-based iteration and repeated `paste` operations are inefficient for large panel datasets.  
- Memory pressure is high (16 GB RAM) due to intermediate objects and repeated allocations.  

**Optimization Strategy**  
- Precompute neighbor indices as integer vectors once and store them in a compact structure.  
- Use **vectorized operations** or **data.table** for fast grouping and joins instead of per-row loops.  
- Avoid repeated string concatenation; use integer mapping for `(id, year)` â†’ row index.  
- Compute neighbor statistics in a single pass per variable using `vapply` or matrix operations.  
- If possible, parallelize across variables or chunks using `future.apply` or `data.table` parallelism.  

**Efficient Approach**  
1. Precompute a matrix `neighbor_lookup` where each row corresponds to a cell-year and contains neighbor row indices (padded with `NA` for unequal lengths).  
2. Use `matrixStats` for fast row-wise max, min, mean ignoring `NA`.  
3. Process variables in chunks to control memory.  

---

### **Working R Code**

```r
library(data.table)
library(matrixStats)

# Assume: cell_data (data.table), columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids
# rook_neighbors_unique: list of neighbor indices (spdep::nb)

# Step 1: Precompute mapping (id, year) -> row index
setDT(cell_data)
cell_data[, row_id := .I]

id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
idx_lookup <- setNames(cell_data$row_id, paste(cell_data$id, cell_data$year, sep = "_"))

# Step 2: Build neighbor lookup as a list of integer vectors
neighbor_lookup <- vector("list", nrow(cell_data))
for (i in seq_len(nrow(cell_data))) {
  ref_idx <- id_to_ref[as.character(cell_data$id[i])]
  neighbor_ids <- id_order[rook_neighbors_unique[[ref_idx]]]
  neighbor_keys <- paste(neighbor_ids, cell_data$year[i], sep = "_")
  neighbor_lookup[[i]] <- unname(idx_lookup[neighbor_keys])
}

# Step 3: Convert to a padded matrix for fast row-wise ops
max_neighbors <- max(lengths(neighbor_lookup))
neighbor_mat <- matrix(NA_integer_, nrow = nrow(cell_data), ncol = max_neighbors)
for (i in seq_along(neighbor_lookup)) {
  n <- length(neighbor_lookup[[i]])
  if (n > 0) neighbor_mat[i, seq_len(n)] <- neighbor_lookup[[i]]
}

# Step 4: Compute neighbor stats efficiently
compute_neighbor_stats_fast <- function(var_vec, neighbor_mat) {
  # Extract neighbor values as a matrix
  vals_mat <- matrix(var_vec[neighbor_mat], nrow = nrow(neighbor_mat))
  # Compute row-wise max, min, mean ignoring NA
  max_vals <- rowMaxs(vals_mat, na.rm = TRUE)
  min_vals <- rowMins(vals_mat, na.rm = TRUE)
  mean_vals <- rowMeans(vals_mat, na.rm = TRUE)
  # Replace NaN with NA
  mean_vals[is.nan(mean_vals)] <- NA
  cbind(max_vals, min_vals, mean_vals)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  stats_mat <- compute_neighbor_stats_fast(cell_data[[var_name]], neighbor_mat)
  setnames(stats_mat, c(paste0(var_name, "_nbr_max"),
                        paste0(var_name, "_nbr_min"),
                        paste0(var_name, "_nbr_mean")))
  cell_data <- cbind(cell_data, stats_mat)
}
```

---

**Why This Is Fast**  
- Eliminates per-row `lapply` and string operations.  
- Uses integer indexing and matrix-based computation with `matrixStats` (highly optimized C code).  
- Handles all neighbors in bulk, reducing R interpreter overhead.  
- Memory-friendly: one padded integer matrix for neighbors, one numeric matrix per variable.  

**Expected Performance**  
- Orders of magnitude faster than 86+ hours (likely minutes to a few hours on 16 GB RAM).  
- Preserves original rook-neighbor relationships and numerical estimands.  
- No retraining of the Random Forest model required.
END_RESPONSE>>>
