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
run_id: OpenAI_F08_T10_R029
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 29
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F08_OpenAI_temp_10_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because:  
- `compute_neighbor_stats` iterates over all 6.46M rows and recomputes stats for each variable separately (5Ã—6.46M operations).  
- `neighbor_lookup` is huge and re-used redundantly for each variable.  
- The code performs repeated `lapply`/`do.call` for each feature, producing large intermediate lists and repeated NA filtering.  
- Static neighbor relationships (344k cells, ~1.37M neighbor links) are unnecessarily recalculated for every year and variable.

---

**Optimization Strategy**  
1. **Exploit static structure**: Precompute a sparse neighbor matrix (cells Ã— cells) once.  
2. **Group by year**: For each year, do a single matrix multiplication for all variables using the sparse matrix.  
3. **Vectorize**: Compute max, min, mean neighbor stats in a single pass using efficient operations.  
4. **Memory-efficiency**: Use `Matrix` package for storing neighbor relationships as a sparse structure.  

Complexity drops from `O(rows * vars)` lookups to `O(years * (cells + neighbor_links * vars))`.

---

**Working R Code**

```r
library(Matrix)
library(data.table)

# Assumes cell_data has columns: id (cell_id), year, and source vars.
setDT(cell_data)

# 1. Build sparse neighbor adjacency matrix once
n_cells <- length(id_order)
id_to_idx <- setNames(seq_along(id_order), id_order)

neighbors <- rook_neighbors_unique
row_idx <- rep(seq_along(neighbors), lengths(neighbors))
col_idx <- unlist(neighbors)
adj <- sparseMatrix(
  i = row_idx,
  j = col_idx,
  dims = c(n_cells, n_cells),
  x = 1
)

# 2. Prepare result container
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_year_key <- paste(cell_data$id, cell_data$year)
res_list <- vector("list", length(neighbor_source_vars))
names(res_list) <- neighbor_source_vars

# 3. Compute neighbor stats by year in a loop
years <- sort(unique(cell_data$year))
for (var in neighbor_source_vars) {
  # Initialize output matrices
  max_mat <- numeric(nrow(cell_data))
  min_mat <- numeric(nrow(cell_data))
  mean_mat <- numeric(nrow(cell_data))
  
  for (yr in years) {
    idx_year <- which(cell_data$year == yr)
    vals <- cell_data[[var]][idx_year]
    
    # Convert to dense vector (cells order)
    v <- rep(NA_real_, n_cells)
    v[id_to_idx[cell_data$id[idx_year]]] <- vals
    
    # Compute neighbor values for each cell
    # Extract indices of non-NA neighbors efficiently
    for (cell in which(!is.na(v))) {
      neigh_idx <- neighbors[[cell]]
      neigh_vals <- v[neigh_idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        row_pos <- idx_year[which(cell_data$id[idx_year] == id_order[cell])]
        max_mat[row_pos] <- max(neigh_vals)
        min_mat[row_pos] <- min(neigh_vals)
        mean_mat[row_pos] <- mean(neigh_vals)
      }
    }
  }
  
  res_list[[var]] <- data.table(
    id = cell_data$id,
    year = cell_data$year,
    paste0(var, "_ngb_max") := max_mat,
    paste0(var, "_ngb_min") := min_mat,
    paste0(var, "_ngb_mean") := mean_mat
  )
}

# 4. Merge results into original data
for (var in neighbor_source_vars) {
  cell_data <- cbind(cell_data, res_list[[var]][, -c("id", "year")])
}

# Random Forest prediction step remains unchanged
```

---

**Performance Impact**  
- Eliminates repeated giant `lapply` calls.
- Works year-by-year to keep memory manageable.
- Preserves original numerical estimand and trained RF model.
- Estimated runtime drops from 86+ hours to a few hours (depending on I/O and sparse handling).  

Further optimization: Parallelize the inner year loops (`future.apply`), or precompute neighbor indices as vectors to reduce repeated lookups.
END_RESPONSE>>>
