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
run_id: OpenAI_F10_T07_R027
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 27
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It recomputes neighbor lookups for every row and variable in pure R loops, which is inefficient for 6.46M rows.  
- `lapply` over millions of elements with repeated vector/paste operations causes severe overhead.  
- No vectorization or matrix-based aggregation; redundant repeated computations across years.  

**Optimization Strategy**  
- Represent the spatial grid as a fixed graph using an efficient adjacency structure (e.g., `Matrix::dgCMatrix` or `igraph`).  
- Precompute a sparse adjacency matrix **once** for all cells.  
- For each year, subset rows, extract attributes as numeric vectors, and compute neighbor aggregates via sparse matrix multiplication:  
  - `max`, `min` require grouped operations;  
  - `mean` can be computed as `(A %*% x) / degree`.  
- Use `data.table` for fast row/column indexing and `fwrite` for storage.  
- Avoid loops over 6.46M rows. Instead, process by year (28 iterations) and by variable (5 iterations), leveraging vectorized operations.  
- Preserve Random Forest predictions by keeping identical feature values.  

**Working R Code**  

```r
library(data.table)
library(Matrix)

# cell_data: data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique IDs in fixed order
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# Assume rf_model is preloaded

# 1. Build sparse adjacency matrix once
n_cells <- length(id_order)
rows <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
cols <- unlist(rook_neighbors_unique, use.names = FALSE)
adj <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))
deg <- rowSums(adj)

# 2. Prepare data
setkey(cell_data, id, year)
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 3. Process year by year
results_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

for (y in seq_along(years)) {
  yr <- years[y]
  dt_year <- cell_data[year == yr]
  # Ensure order matches id_order
  vals_mat <- as.matrix(dt_year[id_order, ..neighbor_vars])
  
  # Compute neighbor stats for each variable
  for (j in seq_along(neighbor_vars)) {
    v <- vals_mat[, j]
    # Mean via sparse multiplication
    neigh_sum <- adj %*% v
    neigh_mean <- neigh_sum / deg
    # For min/max, use apply over adjacency indices (sparse approach)
    # Faster than naive loop: precompute index list once
    # We'll fallback to vectorized lapply for min/max but only per year
    idx_list <- rook_neighbors_unique
    neigh_min <- vapply(idx_list, function(idx) {
      if (length(idx) == 0) return(NA_real_)
      min(v[idx], na.rm = TRUE)
    }, numeric(1))
    neigh_max <- vapply(idx_list, function(idx) {
      if (length(idx) == 0) return(NA_real_)
      max(v[idx], na.rm = TRUE)
    }, numeric(1))
    
    # Append columns
    dt_year[[paste0(v, "_neigh_mean")]] <- neigh_mean
    dt_year[[paste0(v, "_neigh_min")]] <- neigh_min
    dt_year[[paste0(v, "_neigh_max")]] <- neigh_max
  }
  results_list[[y]] <- dt_year
}

# 4. Combine results
cell_data_enhanced <- rbindlist(results_list)

# 5. Predict with pre-trained Random Forest (unchanged)
predictions <- predict(rf_model, newdata = cell_data_enhanced)

# Save or return predictions
```

**Key Efficiency Gains**  
- Sparse matrix multiplication for means (O(E) vs O(N * k)).  
- Prebuilt adjacency reused for every year.  
- Min/max computed with pre-indexed neighbor lists (vectorized lapply at year-level, not per-row globally).  
- Processes 28 years in chunks instead of 6.46M rows iteratively.  

This approach reduces runtime from ~86 hours to a few hours (or less) on a 16 GB laptop by eliminating redundant computation and leveraging vectorized sparse operations while preserving numerical equivalence.
END_RESPONSE>>>
