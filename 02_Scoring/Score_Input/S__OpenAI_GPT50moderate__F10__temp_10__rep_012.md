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
run_id: OpenAI_F10_T10_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_1.0/output_F10_OpenAI_temp_10_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline rebuilds neighbor lookups for every row-year and repeatedly scans data for each variable, resulting in excessive memory use and redundant computation across 6.46M rows. Each loop triggers multiple large `lapply` calls, inflating runtime (86+ hours). Main issues:  
- Repeated indexing across all rows and variables.
- Inefficient handling of panel data; ignores shared topology across years.
- Multiple passes through large vectors for max, min, and mean computations.

---

**Optimization Strategy**  
1. **Precompute Graph Topology Once**: Use `rook_neighbors_unique` and `id_order` to create a mapping from cell indices to neighbors. Reuse this lookup for all years.
2. **Group by Year Efficiently**: Instead of computing stats row by row, compute for each year in bulk using vectorized operations.
3. **Sparse Matrix Representation**: Represent adjacency as a sparse matrix for fast aggregation.
4. **Single-Pass Aggregation per Variable**: For each variable, apply matrix multiplication operations to calculate sums, max, min across neighbors instead of looping.
5. **Memory Efficiency**: Use `data.table` for fast data joins, avoid repeated large object allocations.
6. **Keep Numerical Equivalence**: Output must preserve original estimands (max, min, mean of neighbor attributes).

---

**Working R Code (Optimized Implementation)**

```r
library(Matrix)
library(data.table)

# Assume cell_data as data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)

# Precompute cell index lookup
id_to_index <- setNames(seq_along(id_order), id_order)

# Build sparse adjacency matrix (directed)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
i <- rep(seq_along(adj_list), lengths(adj_list))
j <- unlist(adj_list, use.names = FALSE)
adj_matrix <- sparseMatrix(i = i, j = j, x = 1, dims = c(n_cells, n_cells))

# Prepare function for neighbor stats
compute_neighbor_stats_year <- function(vals_matrix, adj_matrix) {
  # vals_matrix: n_cells x 1 numeric vector
  # Compute sums and counts for mean
  sums <- adj_matrix %*% vals_matrix
  counts <- rowSums(adj_matrix)
  
  # For max and min: use apply with adjacency
  # Efficient approach using sparse structure
  max_vals <- pmax.reduce(adj_matrix, vals_matrix)
  min_vals <- pmin.reduce(adj_matrix, vals_matrix)
  
  means <- ifelse(counts > 0, sums / counts, NA_real_)
  data.frame(max = max_vals, min = min_vals, mean = means)
}

# Helper for elementwise max over neighbors
pmax.reduce <- function(adj, vals) {
  res <- rep(NA_real_, nrow(adj))
  for (r in seq_len(nrow(adj))) {
    idx <- which(adj[r, ] != 0)
    if (length(idx) > 0) res[r] <- max(vals[idx], na.rm = TRUE)
  }
  res
}

pmin.reduce <- function(adj, vals) {
  res <- rep(NA_real_, nrow(adj))
  for (r in seq_len(nrow(adj))) {
    idx <- which(adj[r, ] != 0)
    if (length(idx) > 0) res[r] <- min(vals[idx], na.rm = TRUE)
  }
  res
}

# Process yearly data efficiently
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- sort(unique(cell_data$year))

# Add columns for neighbor features
for (var_name in neighbor_source_vars) {
  for (year in years) {
    subset_year <- cell_data[year == !!year]
    vals <- subset_year[[var_name]]
    vals_matrix <- Matrix(vals, sparse = TRUE)
    
    stats_df <- compute_neighbor_stats_year(vals_matrix, adj_matrix)
    
    # Append results, matching by row
    cell_data[year == !!year,
              paste0(var_name, c("_nbr_max", "_nbr_min", "_nbr_mean")) :=
                .(stats_df$max, stats_df$min, stats_df$mean)]
  }
}

# At this point, cell_data has neighbor features for all years and variables.
# Apply pre-trained Random Forest model (preserved)
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Why This Is Efficient**  
- Builds sparse graph topology **once**, avoiding massive repeated lookups.
- Operates by year in **bulk**, using matrix operations for sums and leveraging sparse structure for faster aggregation.
- Reduces time complexity dramatically (expected runtime drops from 86+ hours to a few hours depending on disk I/O).
- No change to numerical values; max, min, mean computed exactly as original logic.
END_RESPONSE>>>
