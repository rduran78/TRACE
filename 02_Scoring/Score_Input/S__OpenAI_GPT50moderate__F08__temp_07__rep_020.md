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
run_id: OpenAI_F08_T07_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because `compute_neighbor_stats` iterates over all 6.46M rows and recomputes neighbor lookups for each year-variable combination. This results in repeated traversals and redundant aggregation. The neighbor structure is static across years, but calculations are re-done per cell-year in a highly inefficient way (essentially O(N * k * #vars * #years), with N â‰ˆ 6.46M). Additionally, using `lapply` with row-wise operations in R on millions of rows is memory and CPU expensive.

---

**Optimization Strategy**  
1. **Exploit Static Neighbor Structure:** Compute a fixed neighbor index for each cell (not cell-year).  
2. **Vectorize Yearly Computations:** For each year and variable, compute neighbor aggregates in a single grouped operation instead of row-by-row loops.  
3. **Use Matrix Representation:** Store variables in a cell Ã— year matrix to allow fast column-wise operations.  
4. **Avoid Large Repeated `paste` and Lookups:** Precompute an integer neighbor list.  
5. **Process in Chunks (Optional):** For memory constraints, handle one variable at a time but vectorized.  

---

**Optimized R Code**

```r
# Precompute neighbor index once (static across years)
build_static_neighbor_lookup <- function(id_order, neighbors) {
  id_to_idx <- setNames(seq_along(id_order), id_order)
  lapply(neighbors, function(nbrs) as.integer(id_to_idx[nbrs]))
}

# Compute stats for all years efficiently
compute_neighbor_stats_matrix <- function(var_matrix, neighbor_lookup) {
  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)
  
  max_mat <- matrix(NA_real_, n_cells, n_years)
  min_mat <- matrix(NA_real_, n_cells, n_years)
  mean_mat <- matrix(NA_real_, n_cells, n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- neighbor_lookup[[i]]
    if (length(nbrs) == 0) next
    nbr_vals <- var_matrix[nbrs, , drop = FALSE]
    # Compute across rows for each column (year)
    max_mat[i, ]  <- apply(nbr_vals, 2, max, na.rm = TRUE)
    min_mat[i, ]  <- apply(nbr_vals, 2, min, na.rm = TRUE)
    mean_mat[i, ] <- apply(nbr_vals, 2, mean, na.rm = TRUE)
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# --- Main Pipeline ---
# Assumptions:
# cell_data: data.frame with columns id, year, and vars
# id_order: vector of unique cell IDs
# rook_neighbors_unique: spdep nb object

neighbor_lookup <- build_static_neighbor_lookup(id_order, rook_neighbors_unique)

years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# Reshape to cell Ã— year matrix for each variable
library(data.table)
dt <- as.data.table(cell_data)
setkey(dt, id, year)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  var_matrix <- matrix(NA_real_, n_cells, n_years)
  var_matrix[as.integer(factor(dt$id, levels = id_order)),
             as.integer(factor(dt$year, levels = years))] <- dt[[var_name]]
  
  stats <- compute_neighbor_stats_matrix(var_matrix, neighbor_lookup)
  
  # Convert back and bind to dt
  for (stat_type in names(stats)) {
    stat_values <- as.vector(t(stats[[stat_type]])) # cell-year order
    new_col <- paste0(var_name, "_nbr_", stat_type)
    dt[, (new_col) := stat_values]
  }
}

# dt now contains original data + neighbor features
cell_data <- as.data.frame(dt)
```

---

**Why This Works**
- **Static neighbor lookup** avoids recomputing for every row-year.  
- **Matrix-based computation** reduces overhead by eliminating 6.46M row loops; instead, operations happen in vectorized chunks.  
- **Preserves trained Random Forest model** (same features, same estimand).  
- Expected runtime drops from **86+ hours to a few hours (or less)** on a standard laptop.  

**Memory Consideration:**  
- For 344,208 cells Ã— 28 years â‰ˆ 9.6M elements per matrix (â‰ˆ 76 MB per variable in double precision), easily fits in 16 GB RAM if processed sequentially per variable.
END_RESPONSE>>>
