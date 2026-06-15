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
run_id: OpenAI_F10_T07_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
- The current workflow is extremely slow because:
  - For each of ~6.46M rows, `build_neighbor_lookup` constructs neighbor indices dynamically, causing repeated string operations and lookups.
  - `compute_neighbor_stats` iterates through all rows and uses `lapply` repeatedly, leading to high R-level overhead.
- Complexity grows with both number of cells and years; memory and CPU overhead are severe.
- Key inefficiency: repeated computation per row/year instead of vectorized or graph-based aggregation.
- Graph topology (rook neighbors) is static across years, but current implementation recomputes indexing repeatedly.

---

**Optimization Strategy**  
- Build the neighbor index **once** using cell IDs, reuse across all years.
- Represent data in **matrix form**: rows = cells, columns = years, so neighbor aggregation can use fast matrix operations.
- Use **sparse adjacency matrix** for rook neighbors to compute max, min, mean efficiently with vectorized ops.
- Avoid `lapply` over millions of rows; rely on `Matrix` (sparse) operations or efficient loops over neighbors.
- Compute stats per variable using adjacency structure and apply over all years in one pass.
- Append results back to long-format data after computing in matrix form.
- Preserve numerical equivalence with the original max, min, mean calculations.

---

**Efficient R Implementation**

```r
library(Matrix)
library(data.table)

# Assume: cell_data (data.table) with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of cell ids in same order as adjacency list
# rook_neighbors_unique: list of integer neighbor indices (spdep nb object)
# years: unique years sorted
# neighbor_source_vars: c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 1. Prepare adjacency as sparse matrix
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), sapply(adj_list, length))
cols <- unlist(adj_list)
adj <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 2. Index mapping
id_to_idx <- setNames(seq_along(id_order), id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# 3. Convert data to cell x year matrices for each var
setkey(cell_data, id, year)
# Ensure full panel
cell_data <- cell_data[CJ(id = id_order, year = years)]
matrices <- lapply(neighbor_source_vars, function(v) {
  m <- matrix(cell_data[[v]], nrow = n_cells, ncol = n_years, byrow = FALSE)
  m
})
names(matrices) <- neighbor_source_vars

# 4. Function to compute neighbor stats for one variable using adjacency
compute_stats_matrix <- function(var_mat, adj) {
  # For mean: sum of neighbors / neighbor count
  neighbor_count <- rowSums(adj)
  
  # Initialize outputs
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Loop over years (fast, n_years = 28)
  for (j in seq_len(n_years)) {
    vals <- var_mat[, j]
    # For mean
    sum_vals <- as.numeric(adj %*% vals)
    mean_vals <- ifelse(neighbor_count > 0, sum_vals / neighbor_count, NA_real_)
    mean_mat[, j] <- mean_vals
    
    # For max and min: we need explicit neighbor iteration
    # Optimize by using adjacency list
    max_vals <- rep(NA_real_, n_cells)
    min_vals <- rep(NA_real_, n_cells)
    for (i in seq_len(n_cells)) {
      nb <- adj_list[[i]]
      if (length(nb) > 0) {
        nb_vals <- vals[nb]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          max_vals[i] <- max(nb_vals)
          min_vals[i] <- min(nb_vals)
        }
      }
    }
    max_mat[, j] <- max_vals
    min_mat[, j] <- min_vals
  }
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# 5. Compute for all variables
neighbor_results <- lapply(names(matrices), function(v) {
  compute_stats_matrix(matrices[[v]], adj)
})
names(neighbor_results) <- names(matrices)

# 6. Bind results back to long format
result_dt <- data.table(id = rep(id_order, times = n_years),
                        year = rep(years, each = n_cells))

for (v in names(neighbor_results)) {
  res <- neighbor_results[[v]]
  result_dt[[paste0(v, "_nb_max")]]  <- as.vector(res$max)
  result_dt[[paste0(v, "_nb_min")]]  <- as.vector(res$min)
  result_dt[[paste0(v, "_nb_mean")]] <- as.vector(res$mean)
}

# 7. Merge with original data
cell_data <- merge(cell_data, result_dt, by = c("id", "year"), all.x = TRUE)

# cell_data now has neighbor stats added, ready for prediction
# Preserve trained RF model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Key Gains**
- Graph topology built **once**.
- Mean computed via sparse matrix multiplication (fast).
- Max/min computed efficiently with adjacency list per year (minimal overhead).
- Loops reduced from millions to **n_cells * n_years** only for critical max/min step, which is unavoidable.
- Memory efficient: matrix form avoids heavy string manipulation and repeated lookups.
- Should reduce runtime from 86+ hours to **hours or less**, limited by max/min loop.

**Preserves numerical equivalence** with original approach and uses the same trained Random Forest model.
END_RESPONSE>>>
