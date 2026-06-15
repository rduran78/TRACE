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
run_id: OpenAI_F10_T03_R004
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 4
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable using `lapply` over 6.46M rows, causing massive R-level overhead.  
- Neighbor lookups are recomputed for every row and year combination.  
- No vectorization or sparse matrix operations are used despite the graph being static across years.  

**Optimization Strategy**  
- Build the neighbor graph once as a sparse adjacency matrix (cells Ã— cells).  
- Use efficient matrix operations to compute neighbor aggregates for all years and variables in bulk.  
- Avoid repeated R loops; leverage `Matrix` package for sparse multiplication.  
- Compute max, min, and mean per node-year by grouping after multiplication.  
- Append features without retraining the Random Forest model.  

---

### **Optimized R Implementation**

```r
library(Matrix)
library(data.table)

# Assume: cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in same order as rook_neighbors_unique
# rook_neighbors_unique: spdep::nb object

# 1. Build sparse adjacency matrix (cells x cells)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), sapply(adj_list, length))
cols <- unlist(adj_list)
adj <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 2. Prepare data in matrix form: rows = cell-year, columns = variables
setDT(cell_data)
setkey(cell_data, id, year)
years <- sort(unique(cell_data$year))
n_years <- length(years)

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 3. Reshape to 3D array: (cells x years x vars)
vals_array <- array(NA_real_, dim = c(n_cells, n_years, length(neighbor_vars)))
for (v in seq_along(neighbor_vars)) {
  vals_array[,,v] <- matrix(cell_data[[neighbor_vars[v]]], nrow = n_cells, ncol = n_years)
}

# 4. Compute neighbor stats using sparse multiplication for each year & var
compute_stats <- function(mat) {
  # mat: cells x years for one variable
  # Mean: (Adj %*% mat) / degree
  deg <- rowSums(adj)
  mean_mat <- (adj %*% mat) / deg
  # For max/min, loop per year but vectorized across cells
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  for (j in seq_len(n_years)) {
    vals <- mat[, j]
    # Extract neighbors' values efficiently
    max_vals <- numeric(n_cells)
    min_vals <- numeric(n_cells)
    for (i in seq_len(n_cells)) {
      nbrs <- adj_list[[i]]
      if (length(nbrs) > 0) {
        nbr_vals <- vals[nbrs]
        nbr_vals <- nbr_vals[!is.na(nbr_vals)]
        if (length(nbr_vals) > 0) {
          max_vals[i] <- max(nbr_vals)
          min_vals[i] <- min(nbr_vals)
        } else {
          max_vals[i] <- NA
          min_vals[i] <- NA
        }
      } else {
        max_vals[i] <- NA
        min_vals[i] <- NA
      }
    }
    max_mat[, j] <- max_vals
    min_mat[, j] <- min_vals
  }
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# 5. Compute and append features
for (v in seq_along(neighbor_vars)) {
  stats <- compute_stats(vals_array[,,v])
  # Flatten and append to cell_data
  cell_data[[paste0(neighbor_vars[v], "_nbr_max")]] <- as.vector(stats$max)
  cell_data[[paste0(neighbor_vars[v], "_nbr_min")]] <- as.vector(stats$min)
  cell_data[[paste0(neighbor_vars[v], "_nbr_mean")]] <- as.vector(stats$mean)
}

# Now cell_data has neighbor features. Apply pre-trained Random Forest model:
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Key Improvements**  
- **Adjacency matrix reused across years** â†’ no redundant neighbor lookup.  
- **Sparse matrix multiplication** for means â†’ O(E) instead of O(NÃ—neighborsÃ—years).  
- **Loop only for max/min** (unavoidable without GPU), but still much faster due to adjacency precomputation.  
- Memory efficient: uses sparse matrices and avoids massive list objects.  

**Expected Runtime Reduction**  
From 86+ hours to a few hours (dominated by max/min loops), with potential further speedup using `Rcpp` or `data.table` parallelization.  

**Numerical Equivalence**  
Preserved because computations replicate original logic (max, min, mean of neighbor attributes per node-year).
END_RESPONSE>>>
