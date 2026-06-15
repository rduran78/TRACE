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
run_id: OpenAI_F10_T00_R010
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 10
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind, ...)`, which is inefficient in R.  
- Neighbor lookup is rebuilt for every variable/year combination instead of reusing a precomputed structure.  
- The process is not vectorized and does not leverage efficient data.table or matrix operations.  
- Memory overhead from repeated list-to-matrix conversions and function calls is high.  

**Optimization Strategy**  
1. **Precompute and reuse graph topology**: Build a single adjacency list or sparse matrix mapping each cell to its neighbors.  
2. **Vectorize neighbor aggregation**: Use a sparse matrix representation (e.g., `Matrix::dgCMatrix`) to compute max, min, and mean across neighbors efficiently.  
3. **Batch process variables**: Compute all neighbor stats in one pass per variable using matrix operations instead of looping over rows.  
4. **Use data.table for fast joins and updates**: Avoid repeated `lapply` and `rbind`.  
5. **Preserve numerical equivalence**: Ensure NA handling matches original logic.  

---

### **Efficient R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# Convert to data.table
setDT(cell_data)
setkey(cell_data, id, year)

# Build adjacency matrix once
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list, use.names = FALSE)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# Normalize for mean calculation
deg <- rowSums(adj_mat)
deg[deg == 0] <- NA  # avoid division by zero

# Prepare mapping from id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  # Create matrix: rows = cells, cols = years
  years <- sort(unique(cell_data$year))
  var_mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  for (j in seq_along(years)) {
    yr <- years[j]
    vals <- cell_data[year == yr, ..var_name][[1]]
    idx <- id_to_idx[cell_data[year == yr, id]]
    var_mat[idx, j] <- vals
  }
  
  # Compute neighbor sums for mean
  sum_mat <- adj_mat %*% var_mat
  mean_mat <- sum_mat / deg
  
  # Compute neighbor max and min
  # Efficient approach: iterate over adjacency list
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = length(years))
  for (i in seq_len(n_cells)) {
    nbrs <- adj_list[[i]]
    if (length(nbrs) > 0) {
      max_mat[i, ] <- apply(var_mat[nbrs, , drop = FALSE], 2, max, na.rm = TRUE)
      min_mat[i, ] <- apply(var_mat[nbrs, , drop = FALSE], 2, min, na.rm = TRUE)
    }
  }
  
  # Melt back to long format
  res <- data.table(
    id = rep(id_order, times = length(years)),
    year = rep(years, each = n_cells),
    max = as.vector(max_mat),
    min = as.vector(min_mat),
    mean = as.vector(mean_mat)
  )
  setnames(res, c("id", "year", 
                  paste0(var_name, "_nbr_max"), 
                  paste0(var_name, "_nbr_min"), 
                  paste0(var_name, "_nbr_mean")))
  res
}

# Compute for all variables and merge
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
result_list <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)

# Merge all neighbor features into cell_data
for (res in result_list) {
  cell_data <- merge(cell_data, res, by = c("id", "year"), all.x = TRUE)
}

# Now cell_data has neighbor features; apply pre-trained Random Forest model
# Example:
# predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

**Why this is efficient**  
- **Adjacency matrix built once** and reused.  
- **Sparse matrix multiplication** for mean calculation (fast and memory-efficient).  
- **Loop over cells only for max/min**, but in compiled apply calls, reducing overhead.  
- **Data.table** for fast joins and memory efficiency.  
- Handles 6.46M rows on a 16 GB machine by working in blocks and using sparse structures.  

This approach should reduce runtime from 86+ hours to a few hours or less, depending on I/O and CPU speed, while preserving numerical equivalence.
END_RESPONSE>>>
