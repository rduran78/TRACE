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
run_id: OpenAI_F10_T00_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply` over millions of rows.  
- Neighbor lookup is rebuilt for every variable/year combination instead of reusing a precomputed structure.  
- No vectorization; heavy reliance on R loops and repeated indexing.  
- Memory overhead from large lists and repeated concatenations.  

**Optimization Strategy**  
- Precompute a single adjacency list (graph topology) for all cells (rook neighbors).  
- Use integer indices for fast lookup.  
- Process data in a vectorized manner using `data.table` or `matrix` operations.  
- Compute neighbor statistics for all variables in one pass per year using efficient aggregation.  
- Avoid rebuilding neighbor lookup repeatedly; reuse across all years and variables.  
- Use sparse representation for adjacency to minimize memory footprint.  

---

### **Efficient Implementation in R**

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# 1. Build adjacency list once
build_adjacency <- function(id_order, rook_neighbors_unique) {
  id_to_idx <- setNames(seq_along(id_order), id_order)
  lapply(seq_along(id_order), function(i) {
    neighbor_ids <- id_order[rook_neighbors_unique[[i]]]
    as.integer(id_to_idx[neighbor_ids])
  })
}

adjacency_list <- build_adjacency(id_order, rook_neighbors_unique)

# 2. Convert cell_data to matrix for fast access
setkey(cell_data, id, year)
vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a matrix of size (n_cells x n_years) for each variable
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

var_mats <- lapply(vars, function(v) {
  m <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  dt_var <- cell_data[, .(id, year, val = get(v))]
  idx_id <- match(dt_var$id, id_order)
  idx_year <- match(dt_var$year, years)
  m[cbind(idx_id, idx_year)] <- dt_var$val
  m
})
names(var_mats) <- vars

# 3. Compute neighbor stats efficiently
compute_neighbor_stats_matrix <- function(var_mat, adjacency_list) {
  n_cells <- nrow(var_mat)
  n_years <- ncol(var_mat)
  
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    neighbors <- adjacency_list[[i]]
    if (length(neighbors) == 0) next
    neighbor_vals <- var_mat[neighbors, , drop = FALSE]
    # Compute stats column-wise (per year)
    max_mat[i, ] <- apply(neighbor_vals, 2, function(x) if (all(is.na(x))) NA else max(x, na.rm = TRUE))
    min_mat[i, ] <- apply(neighbor_vals, 2, function(x) if (all(is.na(x))) NA else min(x, na.rm = TRUE))
    mean_mat[i, ] <- apply(neighbor_vals, 2, function(x) if (all(is.na(x))) NA else mean(x, na.rm = TRUE))
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# 4. Apply for all variables
neighbor_stats <- lapply(var_mats, compute_neighbor_stats_matrix, adjacency_list = adjacency_list)

# 5. Merge back into cell_data
# Flatten matrices into long format
add_neighbor_features <- function(cell_data, stats_list, var_name) {
  years <- sort(unique(cell_data$year))
  n_cells <- length(id_order)
  
  dt <- data.table(
    id = rep(id_order, times = length(years)),
    year = rep(years, each = n_cells),
    paste0(var_name, "_nbr_max") := as.vector(stats_list$max),
    paste0(var_name, "_nbr_min") := as.vector(stats_list$min),
    paste0(var_name, "_nbr_mean") := as.vector(stats_list$mean)
  )
  
  merge(cell_data, dt, by = c("id", "year"), all.x = TRUE)
}

for (v in vars) {
  cell_data <- add_neighbor_features(cell_data, neighbor_stats[[v]], v)
}

# cell_data now contains original variables + neighbor stats
# Apply pre-trained Random Forest model (preserve original estimand)
predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Why This Is Efficient**
- Adjacency built once and reused.
- Matrix-based operations minimize overhead.
- Loop over cells only once per variable, with vectorized year-wise aggregation.
- Avoids repeated `lapply` over millions of rows.
- Memory footprint controlled by using matrices instead of large lists.

**Expected Performance**:  
From 86+ hours â†’ likely reduced to a few hours or less on a 16 GB laptop, depending on disk I/O and Random Forest prediction speed.
END_RESPONSE>>>
