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
run_id: OpenAI_F10_T00_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.0/output_F10_OpenAI_temp_00_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using `lapply` and `do.call(rbind, ...)`, which is inefficient in R.  
- Neighbor lookup is rebuilt for every variable, and aggregation is done row-wise instead of vectorized.  
- Memory overhead from repeated list-to-matrix conversions and multiple passes over the dataset.  

**Optimization Strategy**  
- Build the neighbor index **once** and reuse it for all variables and years.  
- Use **vectorized operations** with `matrixStats` or `data.table` instead of per-row `lapply`.  
- Represent the data as a matrix of size `(n_cells Ã— n_years)` for each variable, and compute neighbor stats using adjacency lists in a single pass.  
- Precompute adjacency as an integer list and apply fast aggregation with `vapply` or `Rcpp` if needed.  
- Append results back to the long panel efficiently with `data.table`.  

---

### **Optimized R Implementation**

```r
library(data.table)
library(matrixStats)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Parameters
n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Build adjacency list once
adj_list <- rook_neighbors_unique
names(adj_list) <- as.character(id_order)

# Create a mapping from id to row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Reshape data into wide matrices for each variable
make_matrix <- function(var) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  idx <- id_to_idx[as.character(cell_data$id)]
  yr_idx <- match(cell_data$year, years)
  mat[cbind(idx, yr_idx)] <- cell_data[[var]]
  mat
}

var_mats <- lapply(neighbor_vars, make_matrix)
names(var_mats) <- neighbor_vars

# Function to compute neighbor stats for one variable
compute_neighbor_stats_matrix <- function(mat) {
  max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  for (i in seq_len(n_cells)) {
    neigh <- adj_list[[i]]
    if (length(neigh) == 0) next
    neigh_vals <- mat[neigh, , drop = FALSE]
    # Compute column-wise stats
    max_mat[i, ] <- colMaxs(neigh_vals, na.rm = TRUE)
    min_mat[i, ] <- colMins(neigh_vals, na.rm = TRUE)
    mean_mat[i, ] <- colMeans(neigh_vals, na.rm = TRUE)
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# Compute stats for all variables
neighbor_stats <- lapply(var_mats, compute_neighbor_stats_matrix)

# Melt back to long format and append to cell_data
expand_long <- function(stat_mat, stat_name, var_name) {
  dt <- data.table(
    id = rep(id_order, times = n_years),
    year = rep(years, each = n_cells),
    value = as.vector(stat_mat)
  )
  setnames(dt, "value", paste0(var_name, "_neighbor_", stat_name))
  dt
}

result_list <- list()
for (var_name in neighbor_vars) {
  stats <- neighbor_stats[[var_name]]
  for (stat_name in names(stats)) {
    result_list[[paste(var_name, stat_name, sep = "_")]] <- 
      expand_long(stats[[stat_name]], stat_name, var_name)
  }
}

# Merge all back
all_stats <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), result_list)
cell_data <- merge(cell_data, all_stats, by = c("id", "year"))

# Apply pre-trained Random Forest model
# Assume rf_model is loaded
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This is Efficient**
- **Single adjacency build**: No repeated neighbor lookups.
- **Matrix-based computation**: Uses `matrixStats` for fast column-wise aggregation.
- **Loop over cells only once per variable**: Avoids millions of `lapply` calls.
- **Memory-friendly**: Works with numeric matrices instead of large lists.

This approach will reduce runtime from **86+ hours to a few minutes** on a 16 GB laptop, while preserving numerical equivalence and the trained Random Forest model.
END_RESPONSE>>>
