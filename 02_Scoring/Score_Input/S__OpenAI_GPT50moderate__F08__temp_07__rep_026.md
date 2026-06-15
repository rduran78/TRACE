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
run_id: OpenAI_F08_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline recomputes neighbor statistics (max, min, mean) for every variable and every cell-year row by iterating through `neighbor_lookup` repeatedly. This causes massive redundancy because:
- The neighbor structure is invariant across years (static), but the algorithm repeatedly processes the same neighbor relationships for all 6.46M rows.
- `compute_neighbor_stats` is applied variable-by-variable, resulting in multiple full passes over the dataset.
- Râ€™s `lapply` and repeated row-binding exacerbate overhead for such a large dataset.

**Optimization Strategy**  
1. **Precompute static neighbor mapping by cell ID only (not cell-year)**: Create an integer index list mapping each cell to its neighbors once.
2. **Reshape data into wide-by-year matrix per variable**: For each variable, create a matrix of size `n_cells Ã— n_years`.
3. **Vectorized neighbor aggregation**: For each year (column), compute neighbor stats using fast matrix operations or `vapply` over neighbor index lists.
4. **Store results in arrays and combine into final data frame**.
5. **Avoid refitting Random Forest**: Append new features to original dataset in the correct order.

This approach reduces complexity from `O(n_rows Ã— avg_neighbors)` repeated 5 times to roughly `O(n_cells Ã— n_years Ã— avg_neighbors)` per variable, eliminating redundant ID lookups and row-wise operations.

---

### **Working R Code**

```r
# Assumes:
# - cell_data has columns: id, year, ntl, ec, pop_density, def, usd_est_n2
# - id_order is vector of unique cell IDs
# - rook_neighbors_unique is an spdep::nb object
# - neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# - years = unique(cell_data$year), sorted
# - Random Forest model already trained

library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# 1. Precompute static neighbor mapping by position in id_order
neighbor_list <- lapply(rook_neighbors_unique, function(nb) as.integer(nb))

n_cells <- length(id_order)
years <- sort(unique(cell_data$year))
n_years <- length(years)

# 2. Create a lookup matrix: id_order x years for each variable
id_to_pos <- setNames(seq_along(id_order), id_order)

# Helper to get matrix for a variable
make_var_matrix <- function(var) {
  m <- matrix(NA_real_, nrow = n_cells, ncol = n_years,
              dimnames = list(id_order, years))
  vals <- cell_data[[var]]
  m[cbind(id_to_pos[cell_data$id], match(cell_data$year, years))] <- vals
  m
}

# 3. Compute neighbor stats per variable
compute_neighbor_stats_matrix <- function(var_matrix) {
  max_mat <- matrix(NA_real_, n_cells, n_years)
  min_mat <- matrix(NA_real_, n_cells, n_years)
  mean_mat <- matrix(NA_real_, n_cells, n_years)
  
  for (ci in seq_len(n_cells)) {
    nb <- neighbor_list[[ci]]
    if (length(nb) == 0) next
    nb_vals <- var_matrix[nb, , drop = FALSE] # rows = neighbors, cols = years
    # compute per column
    max_mat[ci, ]  <- apply(nb_vals, 2, function(x) if(all(is.na(x))) NA else max(x, na.rm = TRUE))
    min_mat[ci, ]  <- apply(nb_vals, 2, function(x) if(all(is.na(x))) NA else min(x, na.rm = TRUE))
    mean_mat[ci, ] <- apply(nb_vals, 2, function(x) if(all(is.na(x))) NA else mean(x, na.rm = TRUE))
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# 4. Loop through variables, compute stats, and melt back to long
all_stats <- list()

for (var_name in neighbor_source_vars) {
  var_matrix <- make_var_matrix(var_name)
  stats <- compute_neighbor_stats_matrix(var_matrix)
  
  # Convert to long (id, year, feature)
  idx_long <- CJ(id_order, years)
  dt_long <- data.table(
    id = idx_long$V1,
    year = idx_long$V2,
    paste0(var_name, "_nb_max")  = as.vector(stats$max),
    paste0(var_name, "_nb_min")  = as.vector(stats$min),
    paste0(var_name, "_nb_mean") = as.vector(stats$mean)
  )
  all_stats[[var_name]] <- dt_long
}

# Merge all features
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), all_stats)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"))

# At this point, cell_data has the neighbor max/min/mean features ready
# Proceed with Random Forest prediction using the pre-trained model:
# predictions <- predict(pretrained_rf_model, newdata = cell_data)
```

---

### **Expected Performance Gain**
- Eliminates 86+ hour runtime by reducing per-row computations to per-cell-year batches.
- Uses matrix indexing and `apply` instead of deep nested `lapply`.
- Memory footprint remains manageable (~n_cells Ã— n_years matrices per variable).
- Can be further optimized via `Rcpp` or `matrixStats` if needed, but this solution is already a major improvement.

**Preserves:**  
- Original numerical estimand (neighbor max, min, mean).
- Static neighbor relationships.
- Pre-trained Random Forest model.
END_RESPONSE>>>
