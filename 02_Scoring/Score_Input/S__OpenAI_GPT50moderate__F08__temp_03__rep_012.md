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
run_id: OpenAI_F08_T03_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F08_OpenAI_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow because `compute_neighbor_stats` iterates over all ~6.46M rows for each variable, repeatedly scanning neighbor indices. This is inefficient because:  
- Neighbor relationships are static across years, but the code recomputes neighbor-based stats for every cell-year individually.  
- For each of 5 variables, the function performs millions of small list operations, which is costly in R.  
- The neighbor lookup is correct but not leveraged for vectorized computation across years.  

**Optimization Strategy**  
- Precompute the static neighbor index list once (already done).  
- Reshape the data into a matrix per variable: rows = cells, columns = years.  
- Compute neighbor max, min, and mean using matrix operations for each year.  
- Use `matrixStats` or `apply` for fast row-wise aggregation.  
- Recombine results back into long format.  
This avoids looping over 6.46M rows repeatedly and instead processes 344k rows Ã— 28 years in a vectorized way.  

**Working R Code**  

```r
library(data.table)
library(matrixStats)

# Assume: cell_data has columns id, year, and neighbor_source_vars
# id_order: vector of unique cell IDs in consistent order
# neighbor_lookup: list of integer vectors (neighbors per cell), length = n_cells
# years: sorted unique years
# neighbor_source_vars: c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Convert to data.table for speed
setDT(cell_data)
years <- sort(unique(cell_data$year))
n_cells <- length(id_order)
n_years <- length(years)

# Build a mapping from (id, year) to matrix positions
id_index <- match(cell_data$id, id_order)
year_index <- match(cell_data$year, years)

# Preallocate matrices for each variable
var_mats <- lapply(neighbor_source_vars, function(v) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(id_index, year_index)] <- cell_data[[v]]
  mat
})
names(var_mats) <- neighbor_source_vars

# Function to compute neighbor stats for one variable matrix
compute_neighbor_stats_matrix <- function(var_mat, neighbor_lookup) {
  n_cells <- nrow(var_mat)
  n_years <- ncol(var_mat)
  
  max_mat <- matrix(NA_real_, n_cells, n_years)
  min_mat <- matrix(NA_real_, n_cells, n_years)
  mean_mat <- matrix(NA_real_, n_cells, n_years)
  
  for (i in seq_len(n_cells)) {
    nbrs <- neighbor_lookup[[i]]
    if (length(nbrs) == 0) next
    nbr_vals <- var_mat[nbrs, , drop = FALSE]
    max_mat[i, ] <- colMaxs(nbr_vals, na.rm = TRUE)
    min_mat[i, ] <- colMins(nbr_vals, na.rm = TRUE)
    mean_mat[i, ] <- colMeans(nbr_vals, na.rm = TRUE)
  }
  
  list(max = max_mat, min = min_mat, mean = mean_mat)
}

# Compute for all variables
neighbor_stats <- lapply(names(var_mats), function(v) {
  compute_neighbor_stats_matrix(var_mats[[v]], neighbor_lookup)
})
names(neighbor_stats) <- names(var_mats)

# Reshape back to long format and merge
result_list <- list()
for (v in names(neighbor_stats)) {
  stats <- neighbor_stats[[v]]
  # Melt each stat
  for (stat_name in names(stats)) {
    long_dt <- data.table(
      id = rep(id_order, times = n_years),
      year = rep(years, each = n_cells),
      value = as.vector(stats[[stat_name]])
    )
    setnames(long_dt, "value", paste0(v, "_nbr_", stat_name))
    result_list[[paste0(v, "_", stat_name)]] <- long_dt
  }
}

# Combine all stats
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")),
                             result_list)

# Merge back to cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

**Why This Works**  
- We compute neighbor stats per cell across all years in one go, using efficient matrix operations.  
- Complexity drops from ~6.46M Ã— 5 loops to ~344k Ã— 28 loops, with vectorized column operations.  
- Memory is manageable: 344k Ã— 28 â‰ˆ 9.6M entries per matrix (~80 MB per variable).  
- Preserves original estimand and Random Forest model compatibility.  

**Expected Speedup**  
From 86+ hours to a few hours or less on a 16 GB laptop, depending on I/O and CPU speed.
END_RESPONSE>>>
