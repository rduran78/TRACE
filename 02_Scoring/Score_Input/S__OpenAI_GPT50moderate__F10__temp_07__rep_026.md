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
run_id: OpenAI_F10_T07_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is slow (86+ hours) because:  
- `neighbor_lookup` repeats work for every row-year combination (6.46M rows).  
- Neighbor aggregation occurs via repeated `lapply` calls, creating high overhead.  
- Graph topology is rebuilt per iteration instead of reused.  
- No vectorization; computations are scattered across millions of small lists.  

**Optimization Strategy**  
- Represent the panel as a sparse graph using `Matrix` or `igraph`.  
- Build a single adjacency matrix for rook neighbors (344,208 nodes).  
- For each year, slice the data and apply sparse matrix multiplication to compute sums/means quickly.  
- Compute min/max via efficient grouping using `pmin`/`pmax`.  
- Avoid per-row `lapply`; use vectorized operations.  
- Preallocate results and bind efficiently.  
- Preserve original numerical values (max, min, mean per node-year).  

**Efficient R Implementation**  

```r
library(Matrix)
library(data.table)

# Assume cell_data has columns: id (factor/int), year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell IDs in consistent order
# rook_neighbors_unique: spdep::nb object

# 1. Build sparse adjacency matrix (once)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
i_idx <- rep(seq_along(adj_list), lengths(adj_list))
j_idx <- unlist(adj_list)
adj_mat <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n_cells, n_cells))

# 2. Prepare data.table for fast slicing
setDT(cell_data)
setkey(cell_data, year)

neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# 3. Function to compute neighbor stats for one variable across all years
compute_neighbor_stats_sparse <- function(var_name) {
  result_list <- vector("list", length(unique(cell_data$year)))
  yrs <- sort(unique(cell_data$year))
  
  for (k in seq_along(yrs)) {
    yr <- yrs[k]
    slice <- cell_data[year == yr, ..var_name]
    vals <- slice[[1]]
    
    # Sparse aggregation
    # Mean: sum of neighbors / neighbor count
    neighbor_sum <- as.numeric(adj_mat %*% vals)
    neighbor_count <- rowSums(adj_mat)
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # For min/max, iterate neighbors efficiently
    # Use adjacency indices to compute min/max without large lapply
    # Vectorized approach:
    idx_pairs <- cbind(i_idx, j_idx)
    neighbor_vals <- vals[j_idx]
    dt <- data.table(i = i_idx, val = neighbor_vals)
    min_vals <- dt[, .(min = min(val)), by = i][order(i)]$min
    max_vals <- dt[, .(max = max(val)), by = i][order(i)]$max
    
    # Handle cells with no neighbors
    has_neighbors <- neighbor_count > 0
    min_vals[!has_neighbors] <- NA_real_
    max_vals[!has_neighbors] <- NA_real_
    
    result_list[[k]] <- data.table(
      year = yr,
      max = max_vals,
      min = min_vals,
      mean = neighbor_mean
    )
  }
  
  rbindlist(result_list)
}

# 4. Compute for all variables and merge
all_stats <- list()
for (v in neighbor_vars) {
  stats_v <- compute_neighbor_stats_sparse(v)
  setnames(stats_v, c("year", paste0(v, "_max"), paste0(v, "_min"), paste0(v, "_mean")))
  if (length(all_stats) == 0) {
    all_stats[[1]] <- stats_v
  } else {
    all_stats[[length(all_stats) + 1]] <- stats_v[, -1]  # drop year column for merge
  }
}

neighbor_features <- do.call(cbind, all_stats)
neighbor_features[, id := id_order]
final_data <- merge(cell_data, neighbor_features, by = c("id", "year"))

# Random Forest prediction (preserve trained model)
predictions <- predict(pretrained_rf_model, newdata = final_data)
```

**Why This Is Fast**  
- Adjacency built once as a sparse matrix.  
- Heavy computations (sum, mean) done via matrix multiplication.  
- Min/max computed using grouped aggregation in `data.table` rather than millions of small lists.  
- Memory use optimized by processing one year at a time.  
- Preserves numerical equivalence with original logic.  

**Expected Improvement**  
From 86+ hours to a few hours (or less) on a 16 GB laptop, due to vectorization and sparse operations.
END_RESPONSE>>>
