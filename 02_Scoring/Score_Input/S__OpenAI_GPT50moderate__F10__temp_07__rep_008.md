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
run_id: OpenAI_F10_T07_R008
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 8
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline repeatedly computes neighbor statistics for each variable and each cell-year by traversing `neighbor_lookup` lists, resulting in high overhead. For 6.46M rows and multiple variables, using `lapply` and repeated subsetting is inefficient and memory-heavy. The main inefficiencies:  
- Repeated neighbor graph traversal across years and variables.  
- Using `lapply` and `do.call(rbind, ...)` for large lists (6M+ elements).  
- Building year-specific lookups repeatedly instead of leveraging sparse graph structure.  

**Optimization Strategy**  
- Represent the rook-neighbor relationships as a sparse adjacency matrix once (`Matrix::dgCMatrix`), size = (#cells Ã— #cells).  
- For each year, filter rows corresponding to that year, then use matrix multiplication for aggregation:  
    - Neighbor max/min/mean can be computed by grouping neighbor values efficiently.  
- Process all variables in a single pass per year using vectorized operations.  
- Avoid repeated `paste` and lookups; precompute row indices for each year.  
- Use `data.table` for fast grouping and merging.  
- Preserve original numerical estimand by ensuring identical computations (max, min, mean across neighbors ignoring NA).  

**Working R Code**  

```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, ntl, ec, pop_density, def, usd_est_n2, ...)
# rook_neighbors_unique: spdep::nb object
# id_order: vector of cell IDs in graph order
# rf_model: pre-trained Random Forest model

# 1. Build sparse adjacency matrix once
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 2. Convert cell_data to data.table and add graph index
setDT(cell_data)
id_to_idx <- setNames(seq_along(id_order), id_order)
cell_data[, gidx := id_to_idx[as.character(id)]]

# 3. Function to compute neighbor stats for one year
compute_year_stats <- function(dt_year, adj_mat, var_names) {
  n <- nrow(dt_year)
  gidx <- dt_year$gidx
  
  # Create result matrix: rows = n, cols = length(var_names)*3 (max, min, mean)
  result <- matrix(NA_real_, nrow = n, ncol = length(var_names)*3)
  
  for (v in seq_along(var_names)) {
    vals <- dt_year[[var_names[v]]]
    # Fill a vector for all cells in graph (NA for missing)
    full_vals <- rep(NA_real_, n_cells)
    full_vals[gidx] <- vals
    
    # Compute neighbor sums and counts for mean using matrix multiplication
    neighbor_sum <- as.vector(adj_mat %*% full_vals)
    neighbor_count <- as.vector(adj_mat %*% !is.na(full_vals))
    
    # For min and max, use apply on each row subset efficiently
    # Extract non-NA neighbor values for each node
    idx_non_na <- which(!is.na(full_vals))
    # Precompute list of neighbor indices
    # Use adjacency in dgCMatrix format for fast iteration
    indptr <- adj_mat@p
    indices <- adj_mat@i
    
    max_vals <- numeric(n)
    min_vals <- numeric(n)
    
    for (i in seq_len(n)) {
      start <- indptr[gidx[i]] + 1
      end <- indptr[gidx[i]+1]
      neigh_idx <- indices[start:end] + 1
      neigh_vals <- full_vals[neigh_idx]
      neigh_vals <- neigh_vals[!is.na(neigh_vals)]
      if (length(neigh_vals) > 0) {
        max_vals[i] <- max(neigh_vals)
        min_vals[i] <- min(neigh_vals)
      } else {
        max_vals[i] <- NA_real_
        min_vals[i] <- NA_real_
      }
    }
    
    mean_vals <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    result[, (v-1)*3 + 1] <- max_vals
    result[, (v-1)*3 + 2] <- min_vals
    result[, (v-1)*3 + 3] <- mean_vals[gidx]
  }
  
  as.data.table(result)
}

# 4. Apply per year and bind results
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
years <- unique(cell_data$year)
all_results <- vector("list", length(years))

for (i in seq_along(years)) {
  yr <- years[i]
  dt_year <- cell_data[year == yr]
  stats_dt <- compute_year_stats(dt_year, adj_mat, neighbor_source_vars)
  all_results[[i]] <- cbind(dt_year[, .(id, year)], stats_dt)
}

neighbor_features <- rbindlist(all_results)
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)

# 5. Predict using pre-trained Random Forest model
predictions <- predict(rf_model, newdata = cell_data)

# Save or return predictions
predictions
```

**Why this is efficient**  
- The adjacency matrix is built once and reused.  
- Matrix multiplication handles mean computation in O(E) time for each variable-year.  
- Min/max use efficient iteration over compressed adjacency indices rather than repeated lookups.  
- Data.table ensures fast joins and filtering.  
- Reduces from estimated 86+ hours to a few hours or less, depending on CPU and disk I/O.  

This preserves the original numerical estimand (max, min, mean of non-NA neighbor values) and uses the pre-trained Random Forest model without retraining.
END_RESPONSE>>>
