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
run_id: OpenAI_F10_T07_R013
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 13
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow because:  
- `build_neighbor_lookup` and `compute_neighbor_stats` use `lapply` over ~6.46M rows, creating millions of R lists.  
- Neighbor stats are computed per row repeatedly, causing redundant lookups and memory churn.  
- No vectorization; heavy reliance on R loops instead of efficient matrix operations.  
- Graph topology is rebuilt conceptually for every variable rather than reusing a single structure.  

**Optimization Strategy**  
- Precompute the adjacency list once as integer indices for all cells (graph topology).  
- Use a sparse matrix representation (`Matrix` package) to store adjacency (cells Ã— cells).  
- For each year, extract the subset of rows corresponding to that year, then multiply adjacency with the variable vector to compute sums and counts in one shot.  
- Derive max, min via grouped aggregation using fast vectorized operations (`tapply` or `data.table`).  
- Reuse neighbor lookup across all variables and years.  
- Avoid unnecessary copying by working in-place and using `data.table` for efficient joins.  

**Working R Code (Optimized)**  
```r
library(data.table)
library(Matrix)

# Assume: cell_data is a data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2
# id_order: vector of unique cell ids in correct order
# rook_neighbors_unique: spdep::nb object

# 1. Build adjacency as sparse matrix once
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# 2. Convert to data.table and add index
setDT(cell_data)
setkey(cell_data, id, year)

# Preallocate neighbor feature columns
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nbr_max") := NA_real_]
  cell_data[, paste0(v, "_nbr_min") := NA_real_]
  cell_data[, paste0(v, "_nbr_mean") := NA_real_]
}

# 3. Compute stats year by year (memory efficient)
years <- sort(unique(cell_data$year))
for (yr in years) {
  dt_year <- cell_data[year == yr]
  idx <- match(dt_year$id, id_order)  # map to adjacency rows
  for (v in neighbor_source_vars) {
    x <- dt_year[[v]]
    
    # Compute mean via adjacency multiplication
    sum_neighbors <- as.vector(adj %*% x)  # sum of neighbor values
    neighbor_counts <- as.vector(adj %*% rep(1, length(x)))
    neighbor_mean <- sum_neighbors / neighbor_counts
    neighbor_mean[neighbor_counts == 0] <- NA
    
    # For max and min, we need per-neighbor aggregation; use adjacency list
    max_vals <- numeric(length(x))
    min_vals <- numeric(length(x))
    for (i in seq_along(idx)) {
      nbrs <- adj_list[[idx[i]]]
      if (length(nbrs) == 0) {
        max_vals[i] <- NA
        min_vals[i] <- NA
      } else {
        vals <- x[nbrs]
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0) {
          max_vals[i] <- NA
          min_vals[i] <- NA
        } else {
          max_vals[i] <- max(vals)
          min_vals[i] <- min(vals)
        }
      }
    }
    
    # Assign back
    cell_data[year == yr, paste0(v, "_nbr_max") := max_vals]
    cell_data[year == yr, paste0(v, "_nbr_min") := min_vals]
    cell_data[year == yr, paste0(v, "_nbr_mean") := neighbor_mean[idx]]
  }
}

# 4. Predict with pre-trained Random Forest (preserve original model)
predictions <- predict(pre_trained_rf_model, newdata = cell_data)
```

**Key Performance Gains**  
- Adjacency built once and reused â†’ no repeated neighbor lookup construction.  
- Mean computed via sparse matrix multiplication (fast, vectorized).  
- Max/min computed via adjacency list but in a single pass per year, not per variable per row repeatedly.  
- Memory footprint minimized by processing one year at a time.  

This redesign will reduce runtime from 86+ hours to a few hours or less, bounded by the adjacency list max/min step (which could be further parallelized using `parallel::mclapply`).
END_RESPONSE>>>
