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
run_id: OpenAI_F10_T07_R014
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 14
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F10_OpenAI_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current pipeline is extremely slow (86+ hours) due to:  
- Repeatedly scanning the entire dataset for each row and each variable.  
- Heavy use of `lapply` and list operations rather than vectorized or matrix-based computations.  
- Rebuilding neighbor relationships in a row-wise manner instead of leveraging efficient graph structures.  

The main performance bottleneck: `compute_neighbor_stats` loops through 6.46 million rows Ã— 5 variables, repeatedly extracting subsets from large vectors.  

**Optimization Strategy**  
- Represent the panel as a sparse graph adjacency structure (using `Matrix` or `igraph`).  
- Build a single adjacency matrix for spatial cells (344,208 nodes).  
- For each year, filter rows and compute neighbor stats using **matrix multiplication with sparse matrices**, which is highly optimized in R.  
- Process variable blocks in memory-efficient chunks.  
- Preallocate output matrices and bind results once.  
- Keep the Random Forest model unchanged, preserving numerical equivalence of computed neighbor features.  

**Efficient Implementation in R**  

```r
library(Matrix)
library(data.table)

# Assume: cell_data (id, year, variables), id_order, rook_neighbors_unique (spdep nb object)

# Convert nb object to adjacency matrix (cells only, no time)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_len(n_cells), sapply(adj_list, length))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# Convert data to data.table for speed
setDT(cell_data)
setkey(cell_data, id, year)

# Parameters
years <- sort(unique(cell_data$year))
neighbor_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Preallocate result storage
for (var in neighbor_vars) {
  cell_data[, paste0(var, "_nb_max") := NA_real_]
  cell_data[, paste0(var, "_nb_min") := NA_real_]
  cell_data[, paste0(var, "_nb_mean") := NA_real_]
}

# Process year by year
for (yr in years) {
  # Subset current year
  year_data <- cell_data[year == yr]
  
  # Map rows to adjacency matrix order
  vals_by_id <- year_data[match(id_order, id), ..neighbor_vars]
  
  # Convert to dense matrix for fast row ops
  vals_matrix <- as.matrix(vals_by_id)
  
  # For each variable, compute neighbor stats
  for (j in seq_along(neighbor_vars)) {
    vcol <- vals_matrix[, j]
    
    # Multiply adjacency by values to get sum of neighbors (for mean)
    sum_vals <- adj_mat %*% vcol
    count_vals <- adj_mat %*% !is.na(vcol)
    
    # Compute element-wise
    nb_mean <- as.numeric(sum_vals) / as.numeric(count_vals)
    
    # For max and min, use aggregation on adjacency indices
    # Build index-based aggregation
    # Extract adjacency list once to avoid overhead
    nlist <- adj_list
    nb_max <- nb_min <- numeric(n_cells)
    for (i in seq_len(n_cells)) {
      neigh_idx <- nlist[[i]]
      if (length(neigh_idx) == 0) {
        nb_max[i] <- nb_min[i] <- NA
      } else {
        neigh_vals <- vcol[neigh_idx]
        neigh_vals <- neigh_vals[!is.na(neigh_vals)]
        if (length(neigh_vals) == 0) {
          nb_max[i] <- nb_min[i] <- NA
        } else {
          nb_max[i] <- max(neigh_vals)
          nb_min[i] <- min(neigh_vals)
        }
      }
    }
    
    # Assign back
    cell_data[year == yr, paste0(neighbor_vars[j], "_nb_max") := nb_max[match(id, id_order)]]
    cell_data[year == yr, paste0(neighbor_vars[j], "_nb_min") := nb_min[match(id, id_order)]]
    cell_data[year == yr, paste0(neighbor_vars[j], "_nb_mean") := nb_mean[match(id, id_order)]]
  }
}

# At this point, cell_data has all neighbor features computed.
# Apply pre-trained Random Forest model without retraining:
preds <- predict(pretrained_rf_model, newdata = cell_data)
```

**Key Improvements**  
- Build adjacency once, reuse across years.  
- Use sparse matrix multiplication for neighbor sums (fast for mean).  
- Loop only over 28 years instead of 6.46M rows per variable.  
- Vectorized assignments via `data.table`.  
- Preserves original estimand and Random Forest integrity.  

**Expected Speed Gain**  
From 86+ hours to a few hours or less (mostly limited by max/min loops, which can be further parallelized with `parallel::mclapply`).  

**Further Optimization**  
- Replace max/min loop with `pmax`/`pmin` on chunked adjacency slices or use `Rcpp` for vectorized C++ aggregation.  
- Parallelize per-year or per-variable computations.  
- Memory footprint remains manageable since adjacency is stored once as sparse structure.
END_RESPONSE>>>
