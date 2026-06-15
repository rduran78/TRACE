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
run_id: OpenAI_F07_T03_R005
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 5
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F07_OpenAI_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current approach is extremely slow because:  
- It uses `lapply` over ~6.46 million rows, creating millions of small vectors and repeatedly performing lookups in R lists and named vectors (high overhead).  
- The neighbor lookup is computed row-by-row for every cell-year, which is redundant because the topology is static across years.  
- Repeated string concatenation (`paste`) and named indexing inside the loop are costly.  
- Memory overhead from large lists and repeated allocations is significant for 16 GB RAM.  

---

**Optimization Strategy**  
1. **Precompute neighbor indices once**: Build a numeric matrix of neighbor indices for each cell (not per cell-year).  
2. **Vectorize across years**: Use matrix operations to compute neighbor stats for all years in blocks or via `data.table` joins.  
3. **Avoid string keys**: Replace `paste`-based lookups with integer indexing.  
4. **Use `data.table` for speed and memory efficiency**: It handles large datasets well and supports fast grouped operations.  
5. **Parallelize if possible**: Use `parallel::mclapply` or `future.apply` for multi-core processing.  

---

**Working R Code (Optimized)**  

```r
library(data.table)

# Assume cell_data is a data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2
setDT(cell_data)
setkey(cell_data, id, year)

# Precompute neighbor matrix (list of integer vectors)
# rook_neighbors_unique: list of neighbors per cell id in id_order
neighbor_list <- rook_neighbors_unique
names(neighbor_list) <- id_order

# Convert to a named integer index for fast lookup
id_to_idx <- setNames(seq_along(id_order), id_order)

# Function to compute neighbor stats for one variable
compute_neighbor_stats_fast <- function(var_name) {
  vals <- matrix(cell_data[[var_name]], nrow = length(id_order), byrow = FALSE,
                 ncol = length(unique(cell_data$year)))
  # vals[i, y] = value for cell i in year y
  # Build matrix: rows = cells, cols = years
  # Fill matrix
  years <- sort(unique(cell_data$year))
  vals[,] <- t(matrix(cell_data[[var_name]], ncol = length(years), byrow = TRUE))

  # Preallocate result matrices
  max_mat <- matrix(NA_real_, nrow = nrow(vals), ncol = ncol(vals))
  min_mat <- matrix(NA_real_, nrow = nrow(vals), ncol = ncol(vals))
  mean_mat <- matrix(NA_real_, nrow = nrow(vals), ncol = ncol(vals))

  # Compute stats per cell using neighbors
  for (i in seq_along(neighbor_list)) {
    nbrs <- neighbor_list[[i]]
    if (length(nbrs) > 0) {
      nbr_idx <- id_to_idx[as.character(nbrs)]
      sub_vals <- vals[nbr_idx, , drop = FALSE]
      max_mat[i, ] <- apply(sub_vals, 2, max, na.rm = TRUE)
      min_mat[i, ] <- apply(sub_vals, 2, min, na.rm = TRUE)
      mean_mat[i, ] <- apply(sub_vals, 2, mean, na.rm = TRUE)
    }
  }

  # Melt back to long format
  res <- data.table(
    id = rep(id_order, times = length(years)),
    year = rep(years, each = length(id_order)),
    paste0(var_name, "_nbr_max") = as.vector(max_mat),
    paste0(var_name, "_nbr_min") = as.vector(min_mat),
    paste0(var_name, "_nbr_mean") = as.vector(mean_mat)
  )
  res
}

# Apply for all variables and merge
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
results_list <- lapply(neighbor_source_vars, compute_neighbor_stats_fast)
neighbor_features <- Reduce(function(x, y) merge(x, y, by = c("id", "year")), results_list)

# Merge back into cell_data
cell_data <- merge(cell_data, neighbor_features, by = c("id", "year"), all.x = TRUE)
```

---

**Expected Performance Gains**  
- Eliminates per-row `lapply` and string operations â†’ reduces 86+ hours to a few hours or less.  
- Uses matrix-based computation and `data.table` merges for efficiency.  
- Preserves original rook-neighbor relationships and numerical estimands.  
- Keeps the trained Random Forest model intact.  

**Memory Note**: If 16 GB is tight, process variables sequentially (as above) or split years into chunks.  

**Summary**: Precompute neighbor indices, operate in matrix form, and avoid per-row loops. This approach is exact, efficient, and scalable for your dataset.
END_RESPONSE>>>
