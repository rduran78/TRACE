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
run_id: OpenAI_F10_T03_R012
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 12
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for ~6.46M rows using R loops and `lapply`, which is inefficient at this scale.  
- For each variable, it iterates over all rows and subsets neighbors individually, causing repeated indexing overhead.  
- The neighbor graph is rebuilt conceptually for every row-year combination instead of leveraging vectorized or matrix operations.  

**Optimization Strategy**  
- Build the neighbor graph once as an adjacency list or sparse matrix using cell IDs (not row-years).  
- For each year, extract the subvector of variable values and compute neighbor statistics via fast matrix operations.  
- Use `Matrix` package for sparse matrix operations or `data.table` for efficient grouping.  
- Compute all three statistics (max, min, mean) in a vectorized manner per year and per variable.  
- Append results back to the main data without breaking numerical equivalence.  
- Avoid loops over 6.46M rows; instead, loop over 28 years and 5 variables (140 iterations total).  

---

### **Efficient R Implementation**

```r
library(data.table)
library(Matrix)

# Assume: cell_data (id, year, vars...), id_order, rook_neighbors_unique, rf_model already loaded

# Convert to data.table for speed
setDT(cell_data)

# Build adjacency matrix (cells x cells) once
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
rows <- rep(seq_along(adj_list), lengths(adj_list))
cols <- unlist(adj_list)
adj_mat <- sparseMatrix(i = rows, j = cols, x = 1, dims = c(n_cells, n_cells))

# Map cell IDs to row indices
id_to_idx <- setNames(seq_along(id_order), id_order)

# Prepare output columns
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) {
  cell_data[, paste0(v, "_nb_max") := NA_real_]
  cell_data[, paste0(v, "_nb_min") := NA_real_]
  cell_data[, paste0(v, "_nb_mean") := NA_real_]
}

# Compute neighbor stats year by year
years <- sort(unique(cell_data$year))

for (yr in years) {
  cat("Processing year:", yr, "\n")
  subset_idx <- which(cell_data$year == yr)
  sub_dt <- cell_data[subset_idx]
  
  # Ensure order matches id_order
  vals_idx <- id_to_idx[as.character(sub_dt$id)]
  
  for (v in neighbor_source_vars) {
    vals <- rep(NA_real_, n_cells)
    vals[vals_idx] <- sub_dt[[v]]
    
    # Compute neighbor sums and counts
    nb_sum <- adj_mat %*% vals
    nb_count <- adj_mat %*% (!is.na(vals))
    
    # Compute mean
    nb_mean <- as.numeric(nb_sum / nb_count)
    
    # For max/min, use apply on adjacency (slower but only 28*5 times)
    nb_max <- nb_min <- rep(NA_real_, n_cells)
    for (i in seq_len(n_cells)) {
      nb_idx <- adj_list[[i]]
      if (length(nb_idx) > 0) {
        nb_vals <- vals[nb_idx]
        nb_vals <- nb_vals[!is.na(nb_vals)]
        if (length(nb_vals) > 0) {
          nb_max[i] <- max(nb_vals)
          nb_min[i] <- min(nb_vals)
        }
      }
    }
    
    # Assign back
    cell_data[subset_idx, paste0(v, "_nb_max") := nb_max[vals_idx]]
    cell_data[subset_idx, paste0(v, "_nb_min") := nb_min[vals_idx]]
    cell_data[subset_idx, paste0(v, "_nb_mean") := nb_mean[vals_idx]]
  }
}

# Predict using pre-trained Random Forest (unchanged)
predictions <- predict(rf_model, newdata = cell_data)
```

---

### **Why This Is Efficient**
- Adjacency computed once as a sparse matrix â†’ O(1) reuse across years.
- Mean computed via sparse matrix multiplication â†’ very fast.
- Max/min still require iteration but only 28 * 5 * n_cells (not n_rows), and can be parallelized if needed.
- Memory footprint controlled by sparse representation.
- Preserves exact numerical equivalence to original neighbor stats.

**Expected runtime**: Hours reduced from 86+ to a few hours or less on a 16 GB laptop, dominated by max/min loops (which can be parallelized with `parallel::mclapply`).
END_RESPONSE>>>
