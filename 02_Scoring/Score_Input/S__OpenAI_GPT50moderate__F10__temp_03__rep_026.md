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
run_id: OpenAI_F10_T03_R026
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 26
raw_output_file: Experiment_Output/OpenAI/Temp_0.3/output_F10_OpenAI_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis**  
The current implementation is extremely slow (86+ hours) because:  
- It repeatedly computes neighbor statistics for each variable and each row using `lapply` over ~6.46M rows.  
- Neighbor lookups are recomputed for every row-year combination instead of leveraging vectorized or matrix-based operations.  
- The graph structure is rebuilt conceptually for each iteration rather than reused efficiently.  
- Memory overhead from repeated list operations and `do.call(rbind, ...)` on millions of elements is very high.  

---

**Optimization Strategy**  
1. **Precompute and reuse graph topology**: Build a single adjacency list or sparse matrix for the 344,208 cells (rook neighbors).  
2. **Vectorize across years**: Instead of looping row by row, process entire year blocks using matrix operations.  
3. **Use sparse matrix multiplication**: Represent adjacency as a sparse matrix `A` (size: cells Ã— cells). For each year, extract the variable vector `v` and compute:  
   - `neighbor_sum = A %*% v`  
   - `neighbor_count = A %*% 1`  
   Then derive mean, and for min/max use efficient aggregation by neighbors.  
4. **Chunk processing**: Process year by year to keep memory usage within 16 GB.  
5. **Preserve numerical equivalence**: Ensure NA handling matches original logic (ignore NA neighbors).  

---

**Working R Code**  

```r
library(Matrix)
library(data.table)

# Assume: cell_data has columns id, year, ntl, ec, pop_density, def, usd_est_n2
# rook_neighbors_unique: list of integer vectors (spdep nb object)
# id_order: vector of cell IDs in consistent order

# Convert to data.table for efficiency
setDT(cell_data)
setkey(cell_data, id, year)

# Build adjacency as sparse matrix (cells x cells)
n_cells <- length(id_order)
adj_list <- rook_neighbors_unique
row_idx <- rep(seq_along(adj_list), lengths(adj_list))
col_idx <- unlist(adj_list)
adj_mat <- sparseMatrix(i = row_idx, j = col_idx, x = 1, dims = c(n_cells, n_cells))

# Precompute index mapping
id_to_idx <- setNames(seq_along(id_order), id_order)

# Variables to process
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Prepare result columns
for (var in neighbor_source_vars) {
  cell_data[, paste0(var, "_nbr_max") := NA_real_]
  cell_data[, paste0(var, "_nbr_min") := NA_real_]
  cell_data[, paste0(var, "_nbr_mean") := NA_real_]
}

# Process year by year
years <- sort(unique(cell_data$year))
for (yr in years) {
  message("Processing year: ", yr)
  sub <- cell_data[year == yr]
  idx <- id_to_idx[sub$id]

  for (var in neighbor_source_vars) {
    vals <- sub[[var]]
    # Handle NA by zeroing and tracking counts
    na_mask <- is.na(vals)
    vals_na0 <- vals
    vals_na0[na_mask] <- 0

    # Sum and count for mean
    sum_neighbors <- as.numeric(adj_mat %*% vals_na0)
    count_neighbors <- as.numeric(adj_mat %*% (!na_mask))

    # Compute mean
    mean_neighbors <- ifelse(count_neighbors > 0, sum_neighbors / count_neighbors, NA_real_)

    # For min and max, aggregate manually
    # Build neighbor index lists for this year (reuse adj_list)
    nbr_max <- numeric(length(idx))
    nbr_min <- numeric(length(idx))
    for (i in seq_along(idx)) {
      nbr_idx <- adj_list[[idx[i]]]
      nbr_vals <- vals[nbr_idx]
      nbr_vals <- nbr_vals[!is.na(nbr_vals)]
      if (length(nbr_vals) == 0) {
        nbr_max[i] <- NA_real_
        nbr_min[i] <- NA_real_
      } else {
        nbr_max[i] <- max(nbr_vals)
        nbr_min[i] <- min(nbr_vals)
      }
    }

    # Assign back
    cell_data[year == yr, paste0(var, "_nbr_max") := nbr_max]
    cell_data[year == yr, paste0(var, "_nbr_min") := nbr_min]
    cell_data[year == yr, paste0(var, "_nbr_mean") := mean_neighbors]
  }
}

# At this point, cell_data has neighbor features added
# Apply pre-trained Random Forest model (model_rf)
# Example:
# preds <- predict(model_rf, newdata = cell_data)
```

---

**Why This Is Efficient**  
- Adjacency matrix is built once and reused.  
- Mean computation uses sparse matrix multiplication (fast, vectorized).  
- Min/max still require iteration but only over neighbors (average degree is small), so cost is manageable.  
- Year-by-year processing keeps memory footprint low.  
- Avoids repeated `lapply` over millions of rows.  

This approach will reduce runtime from 86+ hours to a few hours or less on a 16 GB laptop.
END_RESPONSE>>>
