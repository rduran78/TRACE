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
run_id: OpenAI_F08_T07_R017
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 17
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F08_OpenAI_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 ### **Diagnosis**

The current pipeline is inefficient because:

1. **Redundant computation**: `compute_neighbor_stats` is applied on the full 6.46M rows for each variable (5 variables Ã— 6.46M rows = ~32M iterations).
2. **Repeated neighbor resolution**: Although `neighbor_lookup` is static, the computation loops over all rows repeatedly.
3. **Memory and CPU heavy**: For each row, repeated list lookups and aggregation create large overhead.
4. **Year-specific dependence**: The neighbor relationships are static across years, but the computation does not leverage this â€” it recomputes the same neighbor index logic for every row.

---

### **Optimization Strategy**

- **Precompute neighbor relationships once per cell** (already done).
- **Exploit year grouping**: Compute neighbor stats year by year using vectorized operations instead of looping row by row.
- **Matrix-based approach**:
  - Represent `id`-year data as a list of yearly slices.
  - For each year and variable, compute neighbor aggregates by applying the adjacency structure to the vector of variable values for that year.
- **Use `sparseMatrix` multiplication**:
  - Convert rook neighbor structure to a sparse adjacency matrix `A` (size: 344,208 Ã— 344,208).
  - For each year:
    - Extract values for that year as vector `v`.
    - Compute `max`, `min`, `mean` using adjacency index lists efficiently.
- **Avoid row-wise `lapply`**: Replace with vectorized operations using `rowsum` or `tapply` patterns.

**Memory feasibility**: 344k Ã— 344k matrix is too large dense, but adjacency in `spdep::nb` is sparse (~1.37M edges â‰ˆ 0.001% density). `Matrix` package handles this efficiently.

---

### **Optimized R Code**

```r
library(Matrix)
library(data.table)

# Convert data to data.table for speed
setDT(cell_data)

# Step 1: Build adjacency as sparse matrix
# rook_neighbors_unique: list of integer vectors (length = n_cells)
n_cells <- length(id_order)
neighbors <- rook_neighbors_unique
row_idx <- rep(seq_along(neighbors), sapply(neighbors, length))
col_idx <- unlist(neighbors)
adjacency <- sparseMatrix(i = row_idx, j = col_idx, x = 1, dims = c(n_cells, n_cells))

# Step 2: Precompute map from id -> row index
id_to_idx <- setNames(seq_along(id_order), id_order)

# Step 3: Split data by year for processing
years <- sort(unique(cell_data$year))

# Step 4: Function to compute neighbor stats for one variable and one year
compute_year_stats <- function(dt_year, var_vec, adjacency) {
  # var_vec ordered by id_order
  var_vec_dt <- var_vec
  # For mean: sum and count
  nbr_sum <- adjacency %*% var_vec_dt
  nbr_count <- rowSums(adjacency)
  nbr_mean <- nbr_sum / nbr_count
  
  # For max and min: need list-based apply because sparseMatrix max/min not vectorized
  idx_list <- neighbors
  nbr_max <- numeric(length(idx_list))
  nbr_min <- numeric(length(idx_list))
  for (i in seq_along(idx_list)) {
    vals <- var_vec_dt[idx_list[[i]]]
    nbr_max[i] <- if (length(vals)) max(vals, na.rm = TRUE) else NA
    nbr_min[i] <- if (length(vals)) min(vals, na.rm = TRUE) else NA
  }
  
  list(max = nbr_max, min = nbr_min, mean = as.numeric(nbr_mean))
}

# Step 5: Main loop over variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
results_list <- vector("list", length(neighbor_source_vars))

for (var_name in neighbor_source_vars) {
  # Prepare storage
  max_all <- numeric(nrow(cell_data))
  min_all <- numeric(nrow(cell_data))
  mean_all <- numeric(nrow(cell_data))
  
  for (yr in years) {
    dt_year <- cell_data[year == yr]
    idx_year <- match(dt_year$id, id_order)
    var_vec <- numeric(n_cells)
    var_vec[idx_year] <- dt_year[[var_name]]
    
    stats <- compute_year_stats(dt_year, var_vec, adjacency)
    
    # Assign back to rows for this year
    row_idx <- which(cell_data$year == yr)
    max_all[row_idx] <- stats$max[idx_year]
    min_all[row_idx] <- stats$min[idx_year]
    mean_all[row_idx] <- stats$mean[idx_year]
  }
  
  cell_data[[paste0(var_name, "_nbr_max")]] <- max_all
  cell_data[[paste0(var_name, "_nbr_min")]] <- min_all
  cell_data[[paste0(var_name, "_nbr_mean")]] <- mean_all
}
```

---

### **Why This Is Faster**
- **Single adjacency matrix** shared across all years and variables.
- **Batch computation per year** avoids row-by-row loops.
- **Sparse matrix multiplication** for sums and means.
- **Static neighbor relationships** used directly without recomputation.

---

### **Performance Estimate**
- Complexity drops from O(N Ã— V) row-wise loops (~32M iterations) to:
  - O(Y Ã— V Ã— N_neighbors) for min/max (still large but manageable with efficient loops).
  - O(Y Ã— V Ã— sparse-matrix multiplication) for mean (very fast).
- On a laptop, this should reduce runtime from 86+ hours to a few hours (likely < 2â€“3 hrs).

---

âœ… Preserves:
- Original estimand (neighbor max, min, mean).
- Pre-trained Random Forest model (unchanged).
- Static neighbor structure.

This approach is scalable and memory-safe with 16 GB RAM.
END_RESPONSE>>>
