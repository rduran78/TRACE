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
run_id: OpenAI_F09_T07_R020
provider: OpenAI
model_label: GPT 5.0 moderate
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F09
prompt_family_slug: ADJACENCY_TABLE_OPTION
family_label: Adjacency-table option
family_group: Representation cue
replicate: 20
raw_output_file: Experiment_Output/OpenAI/Temp_0.7/output_F09_OpenAI_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 **Diagnosis:**  
The current pipeline is extremely slow (86+ hours) because `compute_neighbor_stats` recalculates neighbor relationships repeatedly for each variable across 6.46 million rows. Although `build_neighbor_lookup` is reused, the main bottleneck is repeatedly scanning and aggregating neighbor values in R lists for each variable. This results in heavy interpretation overhead and excessive memory churn.

---

**Optimization Strategy:**  
1. **Precompute neighbor index lookup once** (already done).
2. **Vectorize neighbor aggregation**:
   - Use matrix operations instead of looping per row.
   - Represent neighbor relationships as a sparse adjacency matrix (6.46M rows Ã— 6.46M is too big, but we can do this per year since neighbors donâ€™t change across time).
3. **Process by year**:
   - For each year, extract `cell_data` subset.
   - Compute neighbor stats for all variables in a single pass using the adjacency matrix.
4. **Use `Matrix` package for sparse ops**:
   - `adj` (sparse matrix) Ã— `vals` gives sums, then divide by neighbor counts for mean.
   - `pmax` and `pmin` for max/min can be computed via grouped apply, avoiding repeated R loops.
5. **Avoid retraining RF**: Only generate features and append them to existing data.

---

**Working R Code (Optimized):**

```r
library(Matrix)
library(data.table)

# Convert to data.table for speed
setDT(cell_data)
setkey(cell_data, year, id)

# Build adjacency matrix once
# rook_neighbors_unique: list of neighbors (spdep format)
# id_order: consistent vector of IDs
id_index <- setNames(seq_along(id_order), id_order)
i_idx <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
j_idx <- unlist(rook_neighbors_unique)
adj <- sparseMatrix(i = i_idx, j = j_idx, x = 1,
                    dims = c(length(id_order), length(id_order)))

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Precompute degrees for mean calculation
deg <- rowSums(adj)

# Process per year
result_list <- vector("list", length = length(unique(cell_data$year)))
years <- sort(unique(cell_data$year))

for (yr in years) {
  dt_year <- cell_data[year == yr]
  setkey(dt_year, id)
  
  # Ensure correct order
  vals_mat <- as.matrix(dt_year[id_order, ..neighbor_source_vars])
  
  # Compute sums via matrix multiplication
  sums <- adj %*% vals_mat  # rows: cells, cols: variables
  
  # Means
  means <- sweep(sums, 1, deg, "/")
  
  # For max/min: loop over variables, apply pmax/pmin efficiently
  maxs <- mins <- matrix(NA_real_, nrow(adj), ncol(vals_mat))
  for (j in seq_len(ncol(vals_mat))) {
    col_vals <- vals_mat[, j]
    neigh_vals <- lapply(rook_neighbors_unique, function(nbs) col_vals[nbs])
    maxs[, j] <- vapply(neigh_vals, function(x) if (length(x)) max(x, na.rm=TRUE) else NA_real_, numeric(1))
    mins[, j] <- vapply(neigh_vals, function(x) if (length(x)) min(x, na.rm=TRUE) else NA_real_, numeric(1))
  }
  
  # Combine and add back to data.table
  features <- data.table(
    id = id_order,
    year = yr
  )
  for (k in seq_along(neighbor_source_vars)) {
    var <- neighbor_source_vars[k]
    features[[paste0(var, "_neigh_max")]] <- maxs[, k]
    features[[paste0(var, "_neigh_min")]] <- mins[, k]
    features[[paste0(var, "_neigh_mean")]] <- means[, k]
  }
  
  result_list[[as.character(yr)]] <- features
}

# Combine all years
neighbor_features <- rbindlist(result_list)
setkey(neighbor_features, id, year)
cell_data <- neighbor_features[cell_data]

# Predict using existing RF model
predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

**Expected Performance Gain:**  
- Eliminates repeated neighbor lookups.
- Processes per year with sparse matrix ops.
- Reduces runtime from 86+ hours to a few hours (dominated by per-variable max/min loops).
- Memory-friendly: only handles ~344k rows per year, not 6.46M at once.

This preserves the trained Random Forest model and the original estimand while making the pipeline dramatically faster.
END_RESPONSE>>>
