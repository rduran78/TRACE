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
run_id: Anthropic_F08_T00_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation is catastrophically slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds per-row (cellÃ—year) neighbor indices.** It creates a list of ~6.46 million elements, each requiring string-pasting (`paste(id, year)`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. But the neighbor *relationships* between cells never change across years â€” only the *values* attached to cells change. This means the function does ~28Ã— more work than necessary.

2. **String-based key lookups are extremely expensive at scale.** The function creates a named vector `idx_lookup` with ~6.46M entries keyed by `"id_year"` strings. Named-vector lookup in R is O(n) per query (linear scan), not O(1). With ~1.37M neighbor edges Ã— 28 years Ã— 5 variables, this produces billions of character comparisons.

3. **`compute_neighbor_stats` iterates via `lapply` over ~6.46M rows**, calling `max`, `min`, `mean` individually per row. This is pure R-level looping with no vectorization.

4. **Memory pressure.** The 6.46M-element list of integer vectors in `neighbor_lookup` is itself a large, fragmented object that thrashes the garbage collector.

### The Key Insight

The neighbor graph is **static** (cell-to-cell topology is year-invariant). The variable values are **dynamic** (they change by year). The current code entangles these two by indexing into the flattened cellÃ—year data frame. The fix is to **separate topology from data**: build the neighbor lookup once over 344,208 cells, then for each year, slice the relevant column, and compute stats using vectorized/matrix operations.

---

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a list of length 344,208 where element `i` contains the integer indices of cell `i`'s neighbors. This is topology-only, year-free, and built once.

2. **Organize data so that each year's values can be accessed as a contiguous vector.** Sort data by `(id, year)` or create a cellÃ—year matrix for each variable. With 344,208 cells Ã— 28 years, a matrix is ~73 MB per variable (doubles) â€” very manageable.

3. **Vectorize the neighbor aggregation.** For each variable and each year, extract the column vector of length 344,208, then compute neighbor max/min/mean using a sparse-matrix multiply or a fast C-backed loop. The sparse adjacency matrix approach turns `neighbor_mean` into a single sparse matrixâ€“vector product per year per variable.

4. **Use a sparse adjacency matrix (from `Matrix` package)** for mean computation (just `A %*% x / row_degrees`), and row-wise operations for min/max. Alternatively, use `data.table` grouped operations.

5. **Result:** Instead of 6.46M Ã— 5 expensive R-level iterations, we do 28 years Ã— 5 variables = 140 vectorized operations over 344K cells. Expected runtime: **minutes, not days**.

---

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build STATIC cell-level neighbor structures (done ONCE)
# ==============================================================================

build_cell_neighbor_structures <- function(id_order, rook_neighbors) {

  # id_order: vector of 344,208 cell IDs in the order used by the nb object
  # rook_neighbors: spdep nb object (list of integer index vectors)
  
  n_cells <- length(id_order)
  
  # --- 1a. Cell-level neighbor list (for min/max) ---
  # rook_neighbors[[i]] already contains integer indices into id_order
  # We just need to clean it (spdep nb objects use 0L for no-neighbor cells)
  cell_neighbor_list <- lapply(rook_neighbors, function(nb_idx) {
    nb_idx[nb_idx > 0L]
  })
  
  # --- 1b. Sparse adjacency matrix (for mean) ---
  # Build COO triplets
  from_idx <- rep(seq_len(n_cells), lengths(cell_neighbor_list))
  to_idx   <- unlist(cell_neighbor_list, use.names = FALSE)
  
  adj_matrix <- sparseMatrix(
    i = from_idx,
    j = to_idx,
    x = 1.0,
    dims = c(n_cells, n_cells)
  )
  
  # Row degrees (number of neighbors per cell) for computing mean
  row_degrees <- diff(adj_matrix@p)  # for dgCMatrix; or use rowSums
  row_degrees <- as.numeric(rowSums(adj_matrix))
  
  list(
    cell_neighbor_list = cell_neighbor_list,
    adj_matrix         = adj_matrix,
    row_degrees        = row_degrees,
    id_order           = id_order,
    n_cells            = n_cells
  )
}

# ==============================================================================
# STEP 2: Compute neighbor stats per variable using STATIC topology + DYNAMIC values
# ==============================================================================

compute_neighbor_stats_fast <- function(cell_data_dt, var_name, cell_structs) {
  # cell_data_dt: data.table with columns id, year, <var_name>, sorted by (id, year)
  # cell_structs: output of build_cell_neighbor_structures
  
  adj        <- cell_structs$adj_matrix
  degrees    <- cell_structs$row_degrees
  nb_list    <- cell_structs$cell_neighbor_list
  id_order   <- cell_structs$id_order
  n_cells    <- cell_structs$n_cells
  
  # Create a cell-index mapping: cell ID -> position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add cell position index to data
  cell_data_dt[, cell_pos := id_to_pos[as.character(id)]]
  
  years <- sort(unique(cell_data_dt$year))
  n_years <- length(years)
  
  # Pre-allocate output columns
  max_col <- rep(NA_real_, nrow(cell_data_dt))
  min_col <- rep(NA_real_, nrow(cell_data_dt))
  mean_col <- rep(NA_real_, nrow(cell_data_dt))
  
  # Process each year independently (vectorized over cells within each year)
  for (yr in years) {
    # Row indices in cell_data_dt for this year
    yr_mask <- which(cell_data_dt$year == yr)
    
    # Build a full-length vector for this year: position -> value
    # (NA for any cell not present in this year's data)
    vals_full <- rep(NA_real_, n_cells)
    positions_this_year <- cell_data_dt$cell_pos[yr_mask]
    vals_full[positions_this_year] <- cell_data_dt[[var_name]][yr_mask]
    
    # --- Neighbor MEAN via sparse matrix-vector product ---
    # adj %*% vals_full gives sum of neighbor values (NAs become 0 in sparse mult)
    # We need to handle NAs properly
    
    not_na <- !is.na(vals_full)
    vals_zero <- vals_full
    vals_zero[is.na(vals_zero)] <- 0.0
    
    neighbor_sum   <- as.numeric(adj %*% vals_zero)
    neighbor_count <- as.numeric(adj %*% as.numeric(not_na))
    
    neighbor_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
    
    # --- Neighbor MAX and MIN via fast vectorized approach ---
    neighbor_max <- rep(NA_real_, n_cells)
    neighbor_min <- rep(NA_real_, n_cells)
    
    # Use vapply over cells (344K iterations â€” fast enough, ~1-2 sec per year)
    # Only iterate over cells that actually appear this year
    cells_to_compute <- positions_this_year
    
    max_min_results <- vapply(cells_to_compute, function(ci) {
      nb_idx <- nb_list[[ci]]
      if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_))
      nb_vals <- vals_full[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) return(c(NA_real_, NA_real_))
      c(max(nb_vals), min(nb_vals))
    }, numeric(2))
    # max_min_results is 2 x length(cells_to_compute)
    
    neighbor_max_yr <- max_min_results[1, ]
    neighbor_min_yr <- max_min_results[2, ]
    
    # Write results back to the correct rows in the output vectors
    mean_col[yr_mask] <- neighbor_mean[positions_this_year]
    max_col[yr_mask]  <- neighbor_max_yr
    min_col[yr_mask]  <- neighbor_min_yr
  }
  
  # Clean up temporary column
  cell_data_dt[, cell_pos := NULL]
  
  list(max = max_col, min = min_col, mean = mean_col)
}

# ==============================================================================
# STEP 3: Full pipeline â€” drop-in replacement for the outer loop
# ==============================================================================

run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {
  
  cat("Building static cell-level neighbor structures...\n")
  cell_structs <- build_cell_neighbor_structures(id_order, rook_neighbors_unique)
  cat(sprintf("  %d cells, adjacency matrix: %d x %d with %d nonzeros\n",
              cell_structs$n_cells,
              nrow(cell_structs$adj_matrix),
              ncol(cell_structs$adj_matrix),
              nnzero(cell_structs$adj_matrix)))
  
  # Convert to data.table for speed (if not already)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  
  # Ensure sorted by (id, year) for consistent cell_pos mapping
  setkey(cell_data, id, year)
  
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for '%s'...\n", var_name))
    t0 <- proc.time()
    
    stats <- compute_neighbor_stats_fast(cell_data, var_name, cell_structs)
    
    # Add columns with same naming convention as original code
    max_name  <- paste0("neighbor_max_", var_name)
    min_name  <- paste0("neighbor_min_", var_name)
    mean_name <- paste0("neighbor_mean_", var_name)
    
    set(cell_data, j = max_name,  value = stats$max)
    set(cell_data, j = min_name,  value = stats$min)
    set(cell_data, j = mean_name, value = stats$mean)
    
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("  Done in %.1f seconds.\n", elapsed))
  }
  
  cat("All neighbor features computed.\n")
  return(cell_data)
}

# ==============================================================================
# USAGE (replaces the original outer loop)
# ==============================================================================

# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed with the pre-trained Random Forest prediction (unchanged):
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup construction** | 6.46M string-key entries, O(n) named-vector lookups | 344K-element integer list + sparse matrix, built once |
| **Iterations for stats** | 6.46M Ã— 5 = 32.3M R-level `lapply` calls | 28 years Ã— 5 vars = 140 vectorized passes |
| **Mean computation** | Per-row R loop with subsetting | Single sparse matrixâ€“vector multiply per year (~0.1 sec) |
| **Max/Min computation** | Per-row R loop over 6.46M rows | Per-row R loop over 344K cells Ã— 28 years (28Ã— fewer, no string ops) |
| **Memory for lookup** | ~6.46M-element ragged list + 6.46M-entry named character vector | ~344K-element list + one sparse matrix (~22 MB) |
| **Estimated total time** | ~86+ hours | **~10â€“20 minutes** |
| **Numerical output** | â€” | Identical (same max, min, mean over same neighbor values) |
| **Trained RF model** | â€” | Untouched; receives identical feature columns |

### Why the numerical estimand is preserved

The optimized code computes exactly the same quantities: for each cell-year row, it finds the same set of rook neighbors (same `nb` object, same cell IDs), retrieves the same variable values for the same year, and applies the same `max`, `min`, and `mean` functions with the same NA-handling logic. The sparse-matrix mean uses `sum(neighbor_values) / count(non-NA neighbors)`, which is algebraically identical to `mean(neighbor_values[!is.na(neighbor_values)])`. The output column names are identical, so the pre-trained Random Forest model receives the same feature matrix and requires no retraining.
END_RESPONSE>>>
