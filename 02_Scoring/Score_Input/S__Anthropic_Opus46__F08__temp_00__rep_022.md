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
run_id: Anthropic_F08_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats all of this identically for each of the 28 years a cell appears. This means:

1. **Redundant topology computation**: The neighbor graph has ~344K cells and ~1.37M directed edges. This is year-invariant. Yet `build_neighbor_lookup` expands it to ~6.46M list entries (344K Ã— 28), each containing the same neighbor indices re-derived via string-key lookups. This alone creates ~38 million string-paste-and-match operations.

2. **Redundant per-variable iteration**: `compute_neighbor_stats` then iterates over all ~6.46M list entries per variable (Ã—5 variables = ~32.3M list traversals), when it could operate on a year-sliced matrix.

3. **Memory bloat**: The 6.46M-element list of integer vectors consumes substantial RAM and causes GC pressure on a 16 GB laptop.

**Root cause summary**: The static neighbor topology is entangled with the dynamic year dimension, causing an O(cells Ã— years) expansion of what should be an O(cells) structure, multiplied again by O(variables).

## Optimization Strategy

**Separate the static topology from the dynamic variable values:**

1. **Build the neighbor lookup once over cells only** (~344K entries, not ~6.46M). Each entry maps a cell to its neighbor cells by positional index into `id_order`. This is year-invariant and built once.

2. **For each variable, extract a cells Ã— years matrix** where row *i* corresponds to `id_order[i]` and columns correspond to years. This is a simple reshape.

3. **Compute neighbor stats via vectorized matrix operations**: For each cell *i* with neighbor set *N(i)*, extract the sub-matrix of neighbor values (rows = *N(i)*, columns = all years), then compute column-wise (i.e., per-year) max, min, mean. This processes all 28 years simultaneously per cell.

4. **Reshape results back** to the long cell-year format and attach to `cell_data`.

This reduces the lookup from ~6.46M entries to ~344K, eliminates all string-key operations, and replaces millions of R-level `lapply` iterations with vectorized matrix column operations. Expected speedup: roughly 20â€“50Ã— (from 86+ hours to 2â€“4 hours or less).

**Numerical equivalence**: The same neighbor sets and the same max/min/mean aggregations are computed, just reorganized. The trained Random Forest model is untouched.

## Working R Code

```r
# =============================================================================
# STEP 1: Build a cell-only neighbor lookup (year-invariant, built ONCE)
# =============================================================================
build_cell_neighbor_lookup <- function(id_order, rook_neighbors) {

  # rook_neighbors is an nb object: list of length = length(id_order),

# each element is an integer vector of neighbor positions in id_order.
  # We simply return it as-is (already positional indices into id_order).
  # Remove any 0-entries (spdep convention for "no neighbors").
  lapply(rook_neighbors, function(nb_idx) {
    nb_idx <- nb_idx[nb_idx != 0L]
    as.integer(nb_idx)
  })
}

# =============================================================================
# STEP 2: Reshape a variable from long cell-year data to a cells x years matrix
# =============================================================================
build_variable_matrix <- function(cell_data, id_order, years, var_name) {
  # Create a matrix: rows = cells (in id_order order), cols = years
  n_cells <- length(id_order)
  n_years <- length(years)

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  colnames(mat) <- as.character(years)

  # Build a fast mapping from cell id to row index in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Map each row of cell_data to (cell_pos, year_col)
  cell_pos <- id_to_pos[as.character(cell_data$id)]
  year_col <- match(cell_data$year, years)

  # Fill the matrix
  valid <- !is.na(cell_pos) & !is.na(year_col)
  mat[cbind(cell_pos[valid], year_col[valid])] <- cell_data[[var_name]][valid]

  mat
}

# =============================================================================
# STEP 3: Compute neighbor max, min, mean for one variable across all years
# =============================================================================
compute_neighbor_stats_optimized <- function(var_matrix, cell_neighbor_lookup) {
  # var_matrix: n_cells x n_years matrix
  # cell_neighbor_lookup: list of length n_cells, each element = integer vector
  #                       of neighbor row indices into var_matrix
  #
  # Returns: list with three matrices (max, min, mean), each n_cells x n_years

  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)

  mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_lookup[[i]]
    if (length(nb) == 0L) next

    if (length(nb) == 1L) {
      # Single neighbor: row vector
      nb_vals <- var_matrix[nb, , drop = FALSE]  # 1 x n_years
      mat_max[i, ]  <- nb_vals[1L, ]
      mat_min[i, ]  <- nb_vals[1L, ]
      mat_mean[i, ] <- nb_vals[1L, ]
    } else {
      # Multiple neighbors: sub-matrix, compute column-wise stats
      nb_vals <- var_matrix[nb, , drop = FALSE]  # length(nb) x n_years

      # Use colMeans, and vectorized pmin/pmax via Reduce or matrixStats
      # For robustness with NAs, use na.rm = TRUE
      mat_max[i, ]  <- apply(nb_vals, 2L, max, na.rm = TRUE)
      mat_min[i, ]  <- apply(nb_vals, 2L, min, na.rm = TRUE)
      mat_mean[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
  }

  # Fix Inf/-Inf from max/min on all-NA columns (apply with na.rm on empty)
  mat_max[is.infinite(mat_max)] <- NA_real_
  mat_min[is.infinite(mat_min)] <- NA_real_
  mat_mean[is.nan(mat_mean)]    <- NA_real_

  list(max = mat_max, min = mat_min, mean = mat_mean)
}

# =============================================================================
# STEP 3b: Faster version using matrixStats (if available) â€” recommended
# =============================================================================
compute_neighbor_stats_fast <- function(var_matrix, cell_neighbor_lookup) {
  require(matrixStats)

  n_cells <- nrow(var_matrix)
  n_years <- ncol(var_matrix)

  mat_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb <- cell_neighbor_lookup[[i]]
    if (length(nb) == 0L) next

    if (length(nb) == 1L) {
      row_vals <- var_matrix[nb, ]
      mat_max[i, ]  <- row_vals
      mat_min[i, ]  <- row_vals
      mat_mean[i, ] <- row_vals
    } else {
      nb_vals <- var_matrix[nb, , drop = FALSE]
      mat_max[i, ]  <- colMaxs(nb_vals, na.rm = TRUE)
      mat_min[i, ]  <- colMins(nb_vals, na.rm = TRUE)
      mat_mean[i, ] <- colMeans2(nb_vals, na.rm = TRUE)
    }
  }

  mat_max[is.infinite(mat_max)] <- NA_real_
  mat_min[is.infinite(mat_min)] <- NA_real_
  mat_mean[is.nan(mat_mean)]    <- NA_real_

  list(max = mat_max, min = mat_min, mean = mat_mean)
}

# =============================================================================
# STEP 4: Unpack matrix results back into the long cell_data data.frame
# =============================================================================
unpack_matrix_to_long <- function(cell_data, stat_matrix, id_order, years,
                                  col_name) {
  # stat_matrix: n_cells x n_years, rows aligned to id_order, cols to years
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  cell_pos  <- id_to_pos[as.character(cell_data$id)]
  year_col  <- match(cell_data$year, years)
  valid     <- !is.na(cell_pos) & !is.na(year_col)

  vals <- rep(NA_real_, nrow(cell_data))
  vals[valid] <- stat_matrix[cbind(cell_pos[valid], year_col[valid])]

  cell_data[[col_name]] <- vals
  cell_data
}

# =============================================================================
# STEP 5: Full optimized pipeline (drop-in replacement for the outer loop)
# =============================================================================
run_optimized_neighbor_pipeline <- function(cell_data, id_order,
                                            rook_neighbors_unique) {

  years <- sort(unique(cell_data$year))
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  # --- Static topology: built ONCE ---
  message("Building cell-level neighbor lookup (year-invariant)...")
  cell_nb <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

  use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)
  compute_fn <- if (use_matrixStats) compute_neighbor_stats_fast else
                                     compute_neighbor_stats_optimized

  # --- Dynamic variables: per-variable matrix operations ---
  for (var_name in neighbor_source_vars) {
    message(sprintf("Processing neighbor stats for: %s", var_name))

    # Reshape long -> wide matrix (cells x years)
    var_mat <- build_variable_matrix(cell_data, id_order, years, var_name)

    # Compute neighbor max, min, mean across all years at once
    stats <- compute_fn(var_mat, cell_nb)

    # Unpack back to long format and attach to cell_data
    # Column naming convention preserved from original:
    #   neighbor_max_{var}, neighbor_min_{var}, neighbor_mean_{var}
    cell_data <- unpack_matrix_to_long(
      cell_data, stats$max, id_order, years,
      paste0("neighbor_max_", var_name)
    )
    cell_data <- unpack_matrix_to_long(
      cell_data, stats$min, id_order, years,
      paste0("neighbor_min_", var_name)
    )
    cell_data <- unpack_matrix_to_long(
      cell_data, stats$mean, id_order, years,
      paste0("neighbor_mean_", var_name)
    )

    # Free memory
    rm(var_mat, stats)
    gc()
  }

  message("Neighbor feature computation complete.")
  cell_data
}

# =============================================================================
# USAGE (replaces the original outer loop)
# =============================================================================
cell_data <- run_optimized_neighbor_pipeline(
  cell_data, id_order, rook_neighbors_unique
)

# Then proceed with the pre-trained Random Forest prediction as before:
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup entries | ~6.46M (cell Ã— year) | ~344K (cell only) |
| String key operations | ~38M paste + match | 0 |
| R-level `lapply` calls per variable | ~6.46M | ~344K |
| Years processed per cell visit | 1 | 28 (vectorized columns) |
| Estimated wall time | 86+ hours | ~2â€“4 hours |
| Peak RAM | High (6.46M-element list + strings) | Moderate (344K Ã— 28 matrix) |
| Numerical result | max/min/mean of neighbor values | Identical |
| RF model | Unchanged | Unchanged |

The key insight: **factor the computation into (static topology) Ã— (dynamic values)** instead of materializing the Cartesian product. The neighbor graph is traversed once per cell; the year dimension is handled by columnar matrix operations inside that single traversal.
END_RESPONSE>>>
