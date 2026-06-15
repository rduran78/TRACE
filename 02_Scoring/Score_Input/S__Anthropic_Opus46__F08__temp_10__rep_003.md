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
run_id: Anthropic_F08_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup of neighbor row indices, producing ~6.46 million list entries. But the neighbor *topology* is **static** â€” it depends only on cell identity, not on year. The current code redundantly recomputes the same neighbor cell IDs 28 times (once per year for each cell) and does expensive string-key lookups (`paste` + named-vector indexing) across all 6.46M rows.

Specifically:

1. **`build_neighbor_lookup`** creates ~6.46M character keys (`id_year`) and performs named-vector lookups for each row. With ~6.46M entries in `idx_lookup`, each named lookup is O(n) or O(log n) depending on hashing, repeated ~6.46M Ã— avg_neighbors times. This alone can take many hours.

2. **`compute_neighbor_stats`** iterates over the 6.46M-element list, extracting values and computing `max`, `min`, `mean`. This is repeated for each of 5 variables â€” so ~32.3M list iterations total.

3. The fundamental waste: the **same neighbor cell relationships** are resolved into row indices 28 times (once per year), when the topology is year-invariant.

## Optimization Strategy

**Separate the static topology from the year-varying data.**

1. **Build a cell-level neighbor index once** â€” a list of length 344,208 mapping each cell's position to its neighbors' positions (in a cell-order vector). This is just `rook_neighbors_unique` itself (an `nb` object already does this).

2. **Reshape each variable into a matrix**: rows = cells (344,208), columns = years (28). Now cell `i`'s neighbor values in year `j` are simply `matrix[neighbors[[i]], j]`.

3. **Compute neighbor stats as matrix operations over the cell dimension only** â€” loop over 344,208 cells (not 6.46M cell-years), and for each cell, extract the neighbor sub-matrix (neighbors Ã— 28), then compute columnwise max/min/mean. This produces a (28)-length vector per cell per stat.

4. **Vectorize further** by recognizing that for each cell, the neighbor sub-matrix extraction and column-wise summary can be done very efficiently, or even fully vectorized using sparse-matrix multiplication (for `mean`) and row-wise grouped operations.

This reduces the work from ~6.46M list lookups to ~344K, and eliminates all string-key construction.

**Expected speedup**: From 86+ hours to roughly **minutes** (the dominant cost becomes ~344K list accesses on small sub-matrices of ~4 neighbors Ã— 28 years).

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the fact that neighbor topology is static across years.
# =============================================================================

#' Build a cell-year matrix from panel data for a single variable.
#' Returns a list with:
#'   - mat: a (n_cells x n_years) matrix
#'   - cell_ids: vector of unique cell IDs (row order)
#'   - years: vector of unique years (column order)
#'   - cell_id_to_row: named integer vector mapping cell ID -> row index in mat
build_variable_matrix <- function(data, var_name) {
  cell_ids <- sort(unique(data$id))
  years    <- sort(unique(data$year))

  n_cells <- length(cell_ids)
  n_years <- length(years)

  # Map cell id and year to matrix indices
  cell_id_to_row <- setNames(seq_along(cell_ids), as.character(cell_ids))
  year_to_col    <- setNames(seq_along(years), as.character(years))

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  row_idx <- cell_id_to_row[as.character(data$id)]
  col_idx <- year_to_col[as.character(data$year)]

  mat[cbind(row_idx, col_idx)] <- data[[var_name]]

  list(
    mat            = mat,
    cell_ids       = cell_ids,
    years          = years,
    cell_id_to_row = cell_id_to_row,
    year_to_col    = year_to_col
  )
}

#' Compute neighbor max, min, mean for one variable across all cells and years.
#' Uses the static nb object directly.
#'
#' @param var_mat      Matrix (n_cells x n_years) of variable values.
#' @param neighbors    An nb object (list of length n_cells), where neighbors[[i]]
#'                     contains integer indices of i's neighbors in the same
#'                     cell ordering used to build var_mat.
#'                     Note: spdep nb objects use 0L to denote "no neighbors".
#' @param id_order     The cell ID vector corresponding to the nb object's ordering.
#' @param cell_id_to_row  Named vector mapping cell ID -> row in var_mat.
#'
#' @return A list with three matrices (n_cells x n_years): nb_max, nb_min, nb_mean.
compute_neighbor_stats_matrix <- function(var_mat, neighbors, id_order,
                                          cell_id_to_row) {
  n_cells <- nrow(var_mat)
  n_years <- ncol(var_mat)

  nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # Build a mapping from nb-object position -> var_mat row.
  # id_order[k] is the cell ID at position k in the nb object.
  # cell_id_to_row maps cell ID -> var_mat row.
  # So nb_pos_to_mat_row[k] gives the var_mat row for nb position k.
  nb_pos_to_mat_row <- cell_id_to_row[as.character(id_order)]
  # nb_pos_to_mat_row is aligned: position k in nb -> row in var_mat

  for (i in seq_len(length(neighbors))) {
    nb_idx <- neighbors[[i]]

    # spdep nb objects use integer(0) or 0L for cells with no neighbors
    if (length(nb_idx) == 0 || (length(nb_idx) == 1 && nb_idx[1] == 0L)) next

    # Map nb positions to var_mat rows
    mat_rows <- nb_pos_to_mat_row[nb_idx]
    mat_rows <- mat_rows[!is.na(mat_rows)]
    if (length(mat_rows) == 0) next

    # The current cell's row in var_mat
    my_mat_row <- nb_pos_to_mat_row[i]
    if (is.na(my_mat_row)) next

    # Extract the sub-matrix of neighbor values: (n_neighbors x n_years)
    if (length(mat_rows) == 1) {
      # Single neighbor: treat as a 1-row matrix
      sub <- matrix(var_mat[mat_rows, ], nrow = 1, ncol = n_years)
    } else {
      sub <- var_mat[mat_rows, , drop = FALSE]
    }

    # Column-wise (i.e., per-year) stats
    # Use colMeans, and manual colMax/colMin to handle NAs
    for (j in seq_len(n_years)) {
      vals <- sub[, j]
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0) next
      nb_max[my_mat_row, j]  <- max(vals)
      nb_min[my_mat_row, j]  <- min(vals)
      nb_mean[my_mat_row, j] <- mean(vals)
    }
  }

  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

#' Flatten a (n_cells x n_years) matrix back into the panel-data row order.
#'
#' @param mat            Matrix (n_cells x n_years).
#' @param data           The panel data.frame (must have $id and $year).
#' @param cell_id_to_row Named vector mapping cell ID -> matrix row.
#' @param year_to_col    Named vector mapping year -> matrix column.
#' @return A numeric vector of length nrow(data), aligned to data's row order.
matrix_to_panel_vector <- function(mat, data, cell_id_to_row, year_to_col) {
  row_idx <- cell_id_to_row[as.character(data$id)]
  col_idx <- year_to_col[as.character(data$year)]
  mat[cbind(row_idx, col_idx)]
}

# =============================================================================
# MAIN PIPELINE (replaces the original outer loop)
# =============================================================================

# --- Inputs assumed to exist: ---
# cell_data              : data.frame with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# id_order               : integer vector of cell IDs in the nb object's order
# rook_neighbors_unique  : spdep nb object (list of neighbor index vectors)
# rf_model               : pre-trained Random Forest model (NOT retrained)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Step 1: Build the cell_id_to_row and year_to_col mappings once.
#         (We build them from the first variable; they are the same for all.)
ref <- build_variable_matrix(cell_data, neighbor_source_vars[1])
cell_id_to_row <- ref$cell_id_to_row
year_to_col    <- ref$year_to_col

cat("Cell-year matrix dimensions:", nrow(ref$mat), "cells x", ncol(ref$mat), "years\n")

# Step 2: For each source variable, build matrix, compute neighbor stats, write back.
for (var_name in neighbor_source_vars) {
  cat("Processing neighbor stats for:", var_name, "\n")
  t0 <- Sys.time()

  # Build (n_cells x n_years) matrix for this variable
  vm <- build_variable_matrix(cell_data, var_name)

  # Compute neighbor max, min, mean using the static topology
  stats <- compute_neighbor_stats_matrix(
    var_mat        = vm$mat,
    neighbors      = rook_neighbors_unique,
    id_order       = id_order,
    cell_id_to_row = cell_id_to_row
  )

  # Flatten back to panel order and attach to cell_data
  cell_data[[paste0("nb_max_", var_name)]] <-
    matrix_to_panel_vector(stats$nb_max, cell_data, cell_id_to_row, year_to_col)
  cell_data[[paste0("nb_min_", var_name)]] <-
    matrix_to_panel_vector(stats$nb_min, cell_data, cell_id_to_row, year_to_col)
  cell_data[[paste0("nb_mean_", var_name)]] <-
    matrix_to_panel_vector(stats$nb_mean, cell_data, cell_id_to_row, year_to_col)

  cat("  Done in", round(difftime(Sys.time(), t0, units = "mins"), 1), "minutes\n")
}

# Step 3: Prediction with the pre-trained Random Forest (unchanged).
# The model is NOT retrained. We simply call predict().
cell_data$predicted <- predict(rf_model, newdata = cell_data)

cat("Pipeline complete.\n")
```

## Further Optimization: Vectorized Inner Loop with `vapply`

The inner double loop (over cells Ã— years) can be further accelerated by replacing the per-year loop with vectorized column operations:

```r
# Drop-in replacement for the inner loop in compute_neighbor_stats_matrix.
# Instead of looping over years inside the cell loop, use colMaxs/colMins/colMeans.

compute_neighbor_stats_matrix_fast <- function(var_mat, neighbors, id_order,
                                                cell_id_to_row) {
  n_cells <- nrow(var_mat)
  n_years <- ncol(var_mat)

  nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  nb_pos_to_mat_row <- cell_id_to_row[as.character(id_order)]

  # Optional: use matrixStats for colMaxs/colMins if available
  use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)

  for (i in seq_along(neighbors)) {
    nb_idx <- neighbors[[i]]
    if (length(nb_idx) == 0 || (length(nb_idx) == 1 && nb_idx[1] == 0L)) next

    mat_rows <- nb_pos_to_mat_row[nb_idx]
    mat_rows <- mat_rows[!is.na(mat_rows)]
    if (length(mat_rows) == 0) next

    my_mat_row <- nb_pos_to_mat_row[i]
    if (is.na(my_mat_row)) next

    if (length(mat_rows) == 1) {
      # Single neighbor: stats are just that neighbor's values
      nb_max[my_mat_row, ]  <- var_mat[mat_rows, ]
      nb_min[my_mat_row, ]  <- var_mat[mat_rows, ]
      nb_mean[my_mat_row, ] <- var_mat[mat_rows, ]
    } else {
      sub <- var_mat[mat_rows, , drop = FALSE]
      if (use_matrixStats) {
        nb_max[my_mat_row, ]  <- matrixStats::colMaxs(sub, na.rm = TRUE)
        nb_min[my_mat_row, ]  <- matrixStats::colMins(sub, na.rm = TRUE)
        nb_mean[my_mat_row, ] <- colMeans(sub, na.rm = TRUE)
      } else {
        nb_max[my_mat_row, ]  <- apply(sub, 2, max, na.rm = TRUE)
        nb_min[my_mat_row, ]  <- apply(sub, 2, min, na.rm = TRUE)
        nb_mean[my_mat_row, ] <- colMeans(sub, na.rm = TRUE)
      }
    }
  }

  # Replace -Inf/Inf from max/min on all-NA columns back to NA
  nb_max[is.infinite(nb_max)] <- NA_real_
  nb_min[is.infinite(nb_min)] <- NA_real_

  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup granularity | Per cell-year (6.46M entries) | Per cell (344K entries) |
| String key construction | ~6.46M `paste()` calls + named lookups | Zero string keys at runtime |
| Inner loop iterations | 6.46M Ã— 5 vars = 32.3M | 344K Ã— 5 vars = 1.72M |
| Data access pattern | Scattered vector indexing | Contiguous matrix sub-rows |
| Estimated runtime | 86+ hours | ~10â€“30 minutes |
| RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical result | Original estimand | Identical (same max/min/mean logic) |
END_RESPONSE>>>
