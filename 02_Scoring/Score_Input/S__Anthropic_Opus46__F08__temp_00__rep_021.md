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
run_id: Anthropic_F08_T00_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F08_Anthropic_temp_00_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor topology is **identical across all 28 years**. This means:

1. **Redundant lookup construction**: The neighbor graph is spatial and static â€” cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds string-keyed lookups for every cell-year combination, creating ~6.46M entries instead of ~344K.

2. **String-key hashing is expensive at scale**: `paste(id, year, sep="_")` and named-vector lookups over 6.46M keys are O(n) in memory and slow in R.

3. **`compute_neighbor_stats` iterates over 6.46M list entries**: Each call to `lapply` over the full row-level lookup, repeated for 5 variables, means ~32.3M list-element iterations total.

4. **The fix is conceptually simple**: Build the neighbor lookup **once over the 344K cells** (not cell-years). Then, for each year, subset the relevant rows, pull neighbor values using the cell-level index, and compute stats. This reduces the lookup from 6.46M entries to 344K entries and makes the per-year computation a fast matrix operation.

## Optimization Strategy

1. **Build a cell-level neighbor index once** (`cell_neighbor_idx`): A list of length 344,208 where each element contains the integer positions of that cell's neighbors within the ordered cell vector. This is year-independent.

2. **Reshape each variable into a matrix**: rows = cells (in `id_order` order), columns = years. This allows vectorized column-slice access.

3. **Compute neighbor stats per year via vectorized operations**: For each year (column), use the static neighbor index to gather neighbor values and compute max/min/mean. This can be done with `vapply` over cells within each year â€” or even more efficiently with a sparse-matrix approach.

4. **Flatten results back into the original data.frame column order** to preserve downstream compatibility with the pre-trained Random Forest model.

**Expected speedup**: The lookup shrinks by 28Ã—. The per-variable computation becomes a loop over 28 years Ã— 344K cells with simple integer indexing (no string hashing). Estimated runtime drops from 86+ hours to minutes.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Exploits the fact that neighbor topology is static across years.
# =============================================================================

#' Build a CELL-level neighbor lookup (year-independent).
#' Returns a list of length n_cells. Each element is an integer vector of
#' neighbor positions in id_order.
#'
#' @param id_order   Integer vector of cell IDs in the order matching the nb object.
#' @param neighbors  An spdep::nb object (list of integer index vectors).
#' @return A named list keyed by cell ID (as character), values are integer
#'         vectors of positions in id_order.
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  n <- length(id_order)
  stopifnot(length(neighbors) == n)
  # neighbors[[i]] already contains integer indices into id_order

  # We just need to ensure they are clean integer vectors.
  lookup <- vector("list", n)
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    # spdep::nb uses 0L to denote "no neighbors"; filter those out
    nb_i <- nb_i[nb_i > 0L]
    lookup[[i]] <- as.integer(nb_i)
  }
  names(lookup) <- as.character(id_order)
  lookup
}

#' Compute neighbor max, min, mean for one variable across all cell-years.
#'
#' @param cell_data           data.frame with columns: id, year, and the target variable.
#' @param var_name            Character: name of the variable column.
#' @param cell_neighbor_lookup List from build_cell_neighbor_lookup().
#' @param id_order            Integer vector of cell IDs in canonical order.
#' @param years               Sorted integer vector of unique years.
#' @return A data.frame with three columns: <var>_neighbor_max, _min, _mean,
#'         in the same row order as cell_data.
compute_neighbor_stats_optimized <- function(cell_data,
                                              var_name,
                                              cell_neighbor_lookup,
                                              id_order,
                                              years) {
  n_cells <- length(id_order)
  n_years <- length(years)
  n_rows  <- nrow(cell_data)

  # --- Step 1: Build a cell-position lookup for fast mapping ----------------

  # Map cell id -> position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # --- Step 2: Reshape variable into a matrix [n_cells x n_years] -----------
  # We need to map each row of cell_data to (cell_pos, year_col).
  cell_pos_vec <- id_to_pos[as.character(cell_data$id)]
  year_to_col  <- setNames(seq_along(years), as.character(years))
  year_col_vec <- year_to_col[as.character(cell_data$year)]

  val_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  vals <- cell_data[[var_name]]
  # Fill the matrix
  idx_linear <- (year_col_vec - 1L) * n_cells + cell_pos_vec
  val_matrix[idx_linear] <- vals

  # --- Step 3: Compute neighbor stats per cell (vectorized over years) ------
  # Result matrices: [n_cells x n_years]
  max_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  min_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mean_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  for (i in seq_len(n_cells)) {
    nb_idx <- cell_neighbor_lookup[[i]]
    if (length(nb_idx) == 0L) next
    # nb_vals is a matrix: [length(nb_idx) x n_years]
    # Each row is one neighbor, each column is one year.
    nb_vals <- val_matrix[nb_idx, , drop = FALSE]
    # Compute column-wise stats (across neighbors, for each year)
    # Using colMeans / apply for max/min â€” but we can be smarter:
    if (length(nb_idx) == 1L) {
      # Single neighbor: max = min = mean = that value
      max_matrix[i, ]  <- nb_vals[1L, ]
      min_matrix[i, ]  <- nb_vals[1L, ]
      mean_matrix[i, ] <- nb_vals[1L, ]
    } else {
      # suppressWarnings to handle all-NA columns gracefully
      max_matrix[i, ]  <- suppressWarnings(apply(nb_vals, 2L, max,  na.rm = TRUE))
      min_matrix[i, ]  <- suppressWarnings(apply(nb_vals, 2L, min,  na.rm = TRUE))
      mean_matrix[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
  }

  # Fix Inf/-Inf from max/min on all-NA slices
  max_matrix[is.infinite(max_matrix)]  <- NA_real_
  min_matrix[is.infinite(min_matrix)]  <- NA_real_
  mean_matrix[is.nan(mean_matrix)]     <- NA_real_

  # --- Step 4: Flatten back to cell_data row order --------------------------
  out_max  <- max_matrix[idx_linear]
  out_min  <- min_matrix[idx_linear]
  out_mean <- mean_matrix[idx_linear]

  result <- data.frame(out_max, out_min, out_mean)
  colnames(result) <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  result
}


# =============================================================================
# MAIN PIPELINE (drop-in replacement for the outer loop)
# =============================================================================

# --- 1. Build the static cell-level neighbor lookup ONCE ---
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# --- 2. Identify canonical orderings ---
years <- sort(unique(cell_data$year))

# --- 3. Compute neighbor features for each source variable ---
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  stats_df <- compute_neighbor_stats_optimized(
    cell_data            = cell_data,
    var_name             = var_name,
    cell_neighbor_lookup = cell_neighbor_lookup,
    id_order             = id_order,
    years                = years
  )
  # Attach the three new columns to cell_data
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats_df[[1L]]
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats_df[[2L]]
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats_df[[3L]]
}

# --- 4. Predict with the pre-trained Random Forest (UNCHANGED) ---
# The model object and predict() call remain exactly as before.
# cell_data now has the same columns with the same numerical values.
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

## Why This Preserves the Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `cell_neighbor_lookup` uses the same `rook_neighbors_unique` nb object and `id_order`. The integer indices in the nb object directly index into `id_order`, so the neighbor sets are identical. |
| **Same aggregation functions** | `max`, `min`, `mean` with `na.rm = TRUE` are applied to the same neighbor value vectors. The only difference is vectorization across years via matrix columns. |
| **Same NA handling** | All-NA neighbor sets produce `NA` (via `Inf`â†’`NA` correction and `NaN`â†’`NA`), matching the original `return(c(NA, NA, NA))` logic. |
| **Same column names** | Output columns are named `<var>_neighbor_max`, `<var>_neighbor_min`, `<var>_neighbor_mean` â€” matching the original `compute_and_add_neighbor_features` output. |
| **RF model untouched** | No retraining. The model object is used as-is for `predict()`. |

## Performance Summary

| Metric | Original | Optimized |
|---|---|---|
| Neighbor lookup entries | ~6.46M (cell-year) | ~344K (cell only) |
| String hashing operations | ~6.46M Ã— 2 (build + query) | 0 at query time |
| Core loop iterations | ~6.46M per variable | ~344K per variable (matrix ops over 28 years) |
| Estimated wall time | 86+ hours | ~10â€“30 minutes |
| Peak RAM | Comparable | ~344K Ã— 28 Ã— 8 bytes â‰ˆ 77 MB per variable matrix (fits easily in 16 GB) |
END_RESPONSE>>>
