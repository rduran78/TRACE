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
run_id: Anthropic_F08_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F08_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation suffers from a critical inefficiency: **it rebuilds a neighbor lookup that maps every cell-year row to its neighbor cell-year rows, even though the neighbor topology is purely spatial and never changes across the 28 years.** Specifically:

1. **`build_neighbor_lookup` creates a list of 6.46 million entries** (one per cell-year row), each containing row indices into the full panel. This involves creating ~6.46M paste-key lookups, hashing them, and then for each row, pasting neighbor keys with the same year, and looking them up. This is O(N Ã— T Ã— avg_neighbors) string work.

2. **`compute_neighbor_stats` iterates over 6.46M entries** per variable. For 5 variables, that's ~32.3M list-element operations, each involving subsetting, NA removal, and computing max/min/mean.

3. **The fundamental waste**: The neighbor relationship is between **cells**, not between cell-years. Cell *i*'s neighbors are the same in 1992 as in 2019. Yet the code re-derives these relationships at the cell-year level, inflating the problem by a factor of 28.

### Quantified bottleneck
- `build_neighbor_lookup`: Allocates a 6.46M-element list, each requiring string construction and hash lookups. This alone can take hours.
- `compute_neighbor_stats` Ã— 5 variables: 32.3M `lapply` iterations in R (not vectorized).

## Optimization Strategy

**Separate the static topology from the year-varying data, then use vectorized/matrix operations:**

1. **Build the neighbor lookup once at the cell level** (344,208 entries, not 6.46M). This is a simple translation of the `spdep::nb` object into a list of integer cell-index vectorsâ€”essentially, it already is one.

2. **Reshape each variable into a matrix of dimension `n_cells Ã— n_years`**, where rows are cells (in `id_order` order) and columns are years. This allows direct column-wise (year-wise) vectorized neighbor aggregation.

3. **Compute neighbor stats per variable using vectorized row-gather operations on the matrix.** For each cell, gather its neighbor rows (same across all years), compute max/min/mean across those rows for all 28 year-columns simultaneously. This replaces 6.46M R-level iterations with 344,208 iterations, each doing vectorized column operations.

4. **Further acceleration**: Use a sparse-matrix multiply to compute neighbor means (and use row-wise sparse operations for max/min), or use `data.table` grouped operations. The approach below uses direct matrix indexing which is cache-friendly and avoids string operations entirely.

**Expected speedup**: ~28Ã— from eliminating year redundancy in topology, plus large constant-factor gains from vectorization. Estimated runtime drops from 86+ hours to ~10â€“30 minutes.

**Numerical equivalence**: The same neighbors are gathered, the same values are read, and the same max/min/mean are computed. The trained Random Forest model is untouched.

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 0: Prepare ordered cell and year vectors
# ============================================================
# id_order: integer vector of length n_cells (344,208), giving cell IDs
#           in the same order as rook_neighbors_unique (the nb object).
# rook_neighbors_unique: spdep nb object, list of length n_cells,
#           each element is an integer vector of neighbor *positions*
#           (indices into id_order). 0-neighbor cells have integer(0).
# cell_data: data.frame/data.table with columns id, year, and all predictors.

n_cells <- length(id_order)
years   <- sort(unique(cell_data$year))
n_years <- length(years)

# Convert cell_data to data.table for fast manipulation
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ============================================================
# STEP 1: Build a cell-level neighbor lookup (STATIC, done once)
# ============================================================
# rook_neighbors_unique is already exactly this: a list where element i
# contains the integer indices (into id_order) of cell i's neighbors.
# We just ensure it's a clean list of integer vectors.

neighbor_cell_idx <- lapply(rook_neighbors_unique, function(nb) {

  nb <- as.integer(nb)
  # spdep nb objects use 0 to indicate no neighbors in some representations;
  # remove any zeros or NAs
  nb[nb > 0L & !is.na(nb)]
})

# ============================================================
# STEP 2: Create mapping from (id, year) to matrix position
# ============================================================
# We need cell_data rows to align to a matrix [n_cells, n_years].
# Map cell IDs to row indices (position in id_order).
# Map years to column indices.

id_to_row  <- setNames(seq_len(n_cells), as.character(id_order))
year_to_col <- setNames(seq_len(n_years), as.character(years))

cell_data[, matrix_row := id_to_row[as.character(id)]]
cell_data[, matrix_col := year_to_col[as.character(year)]]

# ============================================================
# STEP 3: Precompute CSR-style flat vectors for neighbor indices
#          (enables faster vectorized gathering)
# ============================================================
# For max/min we need per-cell neighbor index lists.
# For mean we can also use a sparse matrix multiply.

# Number of neighbors per cell
n_neighbors <- vapply(neighbor_cell_idx, length, integer(1))

# Flat neighbor index vector and pointer vector (CSR-style)
flat_nb     <- unlist(neighbor_cell_idx, use.names = FALSE)
nb_ptr      <- c(0L, cumsum(n_neighbors))  # length n_cells + 1

# ============================================================
# STEP 4: Function to reshape a variable into [n_cells x n_years] matrix
# ============================================================
var_to_matrix <- function(dt, var_name, n_cells, n_years) {
  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  mat[cbind(dt$matrix_row, dt$matrix_col)] <- dt[[var_name]]
  mat
}

# ============================================================
# STEP 5: Compute neighbor stats for one variable (vectorized)
# ============================================================
compute_neighbor_stats_fast <- function(var_mat, neighbor_cell_idx,
                                        flat_nb, nb_ptr, n_cells, n_years) {
  # Output matrices
  nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

  # For each cell, gather its neighbor rows and compute stats across all years

  for (i in seq_len(n_cells)) {
    start <- nb_ptr[i] + 1L
    end   <- nb_ptr[i + 1L]
    if (end < start) next  # no neighbors

    nb_idx <- flat_nb[start:end]

    # Extract neighbor sub-matrix: [num_neighbors x n_years]
    # This is a single matrix-subset operation, very fast
    nb_vals <- var_mat[nb_idx, , drop = FALSE]

    if (length(nb_idx) == 1L) {
      # Single neighbor: no need for colMeans etc.
      nb_max[i, ]  <- nb_vals[1L, ]
      nb_min[i, ]  <- nb_vals[1L, ]
      nb_mean[i, ] <- nb_vals[1L, ]
    } else {
      # Vectorized column-wise operations across all 28 years at once
      # Using matrixStats for speed if available, otherwise base R
      nb_max[i, ]  <- apply(nb_vals, 2, max, na.rm = TRUE)
      nb_min[i, ]  <- apply(nb_vals, 2, min, na.rm = TRUE)
      nb_mean[i, ] <- colMeans(nb_vals, na.rm = TRUE)
    }
  }

  # Fix Inf/-Inf from max/min on all-NA columns
  nb_max[is.infinite(nb_max)] <- NA_real_
  nb_min[is.infinite(nb_min)] <- NA_real_

  list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}

# ============================================================
# STEP 6: Even faster version using matrixStats (recommended)
# ============================================================
# If matrixStats is available, colMaxs/colMins/colMeans2 are much faster
# than apply(). But the main loop is over 344K cells, not 6.46M rows,
# so even base R apply is tractable.
#
# For maximal speed, we can also use Rcpp, but the R version below
# should complete in ~10-30 minutes on a 16GB laptop.

if (requireNamespace("matrixStats", quietly = TRUE)) {
  compute_neighbor_stats_fast <- function(var_mat, neighbor_cell_idx,
                                          flat_nb, nb_ptr, n_cells, n_years) {
    nb_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    nb_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (i in seq_len(n_cells)) {
      start <- nb_ptr[i] + 1L
      end   <- nb_ptr[i + 1L]
      if (end < start) next

      nb_idx  <- flat_nb[start:end]
      nb_vals <- var_mat[nb_idx, , drop = FALSE]

      if (length(nb_idx) == 1L) {
        nb_max[i, ]  <- nb_vals[1L, ]
        nb_min[i, ]  <- nb_vals[1L, ]
        nb_mean[i, ] <- nb_vals[1L, ]
      } else {
        nb_max[i, ]  <- matrixStats::colMaxs(nb_vals, na.rm = TRUE)
        nb_min[i, ]  <- matrixStats::colMins(nb_vals, na.rm = TRUE)
        nb_mean[i, ] <- colMeans(nb_vals, na.rm = TRUE)
      }
    }

    nb_max[is.infinite(nb_max)]  <- NA_real_
    nb_min[is.infinite(nb_min)]  <- NA_real_

    list(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
  }
}

# ============================================================
# STEP 7: Main loop â€” process each variable and write back
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Processing neighbor stats for: %s", var_name))

  # Reshape variable to matrix [n_cells x n_years]
  var_mat <- var_to_matrix(cell_data, var_name, n_cells, n_years)

  # Compute neighbor max, min, mean (returns 3 matrices)
  stats <- compute_neighbor_stats_fast(
    var_mat, neighbor_cell_idx, flat_nb, nb_ptr, n_cells, n_years
  )

  # Column names matching original pipeline
  col_max  <- paste0("n_", var_name, "_max")
  col_min  <- paste0("n_", var_name, "_min")
  col_mean <- paste0("n_", var_name, "_mean")

  # Map matrix positions back to cell_data rows using the same (row, col) index
  idx <- cbind(cell_data$matrix_row, cell_data$matrix_col)

  cell_data[, (col_max)  := stats$nb_max[idx]]
  cell_data[, (col_min)  := stats$nb_min[idx]]
  cell_data[, (col_mean) := stats$nb_mean[idx]]

  # Free memory
  rm(var_mat, stats)
  gc()
}

# ============================================================
# STEP 8: Clean up helper columns
# ============================================================
cell_data[, c("matrix_row", "matrix_col") := NULL]

# ============================================================
# STEP 9: Proceed with pre-trained Random Forest prediction
# ============================================================
# The trained RF model is unchanged. cell_data now has exactly the same
# neighbor feature columns (n_<var>_max, n_<var>_min, n_<var>_mean)
# with numerically identical values. Proceed directly:
#
# predictions <- predict(rf_model, newdata = cell_data)
#

message("Neighbor feature computation complete. Ready for RF prediction.")
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup size** | 6.46M entries (cell-year) | 344K entries (cell only) |
| **String hashing** | ~19M paste + hash ops | **Zero** |
| **Inner loop iterations** | 6.46M Ã— 5 vars = 32.3M | 344K Ã— 5 vars = 1.72M |
| **Work per iteration** | Scalar subsetting | Vectorized across 28 years |
| **Estimated runtime** | 86+ hours | ~10â€“30 minutes |
| **Numerical result** | Original | **Identical** (same neighbors, same values, same aggregation functions) |
| **RF model** | Pre-trained | **Untouched** â€” only `predict()` is called |
| **Peak RAM** | Large list overhead | ~344K Ã— 28 Ã— 8 bytes â‰ˆ 77MB per variable matrix â€” well within 16GB |
END_RESPONSE>>>
