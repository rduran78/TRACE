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
run_id: Anthropic_F05_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Inside the Loop

`build_neighbor_lookup` iterates over all ~6.46 million rows. For **each row**, it:

1. Converts an ID to character and looks it up in a named vector â€” O(1) amortized but with string allocation overhead.
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current year to form string keys â€” allocates new strings per row.
4. **Looks up** those keys in `idx_lookup` (a named vector of 6.46M entries) â€” named-vector lookup in R is hash-based but still involves repeated string hashing.

With ~6.46M rows and an average of ~8 rook neighbors per cell (1,373,394 directed relationships / 344,208 cells â‰ˆ 4 per cell, but bidirectional â‰ˆ 8), this means roughly **50+ million `paste` and hash-lookup operations**. The named-vector approach has high constant factors in R (string allocation, hashing, GC pressure).

### The Deeper Structural Issue

The neighbor topology is **year-invariant**: cell A's neighbors are the same in 1992 as in 2019. Yet the current code re-discovers the mapping from "cell â†’ neighbor rows" independently for every cell-year row. This means the same spatial lookup is repeated 28 times per cell.

Furthermore, `compute_neighbor_stats` is already vectorized over the lookup, but the lookup itself was built row-by-row with string operations. The entire string-keying strategy is unnecessary if we reformulate the problem.

## Optimization Strategy

### Key Insight: Separate Space from Time

Since the neighbor graph is purely spatial and time-invariant:

1. **Build a spatial-only neighbor lookup once** â€” a list of length 344,208 mapping each cell index to its neighbor cell indices (integers, no strings).
2. **Build a year-to-row mapping** â€” for each year, a fast integer vector mapping cell position to row index.
3. **Vectorize the neighbor statistics** using matrix operations: reshape each variable into a `cells Ã— years` matrix, then compute neighbor stats using the spatial neighbor list on columns of the matrix.

This eliminates all string operations, reduces the problem from 6.46M row-level iterations to 344K cell-level iterations (or fully vectorized matrix operations), and cuts memory churn dramatically.

### Expected Speedup

- From ~86 hours to **minutes** (estimated 2â€“10 minutes depending on RAM pressure).
- No string allocation, no hash lookups, pure integer indexing.
- The Random Forest model is untouched; the numerical output is identical.

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical results (max, min, mean of neighbor values)
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # Convert to data.table for fast manipulation (non-destructive)
  dt <- as.data.table(cell_data)

  # ------------------------------------------------------------------
  # Step 1: Establish a canonical cell ordering (spatial index)
  # ------------------------------------------------------------------
  # id_order is the vector of cell IDs in the order matching the nb object.
  # Create a map: cell_id -> spatial_index (integer position in id_order)
  n_cells <- length(id_order)
  id_to_spatial <- setNames(seq_len(n_cells), as.character(id_order))

  # Assign each row its spatial index
  dt[, spatial_idx := id_to_spatial[as.character(id)]]

  # ------------------------------------------------------------------
  # Step 2: Build spatial-only neighbor list (integer indices)
  # ------------------------------------------------------------------
  # rook_neighbors_unique is an nb object: a list of length n_cells,

  # where each element is an integer vector of neighbor positions
  # (indices into id_order). nb objects use 0 for "no neighbors".
  # We just need to clean it.
  spatial_neighbors <- lapply(seq_len(n_cells), function(s) {
    nb <- rook_neighbors_unique[[s]]
    # spdep::nb uses 0L to indicate no neighbors in a single-element vector
    nb <- nb[nb != 0L]
    as.integer(nb)
  })

  # ------------------------------------------------------------------
  # Step 3: Sort data by (spatial_idx, year) for matrix reshaping
  # ------------------------------------------------------------------
  # Ensure consistent year set
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_col <- setNames(seq_len(n_years), as.character(years))

  dt[, year_col := year_to_col[as.character(year)]]

  # Sort by spatial_idx, then year for predictable ordering
  setorder(dt, spatial_idx, year)

  # ------------------------------------------------------------------
  # Step 4: Build row-index matrix: cells x years -> row in dt
  # ------------------------------------------------------------------
  # Not all cells may appear in all years, so we build this carefully.
  # row_matrix[s, t] = row index in dt for spatial cell s, year column t
  # NA if that cell-year doesn't exist.

  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(dt$spatial_idx, dt$year_col)] <- seq_len(nrow(dt))

  # ------------------------------------------------------------------
  # Step 5: For each variable, build a cells x years value matrix,
  #         compute neighbor stats via spatial_neighbors, write back.
  # ------------------------------------------------------------------

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Processing neighbor stats for: %s\n", var_name))

    # Build value matrix: cells x years
    val_vec <- dt[[var_name]]
    val_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    valid_rows <- !is.na(row_matrix)
    val_matrix[valid_rows] <- val_vec[row_matrix[valid_rows]]

    # Allocate output matrices: cells x years
    max_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_matrix  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    mean_matrix <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    # Compute neighbor stats per cell (loop over cells, vectorized over years)
    for (s in seq_len(n_cells)) {
      nb_idx <- spatial_neighbors[[s]]
      if (length(nb_idx) == 0L) next

      # neighbor_vals is a (n_neighbors x n_years) submatrix
      if (length(nb_idx) == 1L) {
        # Single neighbor: treat as a 1-row matrix
        nb_vals <- matrix(val_matrix[nb_idx, ], nrow = 1L, ncol = n_years)
      } else {
        nb_vals <- val_matrix[nb_idx, , drop = FALSE]
      }

      # For each year (column), compute max/min/mean ignoring NAs
      # Use colMeans, colMaxs-style operations via matrixStats or base apply
      # For performance, use vectorized column operations:
      max_matrix[s, ]  <- apply(nb_vals, 2L, max,  na.rm = TRUE)
      min_matrix[s, ]  <- apply(nb_vals, 2L, min,  na.rm = TRUE)
      mean_matrix[s, ] <- colMeans(nb_vals, na.rm = TRUE)
    }

    # Replace Inf/-Inf from max/min on all-NA columns with NA
    max_matrix[is.infinite(max_matrix)]   <- NA_real_
    min_matrix[is.infinite(min_matrix)]   <- NA_real_
    mean_matrix[is.nan(mean_matrix)]      <- NA_real_

    # Map back from (cells x years) matrices to dt rows
    valid <- which(!is.na(row_matrix), arr.ind = TRUE)
    target_rows <- row_matrix[valid]

    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    set(dt, i = target_rows, j = col_max,  value = max_matrix[valid])
    set(dt, i = target_rows, j = col_min,  value = min_matrix[valid])
    set(dt, i = target_rows, j = col_mean, value = mean_matrix[valid])

    cat(sprintf("  Done: %s\n", var_name))
  }

  # ------------------------------------------------------------------
  # Step 6: Clean up helper columns and return
  # ------------------------------------------------------------------
  dt[, c("spatial_idx", "year_col") := NULL]

  # Return as data.frame if the input was a data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_optimized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” just use cell_data for predict().
```

## Further Optimization: Eliminate the Inner `apply` with `matrixStats`

The `apply(..., 2, max)` call inside the cell loop still has overhead. If the `matrixStats` package is available, replace the inner loop body for a significant constant-factor speedup:

```r
# install.packages("matrixStats") if not already available
library(matrixStats)

# Replace the inner cell loop with:
for (s in seq_len(n_cells)) {
  nb_idx <- spatial_neighbors[[s]]
  if (length(nb_idx) == 0L) next

  nb_vals <- val_matrix[nb_idx, , drop = FALSE]

  max_matrix[s, ]  <- colMaxs(nb_vals,  na.rm = TRUE)
  min_matrix[s, ]  <- colMins(nb_vals,  na.rm = TRUE)
  mean_matrix[s, ] <- colMeans2(nb_vals, na.rm = TRUE)
}
```

## Even Further: Fully Vectorized (No Cell Loop) via Sparse Matrix Multiplication

For the **mean**, we can eliminate the cell loop entirely using a sparse adjacency matrix. Max and min still require per-cell work, but mean is a linear operation:

```r
library(Matrix)

# Build sparse row-normalized adjacency matrix (once, reuse for all vars)
# W[s, j] = 1/degree(s) if j is a neighbor of s, else 0
i_idx <- rep(seq_len(n_cells), lengths(spatial_neighbors))
j_idx <- unlist(spatial_neighbors)
degrees <- lengths(spatial_neighbors)
degrees[degrees == 0] <- 1  # avoid division by zero; those rows will be all-zero anyway

W <- sparseMatrix(
  i = i_idx,
  j = j_idx,
  x = rep(1 / degrees[i_idx != 0], lengths(spatial_neighbors)),
  dims = c(n_cells, n_cells)
)

# For each variable, neighbor mean = W %*% val_matrix
# (handles NAs approximately; for exact NA handling, use a second pass)
mean_matrix <- as.matrix(W %*% val_matrix)

# For exact NA-aware mean:
# Replace NAs with 0 in val_matrix, compute sum and count separately
val_nona <- val_matrix
val_nona[is.na(val_nona)] <- 0
indicator <- (!is.na(val_matrix)) * 1.0

# Build unnormalized adjacency (just 1s for neighbors)
W_raw <- sparseMatrix(i = i_idx, j = j_idx, x = 1, dims = c(n_cells, n_cells))

neighbor_sum   <- as.matrix(W_raw %*% val_nona)
neighbor_count <- as.matrix(W_raw %*% indicator)
mean_matrix    <- neighbor_sum / neighbor_count  # NaN where count=0, which is correct
mean_matrix[neighbor_count == 0] <- NA_real_
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup structure** | 6.46M string keys in named vector | Integer matrix indexing |
| **String operations** | ~50M `paste()` + hash lookups | Zero |
| **Redundant work** | Same spatial lookup repeated 28Ã— per cell | Spatial lookup done once |
| **Inner loop granularity** | Per row (6.46M iterations) | Per cell (344K iterations), vectorized over years |
| **Estimated runtime** | 86+ hours | 2â€“10 minutes |
| **Numerical result** | Baseline | Identical |
| **RF model** | Unchanged | Unchanged |
END_RESPONSE>>>
