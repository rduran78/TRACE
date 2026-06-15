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
run_id: Anthropic_F05_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### The Core Problem: Row-by-Row `paste`+`match` Inside `lapply` Over 6.46M Rows

1. **`build_neighbor_lookup`** iterates over every row (`~6.46M`) and, for each row, constructs string keys via `paste()` and looks them up in a named vector (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is hash-based but still carries per-call overhead. With ~6.46M iterations Ã— ~4 neighbors on average, that's ~25.8M `paste` + hash-probe operations. The `idx_lookup` vector itself (6.46M entries) is built once, which is fine, but the per-row key construction is the bottleneck.

2. **The lookup is year-redundant.** Neighbors are purely spatial â€” they don't change across years. Yet the code re-discovers "which rows belong to neighbor cell X in year Y" independently for every row. Since every cell appears in every year, the neighbor *row-index offsets* are **structurally identical across years** and can be computed once from the spatial topology alone.

3. **`compute_neighbor_stats`** is called 5 times (once per variable), each time re-traversing the 6.46M-entry lookup list. This is comparatively cheap versus the build step, but a vectorized/matrix approach eliminates the `lapply` entirely.

### Summary of Redundancies

| Layer | What's repeated | Scale |
|---|---|---|
| String construction | `paste(id, year)` per row | 6.46M Ã— ~4 neighbors |
| Hash lookup | Named-vector indexing per row | 6.46M Ã— ~4 neighbors |
| Year dimension | Same spatial topology re-resolved per year | 28Ã— redundant |
| Variable loop | `lapply` over 6.46M list elements per variable | 5Ã— |

---

## Optimization Strategy

### Key Insight: Separate the Spatial Topology from the Temporal Dimension

Since every cell appears in every year, and neighbors are purely spatial:

1. **Build a spatial neighbor matrix once** â€” a sparse matrix or a simple integer-index list mapping each *cell* (not cell-year) to its neighbor *cells*. This is just `rook_neighbors_unique` translated to integer indices. Cost: O(344K cells).

2. **Sort/index the panel so that rows for the same cell are contiguous or easily addressable by cell-index and year-offset.** If the data is sorted by `(id, year)`, then the row for cell `c` in year `y` is simply `(c_index - 1) * 28 + (y - 1992) + 1`. This replaces all hash lookups with arithmetic.

3. **Vectorize the neighbor statistics** using matrix operations. Reshape each variable to a `344,208 Ã— 28` matrix (cells Ã— years). For each cell, its neighbor values in any year are just the neighbor-row slices of that matrix column. We can compute max/min/mean across neighbors using sparse-matrix multiplication (for mean) and row-wise operations.

4. **Use a sparse neighbor matrix `W`** and matrix multiplication `W %*% X` to get neighbor sums in one shot, then divide by neighbor counts for means. For min/max, iterate over cells (not cell-years) â€” reducing the loop from 6.46M to 344K, a **~18.7Ã— speedup** on the inner loop alone, with each iteration doing simple integer-indexed vector subsetting instead of hash lookups.

### Expected Speedup

| Component | Before | After | Factor |
|---|---|---|---|
| Lookup build | ~6.46M hash lookups | Eliminated (arithmetic indexing) | âˆž |
| Neighbor stats (mean) | 6.46M Ã— 5 `lapply` | 5 sparse matmuls (344K Ã— 344K sparse) Ã— 28 cols | ~100â€“500Ã— |
| Neighbor stats (min/max) | 6.46M Ã— 5 | 344K Ã— 5 loops over integer vectors | ~18Ã— |
| Total estimated time | 86+ hours | **~5â€“15 minutes** | ~350â€“1000Ã— |

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor-feature construction
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact same numerical output columns; trained RF model untouched
# =============================================================================

library(Matrix)  # for sparse matrix operations

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------
  # 1. Establish cell-to-index and row-addressing scheme
  # -------------------------------------------------------------------
  # Ensure data is sorted by (id, year) so we can use arithmetic indexing.
  # If not already sorted, sort and record the original order to restore later.

  cell_data$.orig_order <- seq_len(nrow(cell_data))
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

  unique_ids   <- sort(unique(cell_data$id))
  unique_years <- sort(unique(cell_data$year))
  n_cells      <- length(unique_ids)
  n_years      <- length(unique_years)

  stopifnot(nrow(cell_data) == n_cells * n_years)  # balanced panel check

  # Map cell id -> integer index (1..n_cells)
  id_to_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))

  # Map year -> integer offset (1..n_years)
  year_to_offset <- setNames(seq_along(unique_years), as.character(unique_years))

  # Row address function: cell index c (1-based), year offset t (1-based)
  # row = (c - 1) * n_years + t
  # This works because data is sorted by (id, year).

  cat("Panel dimensions:", n_cells, "cells Ã—", n_years, "years =",
      n_cells * n_years, "rows\n")

  # -------------------------------------------------------------------
  # 2. Build spatial neighbor list in terms of cell indices
  #    (translate from id_order / nb object to integer cell indices)
  # -------------------------------------------------------------------
  # id_order maps position-in-nb-object -> cell id
  # We need: for each cell index (in unique_ids order), its neighbor cell indices

  id_order_to_idx <- id_to_idx[as.character(id_order)]

  # rook_neighbors_unique is an nb object: list of integer vectors
  # rook_neighbors_unique[[k]] gives the positions (in id_order) of neighbors of id_order[k]

  # Build neighbor list indexed by our cell index
  neighbor_list <- vector("list", n_cells)

  for (k in seq_along(id_order)) {
    cell_idx <- id_order_to_idx[k]
    nb_positions <- rook_neighbors_unique[[k]]
    # nb objects use 0 to indicate no neighbors
    if (length(nb_positions) == 1 && nb_positions[1] == 0L) {
      neighbor_list[[cell_idx]] <- integer(0)
    } else {
      neighbor_list[[cell_idx]] <- as.integer(id_order_to_idx[nb_positions])
    }
  }

  # -------------------------------------------------------------------
  # 3. Build sparse neighbor matrix W (n_cells Ã— n_cells)
  #    W[i,j] = 1 if j is a neighbor of i
  # -------------------------------------------------------------------
  # This is used for computing neighbor means via matrix multiplication.

  n_entries <- sum(vapply(neighbor_list, length, integer(1)))
  cat("Building sparse W matrix:", n_cells, "Ã—", n_cells,
      "with", n_entries, "non-zero entries\n")

  row_i <- integer(n_entries)
  col_j <- integer(n_entries)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- neighbor_list[[i]]
    len <- length(nb)
    if (len > 0L) {
      row_i[pos:(pos + len - 1L)] <- i
      col_j[pos:(pos + len - 1L)] <- nb
      pos <- pos + len
    }
  }

  W <- sparseMatrix(i = row_i, j = col_j, x = 1.0,
                    dims = c(n_cells, n_cells))

  # Neighbor count per cell (used for mean calculation)
  neighbor_count <- as.numeric(W %*% rep(1.0, n_cells))  # length n_cells

  rm(row_i, col_j)  # free memory

  # -------------------------------------------------------------------
  # 4. For each source variable, compute neighbor max, min, mean
  #    using matrix operations + vectorized cell-level loops
  # -------------------------------------------------------------------

  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")

    vals <- cell_data[[var_name]]

    # Reshape to matrix: rows = cells, cols = years
    # Row c, col t corresponds to original row (c-1)*n_years + t
    V <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = TRUE)

    # --- Neighbor mean via sparse matrix multiplication ---
    # For each year (column), compute W %*% V[,t] = sum of neighbor values
    # Then divide by neighbor_count.
    # Handle NAs: we need mean of non-NA neighbor values.

    # To correctly handle NAs:
    # - Replace NA with 0 for sum computation
    # - Count non-NA neighbors per cell-year
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0

    # Indicator of non-NA
    V_valid <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)

    # Sparse matmul: W %*% V_nona gives neighbor sums (treating NA as 0)
    # W %*% V_valid gives count of non-NA neighbors per cell-year
    neighbor_sum   <- as.matrix(W %*% V_nona)    # n_cells Ã— n_years
    neighbor_nvalid <- as.matrix(W %*% V_valid)   # n_cells Ã— n_years

    neighbor_mean_mat <- neighbor_sum / neighbor_nvalid
    neighbor_mean_mat[neighbor_nvalid == 0] <- NA

    # --- Neighbor min and max: must iterate over cells (344K, not 6.46M) ---
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (ci in seq_len(n_cells)) {
      nb <- neighbor_list[[ci]]
      if (length(nb) == 0L) next
      # Extract the sub-matrix for all neighbors across all years
      # nb_vals: length(nb) Ã— n_years matrix
      nb_vals <- V[nb, , drop = FALSE]

      if (length(nb) == 1L) {
        # Single neighbor: max = min = that value (may be NA)
        neighbor_max_mat[ci, ] <- nb_vals[1L, ]
        neighbor_min_mat[ci, ] <- nb_vals[1L, ]
      } else {
        # Column-wise max and min ignoring NAs
        # Use matrixStats if available for speed, otherwise base R
        for (t in seq_len(n_years)) {
          col_vals <- nb_vals[, t]
          col_vals <- col_vals[!is.na(col_vals)]
          if (length(col_vals) > 0L) {
            neighbor_max_mat[ci, t] <- max(col_vals)
            neighbor_min_mat[ci, t] <- min(col_vals)
          }
        }
      }

      # Progress indicator (every 50K cells)
      if (ci %% 50000L == 0L) {
        cat("  ", var_name, ": processed", ci, "/", n_cells, "cells\n")
      }
    }

    # --- Flatten back to panel vector (row-major: cell Ã— year) ---
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)

    cell_data[[max_col_name]]  <- as.vector(t(neighbor_max_mat))
    cell_data[[min_col_name]]  <- as.vector(t(neighbor_min_mat))
    cell_data[[mean_col_name]] <- as.vector(t(neighbor_mean_mat))

    # Wait â€” the flattening must match the sorted row order.
    # Data is sorted by (id, year). Row order is:
    #   cell_1-year_1, cell_1-year_2, ..., cell_1-year_28,
    #   cell_2-year_1, ...
    # Matrix is stored row-major as: V[c, t] -> row (c-1)*n_years + t
    # as.vector(t(M)) reads M row by row, which gives exactly this order. âœ“

    rm(V, V_nona, V_valid, neighbor_sum, neighbor_nvalid,
       neighbor_mean_mat, neighbor_max_mat, neighbor_min_mat)
  }

  # -------------------------------------------------------------------
  # 5. Restore original row order
  # -------------------------------------------------------------------
  cell_data <- cell_data[order(cell_data$.orig_order), ]
  cell_data$.orig_order <- NULL

  return(cell_data)
}


# =============================================================================
# Even faster min/max using matrixStats (if available)
# =============================================================================
# The inner loop over cells with the nested year loop can be accelerated
# significantly with matrixStats::colMaxs / colMins.
# Drop-in replacement for the min/max block:

build_neighbor_features_optimized_fast <- function(cell_data,
                                                    id_order,
                                                    rook_neighbors_unique,
                                                    neighbor_source_vars) {

  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    message("matrixStats not available; falling back to base R version.")
    return(build_neighbor_features_optimized(cell_data, id_order,
                                              rook_neighbors_unique,
                                              neighbor_source_vars))
  }

  cell_data$.orig_order <- seq_len(nrow(cell_data))
  cell_data <- cell_data[order(cell_data$id, cell_data$year), ]

  unique_ids   <- sort(unique(cell_data$id))
  unique_years <- sort(unique(cell_data$year))
  n_cells      <- length(unique_ids)
  n_years      <- length(unique_years)

  stopifnot(nrow(cell_data) == n_cells * n_years)

  id_to_idx <- setNames(seq_along(unique_ids), as.character(unique_ids))
  id_order_to_idx <- id_to_idx[as.character(id_order)]

  # Build neighbor list
  neighbor_list <- vector("list", n_cells)
  for (k in seq_along(id_order)) {
    cell_idx <- id_order_to_idx[k]
    nb_positions <- rook_neighbors_unique[[k]]
    if (length(nb_positions) == 1 && nb_positions[1] == 0L) {
      neighbor_list[[cell_idx]] <- integer(0)
    } else {
      neighbor_list[[cell_idx]] <- as.integer(id_order_to_idx[nb_positions])
    }
  }

  # Sparse matrix W
  n_entries <- sum(vapply(neighbor_list, length, integer(1)))
  row_i <- integer(n_entries)
  col_j <- integer(n_entries)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb <- neighbor_list[[i]]
    len <- length(nb)
    if (len > 0L) {
      row_i[pos:(pos + len - 1L)] <- i
      col_j[pos:(pos + len - 1L)] <- nb
      pos <- pos + len
    }
  }
  W <- sparseMatrix(i = row_i, j = col_j, x = 1.0,
                    dims = c(n_cells, n_cells))
  rm(row_i, col_j)

  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "\n")

    vals <- cell_data[[var_name]]
    V <- matrix(vals, nrow = n_cells, ncol = n_years, byrow = TRUE)

    # --- Mean (sparse matmul, NA-safe) ---
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0
    V_valid <- matrix(as.numeric(!is.na(V)), nrow = n_cells, ncol = n_years)

    neighbor_sum    <- as.matrix(W %*% V_nona)
    neighbor_nvalid <- as.matrix(W %*% V_valid)
    neighbor_mean_mat <- neighbor_sum / neighbor_nvalid
    neighbor_mean_mat[neighbor_nvalid == 0] <- NA

    # --- Min / Max (cell-level loop with matrixStats) ---
    neighbor_max_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (ci in seq_len(n_cells)) {
      nb <- neighbor_list[[ci]]
      if (length(nb) == 0L) next
      nb_vals <- V[nb, , drop = FALSE]  # len(nb) Ã— n_years

      if (length(nb) == 1L) {
        neighbor_max_mat[ci, ] <- nb_vals[1L, ]
        neighbor_min_mat[ci, ] <- nb_vals[1L, ]
      } else {
        neighbor_max_mat[ci, ] <- matrixStats::colMaxs(nb_vals, na.rm = TRUE)
        neighbor_min_mat[ci, ] <- matrixStats::colMins(nb_vals, na.rm = TRUE)
      }

      if (ci %% 100000L == 0L) cat("  ", var_name, ":", ci, "/", n_cells, "\n")
    }

    # Fix -Inf/Inf from all-NA columns
    neighbor_max_mat[is.infinite(neighbor_max_mat)] <- NA
    neighbor_min_mat[is.infinite(neighbor_min_mat)] <- NA

    # Flatten and assign
    cell_data[[paste0("neighbor_max_", var_name)]]  <- as.vector(t(neighbor_max_mat))
    cell_data[[paste0("neighbor_min_", var_name)]]   <- as.vector(t(neighbor_min_mat))
    cell_data[[paste0("neighbor_mean_", var_name)]] <- as.vector(t(neighbor_mean_mat))

    rm(V, V_nona, V_valid, neighbor_sum, neighbor_nvalid,
       neighbor_mean_mat, neighbor_max_mat, neighbor_min_mat)
    gc()
  }

  cell_data <- cell_data[order(cell_data$.orig_order), ]
  cell_data$.orig_order <- NULL
  return(cell_data)
}


# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================
#
# # Original code:
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
#
# # Optimized replacement (single call):
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_optimized_fast(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The trained Random Forest model is untouched â€” only feature columns are
# # reconstructed with identical numerical values. Prediction proceeds as before:
# # predictions <- predict(rf_model, cell_data[, predictor_cols])
```

---

## Why This Works and What Changed

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string `paste` + hash probes | Eliminated entirely â€” arithmetic row addressing via sorted panel |
| **Neighbor discovery** | Redone per cell-year (6.46M times) | Done once per cell (344K times), reused across all 28 years |
| **Mean computation** | `lapply` over 6.46M entries, 5 variables | 5 sparse matrix multiplications (`W %*% V`), each ~O(n_entries Ã— n_years) â‰ˆ 38M FLOPs, fully vectorized in C |
| **Min/Max computation** | `lapply` over 6.46M entries | Loop over 344K cells with `matrixStats::colMaxs/colMins` (vectorized C per cell) |
| **Memory** | 6.46M-element list of integer vectors (~500MB+) | ~344K-entry neighbor list + one `n_cells Ã— n_years` matrix at a time (~77MB), well within 16GB |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** |
| **Numerical output** | max/min/mean of non-NA neighbor values per cell-year | Identical â€” same neighbor sets, same aggregation functions, same NA handling |
| **RF model** | N/A | Completely untouched â€” only the feature columns are recomputed with identical values |
END_RESPONSE>>>
