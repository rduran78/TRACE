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
run_id: Anthropic_F07_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(nÂ²) list construction via row-by-row `lapply`

The function iterates over **~6.46 million rows**, and for each row it:
1. Looks up the cell's neighbor IDs.
2. Pastes together character keys (`id_year`).
3. Performs named-vector lookups via `idx_lookup[neighbor_keys]`.

Named vector lookup by character key in R is hash-based but still carries overhead per call. Doing this 6.46 million times with string concatenation and subsetting is extremely slow. The `paste()` and named-vector indexing dominate. The resulting `neighbor_lookup` is a list of ~6.46M integer vectors â€” itself a large object.

### Bottleneck 2: `compute_neighbor_stats` â€” repeated per-variable `lapply` over 6.46M elements

For each of the 5 variables, another `lapply` over 6.46M elements extracts neighbor values, removes NAs, and computes max/min/mean. That's 5 Ã— 6.46M = ~32.3M R-level function calls, each allocating small vectors.

### Summary
- **String operations at scale**: `paste()` and character-key lookups on millions of rows.
- **R-level loops**: `lapply` over millions of elements with per-element allocations.
- **Redundant work**: The neighbor *structure* is the same across all 28 years (only the cell-to-cell adjacency matters, which is time-invariant), yet the lookup is built monolithically across all cell-years.

---

## Optimization Strategy

### Key Insight: Separate spatial structure from temporal replication

The rook-neighbor graph is **purely spatial** â€” it doesn't change across years. There are only **344,208 cells**, not 6.46M cell-years. We should:

1. **Work at the cell level for adjacency** (344K cells), not the cell-year level (6.46M rows).
2. **Vectorize the neighbor aggregation** using sparse matrix multiplication / adjacency matrix operations rather than R-level loops.
3. **Use a sparse adjacency matrix** (Matrix package) â€” a 344,208 Ã— 344,208 sparse matrix with ~1.37M nonzero entries is tiny (~20 MB).
4. **Compute neighbor stats via matrix operations**:
   - **Neighbor mean**: Multiply the (row-normalized) adjacency matrix by the variable column, per year. This is a single sparse matrixâ€“vector multiply per year per variable.
   - **Neighbor max and min**: Use the adjacency structure to index and compute group-wise max/min. This can be done efficiently with `data.table` grouping or with a custom sparse-matrix approach.

### Estimated speedup
- Eliminates all `paste()` / character-key lookups.
- Replaces 6.46M R-level `lapply` iterations with vectorized sparse-matrix operations.
- Expected runtime: **minutes**, not 86+ hours.

### Constraints preserved
- âœ… Trained Random Forest model is untouched.
- âœ… Numerical results are identical (same max, min, mean of the same neighbor sets).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, Matrix
# Preserves the exact numerical estimand of the original implementation.
# Does NOT retrain or modify the Random Forest model.
# =============================================================================

library(data.table)
library(Matrix)

#' Build a sparse binary adjacency matrix from an spdep::nb object.
#'
#' @param nb_obj   An nb object (list of integer neighbor index vectors),
#'                 e.g. rook_neighbors_unique.
#' @param n        Number of spatial units (length of nb_obj).
#' @return A dgCMatrix (sparse, column-compressed) of dimension n x n.
build_adjacency_matrix <- function(nb_obj, n = length(nb_obj)) {
  # Build COO triplets
  from <- rep(seq_len(n), lengths(nb_obj))
  to   <- unlist(nb_obj)
  W    <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  return(W)
}

#' Compute neighbor max, min, and mean for one variable across all years,
#' using sparse-matrix operations and data.table.
#'
#' @param dt        A data.table with columns: id, year, and the target variable.
#' @param var_name  Character name of the variable.
#' @param W         Sparse adjacency matrix (n_cells x n_cells).
#' @param id_order  Integer vector of cell IDs in the order matching W's
#'                  row/column indices (same order as the nb object).
#' @return dt with three new columns appended:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean
compute_neighbor_features_sparse <- function(dt, var_name, W, id_order) {

  n_cells <- length(id_order)

  # Map cell id -> matrix row/col index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Add matrix index column
  dt[, .mat_idx := id_to_idx[as.character(id)]]

  # Get sorted unique years

  years <- sort(unique(dt$year))

  # Pre-extract the adjacency structure once (CSC format -> per-row neighbors)
  # Convert W to dgRMatrix (row-compressed) for efficient row slicing,

  # or just extract from the dgCMatrix column structure on the transpose.
  # Actually, we'll use W in CSC form and do matrix-vector multiplies for mean,
  # and use the explicit neighbor list for max/min.

  # --- Extract neighbor list from sparse matrix (much faster than nb for indexing) ---
  Wt <- t(W)  # transpose so that column j of Wt gives the neighbors of cell j
  # For a symmetric matrix this is the same, but let's be safe.

  # Degree vector (number of neighbors per cell)
  degree <- diff(W@p)  # W is CSC; diff of column pointers gives column counts

  # Actually for row-wise: use rowSums
  degree_vec <- as.numeric(W %*% rep(1, n_cells))

  # Pre-allocate result columns
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]

  # Process year by year â€” each year is an independent cross-section

  for (yr in years) {
    # Subset rows for this year
    yr_mask <- dt$year == yr

    # Build a full-length vector of the variable aligned to matrix indices
    # (NA for cells not present in this year)
    vals_full <- rep(NA_real_, n_cells)
    yr_idx    <- dt$.mat_idx[yr_mask]
    yr_vals   <- dt[[var_name]][yr_mask]
    vals_full[yr_idx] <- yr_vals

    # ---- Neighbor MEAN via sparse matrix-vector multiply ----
    # W %*% vals_full gives the sum of neighbor values for each cell.
    # Divide by degree to get mean. Cells with all-NA neighbors get NA.

    # Replace NA with 0 for the multiply, but track counts of non-NA neighbors
    vals_for_sum <- vals_full
    vals_for_sum[is.na(vals_for_sum)] <- 0

    not_na <- as.numeric(!is.na(vals_full))

    neighbor_sum   <- as.numeric(W %*% vals_for_sum)
    neighbor_count <- as.numeric(W %*% not_na)

    neighbor_mean_full <- ifelse(neighbor_count > 0,
                                 neighbor_sum / neighbor_count,
                                 NA_real_)

    # Assign mean back to dt
    dt[yr_mask, (col_mean) := neighbor_mean_full[yr_idx]]

    # ---- Neighbor MAX and MIN ----
    # Strategy: use the sparse matrix structure directly.
    # For each cell i, neighbors are the nonzero entries in row i of W.
    # We iterate over cells present this year using vectorized group operations.

    # Extract row pointers from W in CSR format
    W_csr <- as(W, "RsparseMatrix")  # dgRMatrix: row-compressed

    # For each cell index in yr_idx, gather neighbor values and compute max/min
    # We vectorize this using data.table:
    # Build a long table of (cell_mat_idx, neighbor_val)

    # Pre-extract all neighbor pairs from W_csr for cells in yr_idx
    # W_csr@i is 0-based row index (but for RsparseMatrix, @j gives column indices)
    # W_csr@p is row pointer (length n_cells + 1)

    # Only process cells that appear this year
    # For each cell in yr_idx, get its neighbors' values
    p <- W_csr@p
    j <- W_csr@j  # 0-based column indices

    # Vectorized extraction:
    row_lengths <- diff(p)[yr_idx]  # number of neighbors for each cell in yr_idx
    total_pairs <- sum(row_lengths)

    if (total_pairs == 0) next

    # Build cell index (repeated) and neighbor column indices
    cell_rep <- rep(seq_along(yr_idx), row_lengths)

    # Gather the column indices for all neighbors of all yr_idx cells
    # p is 0-based, so row i (1-based) has entries from p[i]+1 to p[i+1]
    starts <- p[yr_idx] + 1L    # 1-based start in j
    ends   <- p[yr_idx + 1L]    # 1-based end in j

    # Build index into j
    j_positions <- sequence(row_lengths, from = starts, by = 1L)
    neighbor_col_idx <- j[j_positions] + 1L  # convert 0-based to 1-based

    # Look up neighbor values
    neighbor_vals <- vals_full[neighbor_col_idx]

    # Now compute grouped max and min using data.table
    agg_dt <- data.table(cell_seq = cell_rep, nval = neighbor_vals)
    # Remove NA neighbor values
    agg_dt <- agg_dt[!is.na(nval)]

    if (nrow(agg_dt) > 0) {
      agg <- agg_dt[, .(nmax = max(nval), nmin = min(nval)), by = cell_seq]

      # Map back: cell_seq indexes into yr_idx, which indexes into yr_mask positions
      yr_positions <- which(yr_mask)

      dt_row_for_max <- yr_positions[agg$cell_seq]
      set(dt, i = dt_row_for_max, j = col_max, value = agg$nmax)
      set(dt, i = dt_row_for_max, j = col_min, value = agg$nmin)
    }
  }

  # Clean up temporary column

  dt[, .mat_idx := NULL]

  return(dt)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Convert to data.table for performance (if not already)
cell_data <- as.data.table(cell_data)

# id_order: the vector of cell IDs in the order that matches rook_neighbors_unique
# (i.e., id_order[k] is the cell ID for the k-th element of the nb object)

# 1. Build sparse adjacency matrix (once â€” takes seconds)
W <- build_adjacency_matrix(rook_neighbors_unique, n = length(id_order))

# Optional: if the nb object is not symmetric, symmetrize for safety
# (rook contiguity should be symmetric, but just in case)
# W <- W + t(W)
# W@x <- pmin(W@x, 1)  # ensure binary

# 2. Pre-convert to CSR once (avoid repeated conversion inside the function)
#    The function does this internally per call, but we can also refactor.
#    For clarity, we leave it inside.

# 3. Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Also pre-compute the CSR representation once outside the loop
W_csr <- as(W, "RsparseMatrix")

# Optimized version that accepts pre-computed W_csr:
for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  cell_data <- compute_neighbor_features_sparse(cell_data, var_name, W, id_order)
}

# Convert back to data.frame if downstream code expects it
# cell_data <- as.data.frame(cell_data)

message("Done. All neighbor features computed.")
```

---

## Further Optimization: Avoid Repeated CSR Conversion

The function above converts `W` to CSR inside each call. Here is a streamlined version that does it once:

```r
compute_all_neighbor_features <- function(dt, var_names, W, id_order) {

  n_cells <- length(id_order)
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, .mat_idx := id_to_idx[as.character(id)]]

  years <- sort(unique(dt$year))

  # Sparse matrix preparations (once)
  W_csr <- as(W, "RsparseMatrix")
  p <- W_csr@p
  j_vec <- W_csr@j

  ones <- rep(1, n_cells)

  # Pre-allocate all output columns
  for (var_name in var_names) {
    dt[, paste0(var_name, "_neighbor_max")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_min")  := NA_real_]
    dt[, paste0(var_name, "_neighbor_mean") := NA_real_]
  }

  for (yr in years) {
    yr_mask <- dt$year == yr
    yr_idx  <- dt$.mat_idx[yr_mask]
    yr_positions <- which(yr_mask)

    # Row lengths and neighbor column indices (same for all variables this year)
    row_lengths <- diff(p)[yr_idx]
    total_pairs <- sum(row_lengths)
    if (total_pairs == 0) next

    starts <- p[yr_idx] + 1L
    j_positions <- sequence(row_lengths, from = starts, by = 1L)
    neighbor_col_idx <- j_vec[j_positions] + 1L
    cell_rep <- rep(seq_along(yr_idx), row_lengths)

    for (var_name in var_names) {
      vals_full <- rep(NA_real_, n_cells)
      vals_full[yr_idx] <- dt[[var_name]][yr_mask]

      # --- MEAN via sparse mat-vec ---
      vals_0 <- vals_full; vals_0[is.na(vals_0)] <- 0
      not_na <- as.numeric(!is.na(vals_full))
      n_sum   <- as.numeric(W %*% vals_0)
      n_count <- as.numeric(W %*% not_na)
      n_mean  <- ifelse(n_count > 0, n_sum / n_count, NA_real_)
      set(dt, i = yr_positions, j = paste0(var_name, "_neighbor_mean"),
          value = n_mean[yr_idx])

      # --- MAX / MIN via grouped aggregation ---
      nvals <- vals_full[neighbor_col_idx]
      agg_dt <- data.table(cs = cell_rep, nv = nvals)
      agg_dt <- agg_dt[!is.na(nv)]
      if (nrow(agg_dt) > 0) {
        agg <- agg_dt[, .(mx = max(nv), mn = min(nv)), by = cs]
        set(dt, i = yr_positions[agg$cs],
            j = paste0(var_name, "_neighbor_max"), value = agg$mx)
        set(dt, i = yr_positions[agg$cs],
            j = paste0(var_name, "_neighbor_min"), value = agg$mn)
      }
    }
    if (yr %% 5 == 0) message(sprintf("  Year %d done.", yr))
  }

  dt[, .mat_idx := NULL]
  return(dt)
}

# --- Run ---
cell_data <- as.data.table(cell_data)
W <- build_adjacency_matrix(rook_neighbors_unique)

cell_data <- compute_all_neighbor_features(
  cell_data,
  c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
  W,
  id_order
)
message("All neighbor features computed.")
```

---

## Performance Estimate

| Component | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `paste` + hash lookups) | ~seconds (sparse matrix from nb) |
| Neighbor mean (per var) | ~hours (6.46M `lapply` calls) | ~seconds (28 sparse mat-vec multiplies) |
| Neighbor max/min (per var) | ~hours (6.46M `lapply` calls) | ~minutes (28 Ã— vectorized `data.table` group-by) |
| **Total** | **86+ hours** | **~5â€“15 minutes** |

Memory: The sparse matrix W is ~344K Ã— 344K with ~1.37M entries â‰ˆ **~22 MB**. The `data.table` operations work in-place. Total memory well within 16 GB.

**The trained Random Forest model is completely untouched. The numerical outputs (neighbor max, min, mean) are identical to the original implementation.**
END_RESPONSE>>>
