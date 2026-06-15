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
run_id: Anthropic_F05_T10_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's why:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` runs an `lapply` over every row (6.46M iterations). Inside each iteration it:

1. **Calls `paste()`** to construct string keys for each neighbor â€” this is repeated for every row even though the same cell has the same neighbors in every year.
2. **Indexes into a named character vector** (`idx_lookup[neighbor_keys]`) â€” named-vector lookup by string is O(n) per lookup in the worst case and involves repeated hashing.

The fundamental insight is that the **neighbor topology is year-invariant**: cell A's neighbors are the same in 1992 as in 2019. Yet the current code reconstructs neighbor row-indices for each of the ~6.46M cell-year rows, repeating identical spatial lookups 28 times per cell. That means ~6.46M string-paste + hash-lookup operations when only ~344K unique spatial lookups are needed.

### Broader Algorithmic Issue

Beyond `build_neighbor_lookup`, the result (`neighbor_lookup`) is a list of 6.46M integer vectors. `compute_neighbor_stats` then loops over that same 6.46M-element list **once per variable** (Ã—5 variables). That's ~32.3M list-element accesses.

**Summary of waste:**
| Step | Current Cost | Optimizable? |
|------|-------------|-------------|
| String key construction | 6.46M Ã— avg_neighbors | Yes â€” eliminate entirely |
| Named vector lookup | 6.46M Ã— avg_neighbors | Yes â€” eliminate entirely |
| Neighbor lookup built per cell-year | 6.46M iterations | Yes â€” reduce to 344K Ã— 28 via vectorization |
| `compute_neighbor_stats` per variable | 6.46M Ã— 5 | Yes â€” vectorize with matrix ops |

## Optimization Strategy

1. **Separate the spatial and temporal dimensions.** Build a spatial-only neighbor map (344K cells), then expand to cell-years via vectorized integer arithmetic â€” no strings at all.

2. **Use matrix indexing instead of named-vector lookup.** Create a direct integer mapping from `(cell_index, year_index)` â†’ row number. Neighbor row-indices become a single vectorized matrix subscript operation.

3. **Vectorize `compute_neighbor_stats`** using a sparse-matrix multiply or a pre-built neighbor-index matrix with `rowMeans`/`pmin`/`pmax` applied column-wise, or at minimum use `vapply` instead of `lapply` + `do.call(rbind, ...)`.

4. **Compute all 5 variables' stats in one pass** over the neighbor structure rather than 5 separate passes.

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Drop-in replacement â€” preserves original numerical output exactly.
# =============================================================================

library(data.table)

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {

  # --------------------------------------------------------------------------
  # 1. Convert to data.table for fast column operations (non-destructive)
  # --------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Ensure deterministic ordering: we need a known (id, year) layout.
  # We will create an integer cell-index and year-index.
  dt[, orig_row := .I]  # preserve original row order for final output

  # --- Unique cells and years -----------------------------------------------
  unique_ids   <- id_order                          # 344,208 cells
  n_cells      <- length(unique_ids)
  unique_years <- sort(unique(dt$year))             # 28 years
  n_years      <- length(unique_years)

  # Integer indices for every cell and year
  id_int   <- match(dt$id, unique_ids)
  year_int <- match(dt$year, unique_years)

  # --------------------------------------------------------------------------
  # 2. Build a direct (cell_int, year_int) -> row_number matrix
  #    This replaces ALL string-key hashing.
  # --------------------------------------------------------------------------
  # row_matrix[c, y] = row in dt where id_int==c and year_int==y (or NA)
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(id_int, year_int)] <- seq_len(nrow(dt))

  # --------------------------------------------------------------------------
  # 3. Build spatial-only neighbor index list (344K entries, not 6.46M)
  #    rook_neighbors_unique is an nb object indexed by id_order position.
  # --------------------------------------------------------------------------
  # nb objects store integer indices into the original spatial data, with 0
  # meaning "no neighbors". We just need to ensure alignment with id_order.
  # spdep::nb objects are already a list of integer vectors aligned to the
  # spatial object that produced them, which here is id_order.

  # Validate: length must equal n_cells

  stopifnot(length(rook_neighbors_unique) == n_cells)

  # Clean out any 0-only entries (spdep convention for islands)
  spatial_neighbors <- lapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb > 0L]
    if (length(nb) == 0L) integer(0) else as.integer(nb)
  })

  # --------------------------------------------------------------------------
  # 4. For each cell, find all (neighbor, year) row indices at once
  #    Strategy: for each cell c with k neighbors, we need k Ã— n_years_present
  #    row lookups. We vectorise over cells, not cell-years.
  # --------------------------------------------------------------------------

  # Pre-extract variable columns as a numeric matrix for fast subsetting
  var_mat <- as.matrix(dt[, ..neighbor_source_vars])
  # var_mat[row, var] â€” rows align with dt

  # --------------------------------------------------------------------------
  # 5. Compute stats â€” we build result columns directly
  # --------------------------------------------------------------------------
  # Output columns: for each var, three stats: max, min, mean
  n_out_cols <- length(neighbor_source_vars) * 3
  out_names <- character(n_out_cols)
  k <- 0
  for (v in neighbor_source_vars) {
    out_names[k + 1] <- paste0("neighbor_", v, "_max")
    out_names[k + 2] <- paste0("neighbor_", v, "_min")
    out_names[k + 3] <- paste0("neighbor_", v, "_mean")
    k <- k + 3
  }

  # Pre-allocate output matrix: nrow(dt) Ã— n_out_cols
  out_mat <- matrix(NA_real_, nrow = nrow(dt), ncol = n_out_cols)

  # --------------------------------------------------------------------------
  # Main loop: iterate over CELLS (344K), not cell-years (6.46M)
  # For each cell, vectorize across all years and all neighbors at once.
  # --------------------------------------------------------------------------

  cat("Computing neighbor features for", n_cells, "cells ...\n")
  pct_step <- max(1L, n_cells %/% 20L)

  for (c_idx in seq_len(n_cells)) {

    if (c_idx %% pct_step == 0L)
      cat(sprintf("  %d / %d (%.0f%%)\n", c_idx, n_cells,
                  100 * c_idx / n_cells))

    nb_cells <- spatial_neighbors[[c_idx]]   # neighbor cell-indices (spatial)
    k_nb     <- length(nb_cells)

    # Row indices in dt for this focal cell across all years
    focal_rows <- row_matrix[c_idx, ]          # length n_years, may have NAs
    present    <- which(!is.na(focal_rows))    # which years are present
    if (length(present) == 0L) next

    if (k_nb == 0L) {
      # No neighbors â†’ all stats remain NA (already set)
      next
    }

    # For each present year, gather neighbor rows
    # nb_row_mat[neighbor, year_slot] â€” row indices into dt
    nb_row_mat <- row_matrix[nb_cells, present, drop = FALSE]
    # Dimensions: k_nb Ã— length(present)

    # For each variable compute stats across the k_nb neighbors per year
    col_offset <- 0L
    for (v_idx in seq_along(neighbor_source_vars)) {

      # Extract neighbor values: matrix k_nb Ã— length(present)
      # Using matrix subscript indexing on var_mat
      nb_vals <- matrix(var_mat[nb_row_mat, v_idx],
                        nrow = k_nb, ncol = length(present))
      # Where nb_row_mat is NA (missing cell-year), var_mat[NA, ] gives NA â€” correct.

      # Compute column-wise (per-year) stats, ignoring NAs
      # colMeans, colMins, colMaxs  â€” use matrixStats if available, else base

      # base R approach (robust):
      col_max  <- apply(nb_vals, 2, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else max(x)
      })
      col_min  <- apply(nb_vals, 2, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else min(x)
      })
      col_mean <- apply(nb_vals, 2, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0L) NA_real_ else mean(x)
      })

      target_rows <- focal_rows[present]
      out_mat[target_rows, col_offset + 1L] <- col_max
      out_mat[target_rows, col_offset + 2L] <- col_min
      out_mat[target_rows, col_offset + 3L] <- col_mean
      col_offset <- col_offset + 3L
    }
  }

  cat("Done.\n")

  # --------------------------------------------------------------------------
  # 6. Attach results back to the original data.frame / data.table
  # --------------------------------------------------------------------------
  out_df <- as.data.frame(out_mat)
  names(out_df) <- out_names

  # Bind by original row order
  result <- cbind(cell_data, out_df)
  return(result)
}


# =============================================================================
# EVEN FASTER: matrixStats + vectorized version (recommended)
# =============================================================================

build_neighbor_features_fastest <- function(cell_data,
                                            id_order,
                                            rook_neighbors_unique,
                                            neighbor_source_vars) {

  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    message("Installing matrixStats for 10-50x faster column statistics...")
    install.packages("matrixStats")
  }
  library(matrixStats)
  library(data.table)

  dt <- as.data.table(cell_data)
  dt[, orig_row := .I]

  unique_ids   <- id_order
  n_cells      <- length(unique_ids)
  unique_years <- sort(unique(dt$year))
  n_years      <- length(unique_years)

  id_int   <- match(dt$id, unique_ids)
  year_int <- match(dt$year, unique_years)

  # (cell, year) -> row mapping matrix
  row_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_matrix[cbind(id_int, year_int)] <- seq_len(nrow(dt))

  # Clean spatial neighbors
  spatial_neighbors <- lapply(rook_neighbors_unique, function(nb) {
    nb <- nb[nb > 0L]; if (length(nb) == 0L) integer(0) else as.integer(nb)
  })

  # Variable matrix
  var_mat <- as.matrix(dt[, ..neighbor_source_vars])

  # Output columns
  n_vars     <- length(neighbor_source_vars)
  n_out_cols <- n_vars * 3L
  out_names  <- character(n_out_cols)
  k <- 0L
  for (v in neighbor_source_vars) {
    out_names[k + 1L] <- paste0("neighbor_", v, "_max")
    out_names[k + 2L] <- paste0("neighbor_", v, "_min")
    out_names[k + 3L] <- paste0("neighbor_", v, "_mean")
    k <- k + 3L
  }

  out_mat <- matrix(NA_real_, nrow = nrow(dt), ncol = n_out_cols)

  cat("Computing neighbor features for", n_cells, "cells ...\n")
  flush.console()
  t0 <- proc.time()
  pct_step <- max(1L, n_cells %/% 20L)

  for (c_idx in seq_len(n_cells)) {

    if (c_idx %% pct_step == 0L) {
      elapsed <- (proc.time() - t0)[3]
      rate    <- c_idx / elapsed
      eta     <- (n_cells - c_idx) / rate
      cat(sprintf("  %d / %d (%.0f%%) â€” %.0f cells/s â€” ETA %.1f min\n",
                  c_idx, n_cells, 100 * c_idx / n_cells, rate, eta / 60))
      flush.console()
    }

    nb_cells <- spatial_neighbors[[c_idx]]
    k_nb     <- length(nb_cells)

    focal_rows <- row_matrix[c_idx, ]
    present    <- which(!is.na(focal_rows))
    n_present  <- length(present)
    if (n_present == 0L || k_nb == 0L) next

    # Gather all neighbor rows for all present years: k_nb Ã— n_present matrix
    nb_row_sub <- row_matrix[nb_cells, present, drop = FALSE]

    # For ALL variables at once, extract neighbor value matrices
    # nb_row_sub is k_nb Ã— n_present; var_mat is N Ã— n_vars
    # We want, for each variable, a k_nb Ã— n_present matrix of values.

    target_rows <- focal_rows[present]  # which rows in dt to write to

    for (v_idx in seq_len(n_vars)) {
      # Build neighbor-value matrix
      nb_vals <- matrix(var_mat[nb_row_sub, v_idx],
                        nrow = k_nb, ncol = n_present)

      col_offset <- (v_idx - 1L) * 3L

      if (k_nb == 1L) {
        # Single neighbor: stats are trivial
        vals_vec <- nb_vals[1L, ]
        out_mat[target_rows, col_offset + 1L] <- vals_vec
        out_mat[target_rows, col_offset + 2L] <- vals_vec
        out_mat[target_rows, col_offset + 3L] <- vals_vec
      } else {
        # matrixStats operates column-wise
        out_mat[target_rows, col_offset + 1L] <- colMaxs(nb_vals,  na.rm = TRUE)
        out_mat[target_rows, col_offset + 2L] <- colMins(nb_vals,  na.rm = TRUE)
        out_mat[target_rows, col_offset + 3L] <- colMeans2(nb_vals, na.rm = TRUE)
      }
    }
  }

  elapsed_total <- (proc.time() - t0)[3]
  cat(sprintf("Done in %.1f minutes.\n", elapsed_total / 60))

  # Fix -Inf/Inf from colMaxs/colMins on all-NA columns
  out_mat[is.infinite(out_mat)] <- NA_real_

  out_df <- as.data.frame(out_mat)
  names(out_df) <- out_names
  result <- cbind(cell_data, out_df)
  return(result)
}


# =============================================================================
# USAGE â€” Drop-in replacement for the original outer loop
# =============================================================================
#
# # Original code (86+ hours):
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
#
# # Optimized replacement (estimated 15-45 minutes):
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_fastest(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # The Random Forest model is untouched â€” only feature columns are recomputed
# # with numerically identical values.
```

## Performance Analysis

| Aspect | Original | Optimized |
|--------|----------|-----------|
| **Outer loop iterations** | 6.46M (cell-years) | 344K (cells) |
| **String `paste()` calls** | ~6.46M Ã— avg 4 neighbors = ~25.8M | **0** |
| **Named-vector hash lookups** | ~25.8M | **0** |
| **Lookup mechanism** | String-keyed named vector | Integer matrix subscript `row_matrix[i, j]` |
| **Stats computation** | `lapply` + `do.call(rbind)` over 6.46M, Ã—5 vars | `matrixStats::colMaxs/colMins/colMeans2` vectorized per cell, all years at once |
| **Passes over neighbor structure** | 1 build + 5 stat passes = 6 | **1 single pass** |
| **Estimated runtime** | 86+ hours | **15â€“45 minutes** |
| **Memory** | 6.46M-element list of integer vectors | 344K Ã— 28 integer matrix (~39 MB) |

### Why This is Correct

- **Same estimand**: For each cell-year row, the neighbor set is determined by the spatial rook contiguity (year-invariant) intersected with which neighbors exist in that year. The `row_matrix[nb_cells, year]` lookup produces `NA` for missing cell-years, which are then excluded by `na.rm = TRUE` â€” exactly matching the original `result[!is.na(result)]` filter.
- **No model retraining needed**: Only the feature-construction pipeline is replaced. Output column names and semantics are preserved identically.
END_RESPONSE>>>
