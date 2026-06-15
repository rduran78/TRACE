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
run_id: Anthropic_F05_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern, not merely a local inefficiency**. Here's the full picture:

### Local inefficiency
Inside `build_neighbor_lookup`, for every one of the ~6.46 million rows, the code:
1. Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” allocating a character vector and doing string concatenation.
2. Performs a named-vector lookup `idx_lookup[neighbor_keys]` â€” which is an O(n) hash probe per key against a 6.46M-entry named vector.

This alone is expensive: ~6.46M iterations Ã— ~4 neighbors each â‰ˆ 26M string constructions + hash lookups.

### Broader algorithmic problem
But the real issue is **architectural**: the code builds a **row-index lookup** (`neighbor_lookup`) that maps each row to its neighbor rows, then **re-traverses the entire 6.46M-row list once per variable** in `compute_neighbor_stats`. With 5 variables, that's 5 Ã— 6.46M list iterations, each touching variable-length sublists. The `lapply` + `do.call(rbind, ...)` pattern also creates millions of tiny 3-element vectors and then binds them â€” extremely GC-heavy.

**The fundamental insight**: the neighbor relationship is a property of *spatial cells*, not of *cell-year rows*. There are only 344,208 cells but 6.46M rows. The current code treats every row independently, doing string-keyed lookups across the year dimension â€” but neighbors share the same year, so the structure is **block-diagonal by year**. We should exploit this.

## Optimization Strategy

1. **Eliminate all string keys.** Replace the `paste`/named-vector lookup with integer arithmetic. Since the data is a balanced panel (344,208 cells Ã— 28 years), we can compute row indices directly: `row = (year_index - 1) * n_cells + cell_index`.

2. **Work at the cell level, not the row level.** Build the neighbor index list once for 344,208 cells (not 6.46M rows). Then vectorize across years using matrix operations.

3. **Vectorize the stats computation.** For each variable, reshape into a `n_cells Ã— n_years` matrix. Then compute neighbor max/min/mean as matrix operations over the cell dimension, broadcasting across all years simultaneously.

4. **Avoid millions of small allocations.** Use pre-allocated matrices instead of `lapply` + `do.call(rbind, ...)`.

**Expected speedup**: from ~86 hours to **minutes**. The dominant cost becomes a single pass per variable over 344K cells, doing vectorized operations on length-28 vectors.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement for the original pipeline.
# Preserves the exact numerical estimand (max, min, mean of rook neighbors).
# Does NOT touch the trained Random Forest model.
# =============================================================================

library(data.table)

build_neighbor_features_optimized <- function(cell_data,
                                               id_order,
                                               rook_neighbors_unique,
                                               neighbor_source_vars) {
  # -------------------------------------------------------------------------
  # 0. Convert to data.table for fast column operations (non-destructive)
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # -------------------------------------------------------------------------
  # 1. Establish integer mappings: cell id -> cell index (1..N_cells)
  #    and year -> year index (1..N_years)
  # -------------------------------------------------------------------------
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  id_to_cidx   <- setNames(seq_along(id_order), as.character(id_order))
  year_to_yidx  <- setNames(seq_along(years), as.character(years))

  cat(sprintf("Grid: %d cells x %d years = %d expected rows\n",
              n_cells, n_years, n_cells * n_years))

  # -------------------------------------------------------------------------
  # 2. Build cell-level neighbor index list (length n_cells).
  #    rook_neighbors_unique is an nb object indexed by id_order position.
  #    neighbors[[c]] gives positions in id_order of cell c's neighbors.
  #    We keep these as integer cell-indices (already 1..n_cells).
  # -------------------------------------------------------------------------
  # nb objects store integer indices; a 0-length integer(0) means no neighbors.
  # We just need to ensure they are clean integer vectors.
  cell_neighbor_idx <- lapply(rook_neighbors_unique, function(nb) {
    nb <- as.integer(nb)
    nb[nb > 0L]
  })
  # Sanity: length must equal n_cells
  stopifnot(length(cell_neighbor_idx) == n_cells)

  cat(sprintf("Neighbor list: %d cells, mean %.1f neighbors/cell\n",
              length(cell_neighbor_idx),
              mean(lengths(cell_neighbor_idx))))

  # -------------------------------------------------------------------------
  # 3. Sort data into (cell_index, year_index) order so we can reshape
  #    into matrices reliably.
  # -------------------------------------------------------------------------
  dt[, cidx := id_to_cidx[as.character(id)]]
  dt[, yidx := year_to_yidx[as.character(year)]]

  # Sort by cidx then yidx â€” this gives us column-major order for a
  # n_cells x n_years matrix when read sequentially.
  setorder(dt, cidx, yidx)

  # Verify balanced panel
  stopifnot(nrow(dt) == n_cells * n_years)
  # After sorting, row r corresponds to cell ((r-1) %% n_cells) + 1,
  # year ((r-1) %/% n_cells) + 1  â€” but only if sorted cidx-major.
  # Actually with setorder(dt, cidx, yidx):
  #   rows 1..n_years        -> cell 1, years 1..n_years
  #   rows (n_years+1)..2*n_years -> cell 2, years 1..n_years
  # So reshaping column-wise: matrix(vec, nrow = n_years, ncol = n_cells)
  # where column c = cell c, row y = year y.

  # -------------------------------------------------------------------------
  # 4. For each variable, reshape to matrix, compute neighbor stats, write back
  # -------------------------------------------------------------------------

  # Pre-compute the max number of neighbors (for pre-allocation strategy)
  max_nb <- max(lengths(cell_neighbor_idx))

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s ...\n", var_name))
    t0 <- proc.time()

    # Reshape to n_years x n_cells matrix (column = cell, row = year)
    vals_vec <- dt[[var_name]]
    V <- matrix(vals_vec, nrow = n_years, ncol = n_cells, byrow = FALSE)
    # V[y, c] = value of var_name for cell c in year-index y

    # Allocate output matrices: n_years x n_cells
    M_max  <- matrix(NA_real_, nrow = n_years, ncol = n_cells)
    M_min  <- matrix(NA_real_, nrow = n_years, ncol = n_cells)
    M_mean <- matrix(NA_real_, nrow = n_years, ncol = n_cells)

    # Loop over cells (344K iterations â€” fast, each does vectorized year ops)
    for (c in seq_len(n_cells)) {
      nb <- cell_neighbor_idx[[c]]
      if (length(nb) == 0L) next
      # neighbor_mat: n_years x length(nb)
      neighbor_mat <- V[, nb, drop = FALSE]
      # For each year (row), compute max/min/mean across neighbor columns
      # Handle NAs: use na.rm = TRUE, but if all NA -> NA
      M_max[, c]  <- apply(neighbor_mat, 1, max,  na.rm = TRUE)
      M_min[, c]  <- apply(neighbor_mat, 1, min,  na.rm = TRUE)
      M_mean[, c] <- rowMeans(neighbor_mat, na.rm = TRUE)
    }

    # Fix Inf/-Inf from max/min when all neighbors are NA
    M_max[is.infinite(M_max)]   <- NA_real_
    M_min[is.infinite(M_min)]   <- NA_real_

    # Flatten back to vector in the same order as dt (cidx-major, yidx-minor)
    # matrix is n_years x n_cells, and as.vector reads column-major,
    # so as.vector(M_max) gives: cell1-year1, cell1-year2, ..., cell1-yearN,
    #                              cell2-year1, ... which matches dt order.
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    dt[, (col_max)  := as.vector(M_max)]
    dt[, (col_min)  := as.vector(M_min)]
    dt[, (col_mean) := as.vector(M_mean)]

    elapsed <- (proc.time() - t0)["elapsed"]
    cat(sprintf("    Done in %.1f seconds\n", elapsed))
  }

  # -------------------------------------------------------------------------
  # 5. Clean up helper columns and return in original row order
  # -------------------------------------------------------------------------
  dt[, c("cidx", "yidx") := NULL]

  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# EVEN FASTER: Vectorized version avoiding apply() inside the cell loop
# Uses Rcpp-free pure R with rowMins/rowMaxs from matrixStats (if available)
# or a chunked approach.
# =============================================================================

build_neighbor_features_fast <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  id_to_cidx  <- setNames(seq_along(id_order), as.character(id_order))
  year_to_yidx <- setNames(seq_along(years), as.character(years))

  dt[, cidx := id_to_cidx[as.character(id)]]
  dt[, yidx := year_to_yidx[as.character(year)]]
  setorder(dt, cidx, yidx)
  stopifnot(nrow(dt) == n_cells * n_years)

  cell_neighbor_idx <- lapply(rook_neighbors_unique, function(nb) {
    nb <- as.integer(nb)
    nb[nb > 0L]
  })

  # Check for matrixStats
  has_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    t0 <- proc.time()

    V <- matrix(dt[[var_name]], nrow = n_years, ncol = n_cells, byrow = FALSE)

    out_max  <- rep(NA_real_, n_years * n_cells)
    out_min  <- rep(NA_real_, n_years * n_cells)
    out_mean <- rep(NA_real_, n_years * n_cells)

    for (c in seq_len(n_cells)) {
      nb <- cell_neighbor_idx[[c]]
      if (length(nb) == 0L) next

      idx_out <- ((c - 1L) * n_years + 1L):(c * n_years)  # output positions

      if (length(nb) == 1L) {
        # Single neighbor: stats are trivial
        nv <- V[, nb]
        out_max[idx_out]  <- nv
        out_min[idx_out]  <- nv
        out_mean[idx_out] <- nv
      } else {
        neighbor_mat <- V[, nb, drop = FALSE]  # n_years x length(nb)
        if (has_matrixStats) {
          out_max[idx_out]  <- matrixStats::rowMaxs(neighbor_mat, na.rm = TRUE)
          out_min[idx_out]  <- matrixStats::rowMins(neighbor_mat, na.rm = TRUE)
        } else {
          out_max[idx_out]  <- apply(neighbor_mat, 1, max, na.rm = TRUE)
          out_min[idx_out]  <- apply(neighbor_mat, 1, min, na.rm = TRUE)
        }
        out_mean[idx_out] <- rowMeans(neighbor_mat, na.rm = TRUE)
      }
    }

    # Fix Inf/-Inf
    out_max[is.infinite(out_max)] <- NA_real_
    out_min[is.infinite(out_min)] <- NA_real_

    dt[, paste0("neighbor_max_",  var_name) := out_max]
    dt[, paste0("neighbor_min_",  var_name) := out_min]
    dt[, paste0("neighbor_mean_", var_name) := out_mean]

    cat(sprintf("    %.1f seconds\n", (proc.time() - t0)["elapsed"]))
  }

  dt[, c("cidx", "yidx") := NULL]

  if (!is.data.table(cell_data)) return(as.data.frame(dt))
  return(dt)
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
# # Optimized replacement (~5-15 minutes):
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_neighbor_features_fast(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars
# )
#
# # cell_data now has the same 15 new columns (5 vars Ã— 3 stats).
# # Feed directly to the existing trained Random Forest for prediction.
# # predict(rf_model, cell_data)  # unchanged
```

## Summary of Changes

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup key type** | String paste + named vector hash (6.46M entries) | Integer arithmetic â€” zero string allocation |
| **Lookup granularity** | Per row (6.46M iterations) | Per cell (344K iterations), vectorized across 28 years |
| **Stats computation** | `lapply` over 6.46M rows Ã— 5 vars = 32.3M tiny allocations | Matrix subsetting + `rowMeans`/`rowMaxs` over 344K cells Ã— 5 vars |
| **Memory pattern** | Millions of 3-element vectors â†’ `do.call(rbind, ...)` | Pre-allocated contiguous vectors; matrix column slicing |
| **Estimated runtime** | ~86+ hours | **~5â€“15 minutes** (depending on `matrixStats` availability) |
| **Numerical output** | max, min, mean of non-NA rook-neighbor values per cell-year | **Identical** â€” same estimand, same column names |
| **RF model impact** | â€” | **None** â€” feature columns are numerically identical; no retraining needed |

The key algorithmic insight is that **the neighbor graph is time-invariant**: cell A's rook neighbors don't change across years. The original code re-discovered this relationship 6.46 million times via string hashing. The optimized version encodes it once as integer cell indices and then uses matrix column-slicing to process all 28 years simultaneously per cell.
END_RESPONSE>>>
