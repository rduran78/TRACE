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
run_id: Anthropic_F12_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single vectorized concatenation step per variable â€” only 5 calls total. Each inner function returns a fixed-length-3 numeric vector, so there is no "repeated list binding" inside `compute_neighbor_stats()`; it's a straightforward `lapply` returning pre-allocated vectors.

**The true, deeper bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` and named-vector lookup over 6.46M rows:** `idx_lookup` is a named integer vector of length ~6.46M. For every single row `i`, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` to build string keys, then does `idx_lookup[neighbor_keys]` â€” a **named character-vector lookup** (hash-table probe) for every neighbor of every row. With ~6.46M rows and an average of ~4 rook neighbors per cell, that's ~25.8 million string constructions and hash lookups, each against a 6.46M-entry named vector.

2. **`as.character()` and `id_to_ref[]` per row:** Each iteration converts `data$id[i]` to character and indexes into `id_to_ref`. That's 6.46M individual scalar character conversions and lookups.

3. **`lapply` over 6.46M scalar iterations in R:** The entire function is a row-by-row R-level loop (`lapply` over `row_ids` of length 6.46M). Each iteration performs string operations, subsetting, and NA filtering. This is the classic "R loop over millions of rows" anti-pattern.

4. **Redundant recomputation across years:** The neighbor *structure* is purely spatial â€” cell A's neighbors are the same cells in every year. Yet `build_neighbor_lookup()` recomputes the neighbor row indices separately for each of the 6.46M cell-year rows, instead of computing the spatial neighbor mapping once (344,208 cells) and then expanding across 28 years via vectorized indexing.

**Estimated cost breakdown (approximate):**

| Component | Iterations | Cost |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M Ã— ~4 neighbors = ~25.8M string ops + hash lookups | **~80â€“85+ hours** |
| `compute_neighbor_stats` (5 vars) | 5 Ã— 6.46M simple numeric ops | **~0.5â€“1 hour** |
| `do.call(rbind, ...)` | 5 calls binding 6.46M Ã— 3 matrices | **~minutes** |

The bottleneck is overwhelmingly `build_neighbor_lookup()`.

---

## Optimization Strategy

1. **Separate spatial structure from panel expansion.** Compute the neighbor mapping once over 344,208 unique cells (not 6.46M cell-years). The `rook_neighbors_unique` nb object already provides this.

2. **Use integer arithmetic instead of string key construction.** Map each `(cell_index, year_index)` pair to a row number via simple arithmetic: `row = (cell_index - 1) * n_years + year_index`. This eliminates all `paste()`, `as.character()`, and named-vector hash lookups.

3. **Vectorize `compute_neighbor_stats` using matrix indexing.** Instead of `lapply` over 6.46M elements, pre-build a fixed-width neighbor matrix (padded with NA), extract all neighbor values at once via matrix subsetting, and compute `max/min/mean` with `rowMaxs/rowMins/rowMeans` from the `matrixStats` package (or base R `apply`).

4. **Preserve the trained Random Forest model and original numerical estimand.** The output columns are identical in name, order, and numerical value. Only the computation path changes.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# =============================================================================
# Prerequisites:
#   cell_data        : data.frame with columns id, year, and the source vars
#   id_order         : vector of unique cell IDs in the order matching rook_neighbors_unique
#   rook_neighbors_unique : spdep nb object (list of integer neighbor indices), length = n_cells
#   neighbor_source_vars  : c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# Assumptions validated from the problem statement:
#   - cell_data is sorted by (id, year) or can be sorted
#   - years are contiguous 1992â€“2019 (28 years)
# =============================================================================

library(data.table)

optimized_neighbor_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                        neighbor_source_vars) {

  # --- Step 0: Convert to data.table for speed; record original order --------
  dt <- as.data.table(cell_data)
  dt[, .orig_row := .I]

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  # --- Step 1: Create integer mappings (no strings) --------------------------
  # Map cell id -> spatial index (1..n_cells)
  id_to_sidx <- setNames(seq_along(id_order), as.character(id_order))

  # Map year -> year index (1..n_years)
  year_to_yidx <- setNames(seq_along(years), as.character(years))

  # Sort data by (spatial_index, year_index) so row = (sidx-1)*n_years + yidx
  dt[, sidx := id_to_sidx[as.character(id)]]
  dt[, yidx := year_to_yidx[as.character(year)]]
  setorder(dt, sidx, yidx)
  dt[, .sorted_row := .I]

  # Verify contiguous panel: each cell must have exactly n_years rows
  # (If not perfectly balanced, we handle it below with a safe lookup)
  cell_counts <- dt[, .N, by = sidx]
  is_balanced <- all(cell_counts$N == n_years)

  if (is_balanced) {
    # Fast path: row = (sidx - 1) * n_years + yidx
    # Build neighbor row-index matrix: each row i in dt gets its neighbor rows
    # Spatial neighbors for each cell (max degree needed for padding)
    n_neighbors_per_cell <- lengths(rook_neighbors_unique)
    max_k <- max(n_neighbors_per_cell)

    # Build spatial neighbor matrix: n_cells x max_k, padded with NA
    spatial_nb_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
    for (s in seq_len(n_cells)) {
      nb <- rook_neighbors_unique[[s]]
      if (length(nb) > 0 && !(length(nb) == 1 && nb[1] == 0L)) {
        spatial_nb_mat[s, seq_along(nb)] <- nb
      }
    }

    # For each row in dt, expand spatial neighbors to row indices
    # dt$.sorted_row[i] corresponds to sidx[i], yidx[i]
    # neighbor rows = (spatial_nb_mat[sidx[i], ] - 1) * n_years + yidx[i]
    all_sidx <- dt$sidx
    all_yidx <- dt$yidx
    n_rows   <- nrow(dt)

    # Build full neighbor-row matrix: n_rows x max_k
    # Vectorized: for each column k of spatial_nb_mat, compute all row indices at once
    neighbor_row_mat <- matrix(NA_integer_, nrow = n_rows, ncol = max_k)
    for (k in seq_len(max_k)) {
      nb_sidx_k <- spatial_nb_mat[all_sidx, k]  # spatial neighbor index for column k
      valid <- !is.na(nb_sidx_k)
      neighbor_row_mat[valid, k] <- (nb_sidx_k[valid] - 1L) * n_years + all_yidx[valid]
    }

    # --- Step 2: Compute neighbor stats per variable (fully vectorized) ------
    for (var_name in neighbor_source_vars) {
      vals <- dt[[var_name]]

      # Extract neighbor values into a matrix n_rows x max_k
      nb_vals <- matrix(vals[neighbor_row_mat], nrow = n_rows, ncol = max_k)
      # NAs from padding or missing data are already NA

      # Compute row-wise max, min, mean ignoring NAs
      # Using matrixStats if available, otherwise base R
      if (requireNamespace("matrixStats", quietly = TRUE)) {
        nb_max  <- matrixStats::rowMaxs(nb_vals,  na.rm = TRUE)
        nb_min  <- matrixStats::rowMins(nb_vals,  na.rm = TRUE)
        nb_mean <- matrixStats::rowMeans2(nb_vals, na.rm = TRUE)
      } else {
        nb_max  <- apply(nb_vals, 1, max,  na.rm = TRUE)
        nb_min  <- apply(nb_vals, 1, min,  na.rm = TRUE)
        nb_mean <- apply(nb_vals, 1, mean, na.rm = TRUE)
      }

      # Handle rows where ALL neighbors are NA (matrixStats returns -Inf/Inf/NaN)
      all_na <- rowSums(!is.na(nb_vals)) == 0L
      nb_max[all_na]  <- NA_real_
      nb_min[all_na]  <- NA_real_
      nb_mean[all_na] <- NA_real_

      # Also fix Inf/-Inf from matrixStats when all are NA (redundant but safe)
      nb_max[is.infinite(nb_max)]   <- NA_real_
      nb_min[is.infinite(nb_min)]   <- NA_real_

      # Assign columns with same naming convention as original pipeline
      max_col  <- paste0("neighbor_max_",  var_name)
      min_col  <- paste0("neighbor_min_",  var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      dt[, (max_col)  := nb_max]
      dt[, (min_col)  := nb_min]
      dt[, (mean_col) := nb_mean]
    }

  } else {
    # --- Unbalanced panel fallback (still much faster than original) ----------
    # Build a row lookup: key = sidx * (max_year+1) + yidx for O(1) integer lookup
    row_lookup <- integer(n_cells * n_years)  # pre-allocate; 0 = missing
    row_lookup[(dt$sidx - 1L) * n_years + dt$yidx] <- dt$.sorted_row

    n_neighbors_per_cell <- lengths(rook_neighbors_unique)
    max_k <- max(n_neighbors_per_cell)

    spatial_nb_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
    for (s in seq_len(n_cells)) {
      nb <- rook_neighbors_unique[[s]]
      if (length(nb) > 0 && !(length(nb) == 1 && nb[1] == 0L)) {
        spatial_nb_mat[s, seq_along(nb)] <- nb
      }
    }

    all_sidx <- dt$sidx
    all_yidx <- dt$yidx
    n_rows   <- nrow(dt)

    neighbor_row_mat <- matrix(NA_integer_, nrow = n_rows, ncol = max_k)
    for (k in seq_len(max_k)) {
      nb_sidx_k <- spatial_nb_mat[all_sidx, k]
      valid <- !is.na(nb_sidx_k)
      lookup_keys <- (nb_sidx_k[valid] - 1L) * n_years + all_yidx[valid]
      looked_up <- row_lookup[lookup_keys]
      looked_up[looked_up == 0L] <- NA_integer_
      neighbor_row_mat[valid, k] <- looked_up
    }

    for (var_name in neighbor_source_vars) {
      vals <- dt[[var_name]]
      nb_vals <- matrix(vals[neighbor_row_mat], nrow = n_rows, ncol = max_k)

      if (requireNamespace("matrixStats", quietly = TRUE)) {
        nb_max  <- matrixStats::rowMaxs(nb_vals,  na.rm = TRUE)
        nb_min  <- matrixStats::rowMins(nb_vals,  na.rm = TRUE)
        nb_mean <- matrixStats::rowMeans2(nb_vals, na.rm = TRUE)
      } else {
        nb_max  <- apply(nb_vals, 1, max,  na.rm = TRUE)
        nb_min  <- apply(nb_vals, 1, min,  na.rm = TRUE)
        nb_mean <- apply(nb_vals, 1, mean, na.rm = TRUE)
      }

      all_na <- rowSums(!is.na(nb_vals)) == 0L
      nb_max[all_na]  <- NA_real_
      nb_min[all_na]  <- NA_real_
      nb_mean[all_na] <- NA_real_
      nb_max[is.infinite(nb_max)]   <- NA_real_
      nb_min[is.infinite(nb_min)]   <- NA_real_

      max_col  <- paste0("neighbor_max_",  var_name)
      min_col  <- paste0("neighbor_min_",  var_name)
      mean_col <- paste0("neighbor_mean_", var_name)

      dt[, (max_col)  := nb_max]
      dt[, (min_col)  := nb_min]
      dt[, (mean_col) := nb_mean]
    }
  }

  # --- Step 3: Restore original row order and return as data.frame -----------
  setorder(dt, .orig_row)
  dt[, c(".orig_row", ".sorted_row", "sidx", "yidx") := NULL]

  return(as.data.frame(dt))
}

# =============================================================================
# USAGE â€” drop-in replacement for the original pipeline
# =============================================================================
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- optimized_neighbor_pipeline(
#   cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
# )
#
# # The trained Random Forest model is untouched â€” use it directly for prediction:
# # predictions <- predict(rf_model, newdata = cell_data)
# =============================================================================
```

---

## Summary of Why This Works

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup`: 6.46M R-loop iterations, each doing `paste()` + string hash lookup against a 6.46M-entry named vector | Eliminated entirely |
| **Key construction** | `paste(id, year)` string keys â€” ~25.8M string allocations | Integer arithmetic: `(sidx-1)*n_years + yidx` â€” zero string allocation |
| **Neighbor expansion** | Per-row `lapply` in R (6.46M iterations) | Vectorized column-wise matrix fill (`max_k` iterations, typically 4) |
| **Stats computation** | `lapply` over 6.46M elements per variable | `matrixStats::rowMaxs/rowMins/rowMeans2` â€” single vectorized C call per stat |
| **`do.call(rbind,...)`** | Binds 6.46M 3-element vectors (colleague's claimed bottleneck) | Eliminated; stats written directly as columns |
| **Estimated runtime** | 86+ hours | **~2â€“10 minutes** (depending on RAM/disk pressure) |
| **Numerical output** | max, min, mean of non-NA neighbor values | Identical â€” same values, same column names |
| **RF model** | Untouched | Untouched |

The colleague's diagnosis targeted a minor cost (`do.call(rbind, ...)` and the `lapply` in `compute_neighbor_stats`). The true bottleneck is the **6.46-million-iteration R-level loop in `build_neighbor_lookup()`** performing per-row string construction and hash-table lookups, which is eliminated by exploiting the spatial-temporal separability of the panel structure and replacing string keys with integer arithmetic.
END_RESPONSE>>>
