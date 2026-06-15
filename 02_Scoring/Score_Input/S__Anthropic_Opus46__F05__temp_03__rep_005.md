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
run_id: Anthropic_F05_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F05_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations Over 6.46M Rows

`build_neighbor_lookup` calls `paste()` and performs named-vector lookups **inside a per-row `lapply`** over ~6.46 million rows. Each iteration:

1. **Constructs string keys** via `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” repeated for every row.
2. **Looks up indices** via `idx_lookup[neighbor_keys]` â€” a named-vector character match, which is O(k) per neighbor set but with high constant factor due to string hashing.

The `idx_lookup` named vector itself (6.46M string keys) is built once, but the **per-row key construction and lookup** is the bottleneck. With ~6.46M rows and an average of ~8 rook neighbors per cell (1,373,394 directed / 344,208 cells â‰ˆ 4 per cell, but rook typically gives up to 4, so ~4 neighbors), that's ~25.8M `paste` + hash-match operations.

### Why It's a Broader Algorithmic Problem

The entire approach conflates **spatial structure** (which is time-invariant) with **temporal indexing** (which is regular). The neighbor relationships are identical across all 28 years. The string-keyed lookup re-discovers, for every row, the same spatial neighbor mapping â€” just offset by the year's block position in the data frame. This means:

- **The neighbor lookup can be computed once per cell (344K cells), not once per cell-year (6.46M rows).**
- **Year-matching is trivially an integer offset** if the data is sorted by `(id, year)` or `(year, id)`.

Similarly, `compute_neighbor_stats` is already vectorized over the lookup list, but the lookup list itself is 6.46M entries long when it could be 344K entries + a year-stride offset.

## Optimization Strategy

1. **Separate spatial structure from temporal indexing.** Compute the neighbor index list once per cell (344K entries), not per cell-year.
2. **Sort data by `(id, year)` and use integer arithmetic** to map from cell-level neighbor indices to cell-year row indices. If data is sorted by id then year, cell `j` in year `t` is at row `(j-1)*T + (t - t_min + 1)` where `T = 28`.
3. **Vectorize `compute_neighbor_stats`** using the integer-offset scheme â€” no strings, no hash lookups.
4. **Process all 5 variables in one pass** over the neighbor structure if desired (optional; the variable loop is cheap once the lookup is fast).

This reduces the core operation from ~6.46M string-key constructions + hash lookups to ~344K integer-vector operations + vectorized matrix indexing.

### Complexity Comparison

| Step | Original | Optimized |
|---|---|---|
| Build lookup | O(NÂ·k) string ops, N=6.46M | O(CÂ·k) integer ops, C=344K |
| Per-variable stats | O(N) list traversal (fine) | O(N) vectorized matrix ops |
| Total string ops | ~25.8M paste + hash | **Zero** |

Expected speedup: **~100â€“500x** for the lookup build; overall pipeline from ~86 hours to **minutes**.

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# Preserves the original numerical estimand (max, min, mean of neighbor values)
# Preserves the trained Random Forest model (no retraining needed)
# =============================================================================

#' Build a cell-level neighbor index (time-invariant).
#'
#' @param id_order Integer vector of unique cell IDs in the order matching
#'   the spdep::nb object (rook_neighbors_unique).
#' @param neighbors An spdep::nb object (list of integer index vectors).
#' @return A list of length C (number of cells). Each element is an integer
#'   vector of 1-based positions into id_order for that cell's neighbors.
#'   Positions correspond to the cell's rank in id_order.
build_cell_neighbor_index <- function(id_order, neighbors) {

  # neighbors[[j]] already contains indices into id_order, so we just need

  # to clean them (remove 0s that spdep uses for "no neighbor").
  C <- length(id_order)
  stopifnot(length(neighbors) == C)

  lapply(seq_len(C), function(j) {
    nb <- neighbors[[j]]
    nb <- nb[nb > 0L]  # spdep encodes no-neighbor as 0
    as.integer(nb)
  })
}

#' Sort data into (id, year) order and build mapping infrastructure.
#'
#' @param data Data frame with columns `id` and `year`.
#' @param id_order Integer vector of unique cell IDs (canonical order).
#' @return A list with:
#'   - data: the sorted data frame
#'   - years: sorted unique years
#'   - T: number of years
#'   - C: number of cells
#'   - year_min: minimum year
#'   - cell_rank: named integer vector mapping id -> rank in id_order
#'   - complete_grid: logical, TRUE if data is a complete C x T panel
sort_and_map <- function(data, id_order) {
  C <- length(id_order)
  years <- sort(unique(data$year))
  T_ <- length(years)
  year_min <- min(years)

  # Create cell rank lookup: id -> position in id_order (1-based)
  cell_rank <- setNames(seq_along(id_order), as.character(id_order))

  # Add helper columns for sorting
  data$.cell_rank <- cell_rank[as.character(data$id)]
  data$.year_offset <- data$year - year_min + 1L

  # Sort by (cell_rank, year_offset) so that row for cell j, year t is at
  # position (j-1)*T + t_offset
  data <- data[order(data$.cell_rank, data$.year_offset), ]
  rownames(data) <- NULL

  # Verify completeness: should have exactly C * T rows
  complete_grid <- (nrow(data) == C * T_)
  if (!complete_grid) {
    warning(
      sprintf(
        "Panel is not complete: expected %d rows (C=%d x T=%d), got %d. ",
        C * T_, C, T_, nrow(data)
      ),
      "Falling back to safe mode with explicit row mapping."
    )
  }

  list(
    data = data,
    years = years,
    T = T_,
    C = C,
    year_min = year_min,
    cell_rank = cell_rank,
    complete_grid = complete_grid
  )
}

#' Compute neighbor stats (max, min, mean) for one variable using integer
#' arithmetic on a sorted complete panel.
#'
#' @param vals Numeric vector of length N (= C * T), in (cell, year) sort order.
#' @param cell_nb List of length C; each element is an integer vector of
#'   neighbor cell ranks (1-based into the C cells).
#' @param C Number of cells.
#' @param T_ Number of years.
#' @return A matrix of dimension N x 3 (max, min, mean).
compute_neighbor_stats_fast <- function(vals, cell_nb, C, T_) {
  N <- C * T_

  # Pre-allocate output
  out_max  <- rep(NA_real_, N)
  out_min  <- rep(NA_real_, N)
  out_mean <- rep(NA_real_, N)

  # For each cell, compute stats across all years at once (vectorized over T)

  for (j in seq_len(C)) {
    nb <- cell_nb[[j]]
    if (length(nb) == 0L) next

    # Row indices for cell j across all years: (j-1)*T + 1:T
    base_j <- (j - 1L) * T_
    rows_j <- base_j + seq_len(T_)

    # Row indices for each neighbor cell across all years
    # This creates a matrix: T_ rows x length(nb) columns
    # Each column is the vals for one neighbor across all T years
    nb_bases <- (nb - 1L) * T_  # base offsets for each neighbor

    # Build index matrix: rows = year offsets (1:T), cols = neighbors
    # idx_mat[t, k] = row index of neighbor k in year t
    idx_mat <- outer(seq_len(T_), nb_bases, FUN = "+")
    # idx_mat is T_ x length(nb)

    # Extract neighbor values
    nb_vals <- matrix(vals[idx_mat], nrow = T_, ncol = length(nb))

    # Compute row-wise (i.e., per-year) stats, handling NAs
    # Use matrixStats if available, otherwise base R
    if (requireNamespace("matrixStats", quietly = TRUE)) {
      nb_max  <- matrixStats::rowMaxs(nb_vals, na.rm = TRUE)
      nb_min  <- matrixStats::rowMins(nb_vals, na.rm = TRUE)
      nb_mean <- matrixStats::rowMeans2(nb_vals, na.rm = TRUE)

      # matrixStats returns -Inf/Inf when all NA; fix to NA
      all_na <- matrixStats::rowAlls(is.na(nb_vals))
      nb_max[all_na]  <- NA_real_
      nb_min[all_na]  <- NA_real_
      nb_mean[all_na] <- NA_real_
    } else {
      nb_max  <- apply(nb_vals, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else max(x)
      })
      nb_min  <- apply(nb_vals, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else min(x)
      })
      nb_mean <- apply(nb_vals, 1, function(x) {
        x <- x[!is.na(x)]; if (length(x) == 0) NA_real_ else mean(x)
      })
    }

    out_max[rows_j]  <- nb_max
    out_min[rows_j]  <- nb_min
    out_mean[rows_j] <- nb_mean
  }

  cbind(out_max, out_min, out_mean)
}

#' Fallback: compute neighbor stats for an incomplete panel.
#' Uses integer row-mapping instead of string keys.
#'
#' @param data Sorted data frame with .cell_rank and .year_offset columns.
#' @param cell_nb Cell-level neighbor list.
#' @param var_name Variable name.
#' @param C Number of cells.
#' @param T_ Number of years.
#' @return Matrix N x 3.
compute_neighbor_stats_safe <- function(data, cell_nb, var_name, C, T_) {
  N <- nrow(data)
  vals <- data[[var_name]]

  # Build an integer lookup matrix: row_lookup[j, t] = row index in data
  # for cell rank j, year offset t. NA if missing.
  row_lookup <- matrix(NA_integer_, nrow = C, ncol = T_)
  for (i in seq_len(N)) {
    row_lookup[data$.cell_rank[i], data$.year_offset[i]] <- i
  }

  out_max  <- rep(NA_real_, N)
  out_min  <- rep(NA_real_, N)
  out_mean <- rep(NA_real_, N)

  for (i in seq_len(N)) {
    j <- data$.cell_rank[i]
    t <- data$.year_offset[i]
    nb <- cell_nb[[j]]
    if (length(nb) == 0L) next

    nb_rows <- row_lookup[cbind(nb, rep(t, length(nb)))]
    nb_rows <- nb_rows[!is.na(nb_rows)]
    if (length(nb_rows) == 0L) next

    nb_vals <- vals[nb_rows]
    nb_vals <- nb_vals[!is.na(nb_vals)]
    if (length(nb_vals) == 0L) next

    out_max[i]  <- max(nb_vals)
    out_min[i]  <- min(nb_vals)
    out_mean[i] <- mean(nb_vals)
  }

  cbind(out_max, out_min, out_mean)
}

#' Add neighbor features for one variable to the data frame.
#' Matches the original column naming convention.
#'
#' @param data Data frame (sorted, with .cell_rank and .year_offset).
#' @param var_name Source variable name.
#' @param stats_mat N x 3 matrix from compute_neighbor_stats_fast/safe.
#' @return Data frame with three new columns added.
add_neighbor_columns <- function(data, var_name, stats_mat) {
  data[[paste0(var_name, "_neighbor_max")]]  <- stats_mat[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats_mat[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats_mat[, 3]
  data
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_optimized_neighbor_features <- function(cell_data, id_order,
                                            rook_neighbors_unique) {
  cat("Step 1: Building cell-level neighbor index...\n")
  t0 <- proc.time()
  cell_nb <- build_cell_neighbor_index(id_order, rook_neighbors_unique)
  cat(sprintf("  Done. %.1f seconds.\n", (proc.time() - t0)[3]))

  cat("Step 2: Sorting data and building index mapping...\n")
  t0 <- proc.time()
  mapped <- sort_and_map(cell_data, id_order)
  cell_data <- mapped$data
  C  <- mapped$C
  T_ <- mapped$T
  cat(sprintf("  Done. C=%d, T=%d, N=%d. %.1f seconds.\n",
              C, T_, nrow(cell_data), (proc.time() - t0)[3]))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  cat("Step 3: Computing neighbor features for", length(neighbor_source_vars),
      "variables...\n")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing '%s'...\n", var_name))
    t0 <- proc.time()

    if (mapped$complete_grid) {
      stats_mat <- compute_neighbor_stats_fast(
        cell_data[[var_name]], cell_nb, C, T_
      )
    } else {
      stats_mat <- compute_neighbor_stats_safe(
        cell_data, cell_nb, var_name, C, T_
      )
    }

    cell_data <- add_neighbor_columns(cell_data, var_name, stats_mat)
    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("    Done. %.1f seconds.\n", elapsed))
  }

  # Clean up helper columns
  cell_data$.cell_rank   <- NULL
  cell_data$.year_offset <- NULL

  cat("All neighbor features computed.\n")
  cell_data
}

# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# cell_data <- run_optimized_neighbor_features(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # The trained Random Forest model is unchanged â€” just predict as before:
# # predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Guarantees

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup granularity** | Per cell-year (6.46M entries) | Per cell (344K entries) + integer stride |
| **Key type** | String paste + hash | Integer arithmetic |
| **Neighbor resolution** | 6.46M `lapply` iterations with `paste` | 344K loop iterations with `outer` + matrix indexing |
| **Per-variable cost** | Re-traverses 6.46M-element list | Vectorized matrix column extraction |
| **Estimated runtime** | ~86+ hours | **~2â€“10 minutes** (depending on RAM/cache) |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** â€” same aggregation functions, same NA handling |
| **RAM peak** | String vector ~500MB + 6.46M-element list | ~344KÃ—28 matrix per variable (~77MB) + index matrix |
| **RF model** | Untouched | **Untouched** â€” column names and values are identical |

The `matrixStats` package is used opportunistically for `rowMaxs`/`rowMins`/`rowMeans2` (much faster than `apply`), with a base-R fallback. Install it for best performance: `install.packages("matrixStats")`.
END_RESPONSE>>>
