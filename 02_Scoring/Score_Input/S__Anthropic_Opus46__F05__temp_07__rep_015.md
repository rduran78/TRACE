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
run_id: Anthropic_F05_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local inefficiency
Inside `build_neighbor_lookup`, the call `paste(data$id, data$year, sep = "_")` creates a character vector of ~6.46 million strings to build `idx_lookup`. Then, for **each** of the 6.46 million rows, it calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and does a named-vector lookup. That is O(N Ã— average_neighbors) string constructions and hash lookups â€” roughly 6.46M Ã— 4 (average rook neighbors) â‰ˆ 25.8 billion character operations.

### Broader algorithmic problem
The deeper issue is that the algorithm treats the data as a flat cell-year panel and uses **string keys** to resolve what is fundamentally an **integer indexing** problem. The neighbor relationships are **time-invariant** â€” cell `A` neighbors cell `B` in every year. The data is a balanced panel (344,208 cells Ã— 28 years). Therefore:

1. **The neighbor graph is identical for every year.** There is no need to look up neighbors per cell-year; you only need the neighbor structure per cell, then apply it within each year-slice.
2. **String-keyed lookup is unnecessary.** If you sort/index the data by `(year, id)` â€” or equivalently maintain an integer offset per year â€” you can resolve any neighbor's row position with pure integer arithmetic.
3. **The `lapply` over 6.46M rows is serialized in R's interpreter**, creating millions of small vectors. This should be replaced by a vectorized or matrix-based approach.

### Computational cost summary

| Operation | Current cost | Optimized cost |
|---|---|---|
| Build lookup | O(N) string hashing | O(1) â€” precompute integer offsets |
| Per-row neighbor resolution | O(N Ã— k) string paste + hash lookup | O(N Ã— k) integer arithmetic (vectorized) |
| Neighbor stats (per variable) | O(N) lapply, fine | O(N Ã— k) vectorized matrix ops |
| Total string operations | ~25.8B | 0 |

## Optimization Strategy

1. **Sort the panel by `(year, id)` and build a simple integer index.** Since the panel is balanced, row position = `(year_index - 1) * n_cells + cell_index`. All neighbor lookups become integer addition.

2. **Convert the `nb` object to a sparse adjacency structure once** (a pair of integer vectors: `from`, `to`), then use vectorized operations across all rows simultaneously.

3. **Compute neighbor stats per variable using a sparse-matrix multiply** or vectorized group operation â€” no R-level loop over 6.46M rows.

4. **The trained Random Forest model is untouched.** We only change how the input feature columns are computed; the numerical results are identical (same max, min, mean of the same neighbor values).

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# Preserves exact numerical output. Does not touch the trained RF model.
# =============================================================================

library(Matrix)   # for sparse matrix operations
library(data.table)

#' Build neighbor features for all source variables, vectorized.
#'
#' @param cell_data       data.frame/data.table with columns: id, year, and all
#'                        neighbor_source_vars. Must be a balanced panel.
#' @param id_order        integer vector of cell IDs in the order used by the nb object.
#' @param nb_obj          spdep nb object (rook_neighbors_unique).
#' @param neighbor_source_vars character vector of variable names.
#' @return cell_data with new neighbor feature columns appended.
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        nb_obj,
                                        neighbor_source_vars) {

  # --- Step 0: Convert to data.table for speed; record original order --------
  was_df <- !is.data.table(cell_data)
  dt <- as.data.table(cell_data)
  dt[, ..orig_row_order := .I]

  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  n_years <- length(years)

  stopifnot(
    "Unbalanced panel detected" = nrow(dt) == n_cells * n_years
  )

  # --- Step 1: Build integer cell index and year index -----------------------
  # Map each cell id to its position in id_order (1..n_cells)
  id_to_cidx <- setNames(seq_along(id_order), as.character(id_order))
  year_to_yidx <- setNames(seq_along(years), as.character(years))

  dt[, cidx := id_to_cidx[as.character(id)]]
  dt[, yidx := year_to_yidx[as.character(year)]]

  # Sort by (yidx, cidx) so that row position = (yidx-1)*n_cells + cidx
  setorder(dt, yidx, cidx)

  # Verify the layout
  stopifnot(all(dt$cidx == rep(seq_len(n_cells), n_years)))
  stopifnot(all(dt$yidx == rep(seq_len(n_years), each = n_cells)))

  # --- Step 2: Build sparse adjacency matrix from nb object ------------------
  # nb_obj[[i]] contains integer indices of neighbors of cell i (in id_order).
  # Build COO triplets for a binary adjacency matrix W (n_cells x n_cells).
  from_vec <- integer(0)
  to_vec   <- integer(0)
  for (i in seq_along(nb_obj)) {
    nbrs <- nb_obj[[i]]
    # spdep uses 0L to denote "no neighbors"
    nbrs <- nbrs[nbrs > 0L]
    if (length(nbrs) > 0L) {
      from_vec <- c(from_vec, rep.int(i, length(nbrs)))
      to_vec   <- c(to_vec, nbrs)
    }
  }

  # Sparse binary adjacency matrix: W[i,j] = 1 if j is neighbor of i
  W <- sparseMatrix(
    i = from_vec,
    j = to_vec,
    x = 1,
    dims = c(n_cells, n_cells)
  )

  # Number of neighbors per cell (for computing means)
  n_neighbors <- as.numeric(W %*% rep(1, n_cells))  # length n_cells

  # --- Step 3: For each variable, compute neighbor max/min/mean --------------
  #
  # Key insight: because the data is sorted (yidx, cidx), the column of values

  # for variable v is a vector of length n_cells * n_years laid out as
  # [year1_cell1, year1_cell2, ..., year1_cellN, year2_cell1, ...].
  #
  # We can reshape to a matrix V of shape (n_cells, n_years), then:
  #   - neighbor_sum  = W %*% V          (sparse mat-mul, gives sum per cell per year)
  #   - neighbor_mean = neighbor_sum / n_neighbors
  #   - For max and min we iterate over the adjacency list but vectorized per year.
  #
  # For max/min there is no single sparse-matrix trick, but we can do it

  # efficiently with a vectorized approach over the edge list.

  # Pre-extract the adjacency list as two integer vectors (already have from_vec, to_vec)
  # For each directed edge (from_vec[e], to_vec[e]), and each year y,
  # the "from" cell needs the value at "to" cell in year y.

  # Edge-level row indices into dt (which is sorted by yidx, cidx):
  # row_of(cell=c, year_idx=y) = (y-1)*n_cells + c
  # For edge e in year y: source_row = (y-1)*n_cells + from_vec[e]
  #                        target_row = (y-1)*n_cells + to_vec[e]

  n_edges <- length(from_vec)

  for (var_name in neighbor_source_vars) {

    message(sprintf("  Computing neighbor features for: %s", var_name))

    vals <- dt[[var_name]]  # length = n_cells * n_years, sorted by (yidx, cidx)

    # Reshape to matrix: rows = cells, cols = years
    V <- matrix(vals, nrow = n_cells, ncol = n_years)
    # V[c, y] = value for cell c in year y

    # ---- Neighbor mean via sparse matrix multiply ----------------------------
    # S = W %*% V  => S[i, y] = sum of neighbor values for cell i in year y
    S <- as.matrix(W %*% V)  # dense result, n_cells x n_years

    # Count of non-NA neighbors per cell per year
    V_notna <- matrix(as.numeric(!is.na(vals)), nrow = n_cells, ncol = n_years)
    N_valid <- as.matrix(W %*% V_notna)  # n_cells x n_years

    # For sum, we need W %*% V but treating NA as 0 in the sum:
    V_zero <- V
    V_zero[is.na(V_zero)] <- 0
    S_clean <- as.matrix(W %*% V_zero)

    neighbor_mean <- S_clean / N_valid
    neighbor_mean[N_valid == 0] <- NA

    # ---- Neighbor max and min via edge-list vectorization --------------------
    # Initialize max to -Inf and min to +Inf, then sweep over edges
    neighbor_max <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    neighbor_min <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    # Only initialize cells that actually have neighbors
    has_neighbors <- n_neighbors > 0
    neighbor_max[has_neighbors, ] <- -Inf
    neighbor_min[has_neighbors, ] <-  Inf

    # For each year, vectorized update over all edges
    for (y in seq_len(n_years)) {
      v_y <- V[, y]  # values this year, length n_cells

      edge_vals <- v_y[to_vec]  # value at the "to" end of each edge
      valid     <- !is.na(edge_vals)

      if (any(valid)) {
        f <- from_vec[valid]
        ev <- edge_vals[valid]

        # Compute max per "from" cell using tapply-style vectorization
        # We use data.table for speed
        edge_dt <- data.table(f = f, ev = ev)

        max_dt <- edge_dt[, .(mx = max(ev)), by = f]
        cur_max <- neighbor_max[max_dt$f, y]
        neighbor_max[max_dt$f, y] <- pmax(cur_max, max_dt$mx, na.rm = TRUE)

        min_dt <- edge_dt[, .(mn = min(ev)), by = f]
        cur_min <- neighbor_min[min_dt$f, y]
        neighbor_min[min_dt$f, y] <- pmin(cur_min, min_dt$mn, na.rm = TRUE)
      }
    }

    # Cells with no valid neighbor values -> NA
    neighbor_max[is.infinite(neighbor_max)] <- NA
    neighbor_min[is.infinite(neighbor_min)] <- NA

    # ---- Flatten back to panel vector (n_cells * n_years, sorted by yidx, cidx)
    max_col_name  <- paste0("neighbor_max_", var_name)
    min_col_name  <- paste0("neighbor_min_", var_name)
    mean_col_name <- paste0("neighbor_mean_", var_name)

    dt[, (max_col_name)  := as.vector(neighbor_max)]
    dt[, (min_col_name)  := as.vector(neighbor_min)]
    dt[, (mean_col_name) := as.vector(neighbor_mean)]
  }

  # --- Step 4: Restore original row order and clean up -----------------------
  setorder(dt, ..orig_row_order)
  dt[, c("cidx", "yidx", "..orig_row_order") := NULL]

  if (was_df) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================
#
# # Original code (86+ hours):
# # neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# # for (var_name in neighbor_source_vars) {
# #   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# # }
#
# # Optimized replacement (~minutes):
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- build_all_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   nb_obj               = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # The trained Random Forest model is unchanged.
# # predict(rf_model, new_data) works exactly as before.
```

## Why This Works and Complexity Analysis

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor resolution** | 6.46M string pastes + hash lookups | Integer arithmetic, done implicitly by matrix layout |
| **String operations** | ~25.8 billion characters | **Zero** |
| **Neighbor mean** | R-level `lapply` over 6.46M rows | Single sparse matrix multiply `W %*% V` (~1.37M nonzeros Ã— 28 cols) |
| **Neighbor max/min** | R-level `lapply` with per-row subsetting | Vectorized edge-list sweep, 28 iterations (one per year), each fully vectorized via `data.table` |
| **Memory** | 6.46M-element named character vector + millions of small allocations | Sparse matrix W (~5.5 MB) + dense matrix V (~49 MB per variable, reused) |
| **Estimated time** | 86+ hours | **~2â€“10 minutes** |
| **Numerical result** | max, min, mean of non-NA neighbor values | **Identical** â€” same neighbors, same values, same aggregation functions |

### Key algorithmic insights:

1. **Time-invariant topology.** The rook neighbor graph doesn't change across years. The original code re-resolved this for every cell-year row via string matching. The optimized code encodes the topology once in a sparse matrix `W`.

2. **Balanced panel â‡’ matrix reshape.** With 344,208 cells Ã— 28 years, any variable column can be reshaped to a `(cells Ã— years)` matrix. Sparse matrix multiplication `W %*% V` then computes all neighbor sums across all cells and all years in one operation.

3. **Edge-list vectorization for max/min.** Since max and min are not linear operators, we can't use matrix multiplication. Instead, we iterate over 28 years (not 6.46M rows) and use `data.table` grouped aggregation over the ~1.37M edges per year â€” fully vectorized in C.

4. **No model retraining needed.** The output columns (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc.) contain numerically identical values, so the trained Random Forest model's predictions are unchanged.
END_RESPONSE>>>
