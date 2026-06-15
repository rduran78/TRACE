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
run_id: Anthropic_F10_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation suffers from three critical bottlenecks:

1. **O(nÂ²)-scale neighbor lookup construction**: `build_neighbor_lookup` creates a per-row lookup by pasting string keys and matching them in a named vector for every single row (~6.46M rows). The `paste`/`match` on character keys is extremely slow at this scale.

2. **Row-level R `lapply` over 6.46M rows in `compute_neighbor_stats`**: Each call to the anonymous function has R-interpreter overhead (function dispatch, subsetting, `is.na` checks, `max/min/mean`). Over 6.46M iterations Ã— 5 variables = ~32.3M R-level loop iterations.

3. **Topology is entangled with time**: The neighbor graph is purely spatial (rook contiguity among 344,208 cells) but the lookup is rebuilt as if it's a 6.46M-node graph. The same spatial adjacency is redundantly expanded across all 28 years.

**Key insight**: The rook neighbor graph has ~1.37M directed edges among 344,208 spatial nodes. This topology is *identical* for every year. The task is simply: for each year and each variable, gather neighbor values and compute `max`, `min`, `mean`. This is a **sparse matrixâ€“vector product** (for mean) and analogous sparse aggregation (for max/min), repeated 28 Ã— 5 = 140 times â€” trivially fast if done correctly.

---

## Optimization Strategy

| Principle | Implementation |
|---|---|
| **Separate topology from time** | Build a 344,208 Ã— 344,208 sparse adjacency matrix once from the `nb` object. |
| **Vectorized sparse aggregation** | Use `Matrix` package sparse matrix operations for `mean`. Use `data.table` grouped operations for `max` and `min`. |
| **Avoid character key matching** | Use integer indexing throughout. Map cell IDs to integer positions once. |
| **Process year-by-year in a matrix** | Reshape each variable to a 344,208 Ã— 28 matrix, apply sparse aggregation column-wise. |
| **Minimize memory copies** | Use `data.table` set-by-reference to attach new columns. |

**Expected speedup**: From 86+ hours to **minutes** (typically 5â€“15 minutes on a 16 GB laptop).

---

## Optimized R Code

```r
library(Matrix)
library(data.table)

optimize_neighbor_pipeline <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                 "def", "usd_est_n2")) {

  # ---------------------------------------------------------------
  # 0. Convert to data.table for efficient column operations
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  cat(sprintf("Cells: %d | Years: %d | Rows: %d\n", n_cells, n_years, nrow(cell_data)))

  # ---------------------------------------------------------------
  # 1. Build sparse adjacency matrix ONCE (topology only)
  #    rook_neighbors_unique is an nb object: a list of length n_cells

  #    where element i contains integer indices of neighbors of cell i
  #    (indices into id_order).
  # ---------------------------------------------------------------
  cat("Building sparse adjacency matrix...\n")

  # Build COO (coordinate) representation
  from_list <- lapply(seq_len(n_cells), function(i) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb objects use 0L to indicate no neighbors
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) == 0L) return(NULL)
    list(i = rep.int(i, length(nb_i)), j = nb_i)
  })

  ii <- unlist(lapply(from_list, `[[`, "i"), use.names = FALSE)
  jj <- unlist(lapply(from_list, `[[`, "j"), use.names = FALSE)

  # Binary adjacency matrix: A[i,j] = 1 if j is a rook neighbor of i
  A <- sparseMatrix(i = ii, j = jj, x = 1, dims = c(n_cells, n_cells))

  # Degree vector (number of neighbors per cell) for computing mean
  degree <- as.numeric(rowSums(A))  # length n_cells

  cat(sprintf("Adjacency matrix: %d non-zeros (directed edges)\n", length(ii)))

  # ---------------------------------------------------------------
  # 2. Build integer mappings: cell_id -> position, year -> position
  # ---------------------------------------------------------------
  id_to_pos   <- setNames(seq_len(n_cells), as.character(id_order))
  year_to_col <- setNames(seq_len(n_years), as.character(years))

  # Map each row of cell_data to (cell_position, year_position)
  cell_data[, c("cell_pos__", "year_pos__") := list(
    id_to_pos[as.character(id)],
    year_to_col[as.character(year)]
  )]

  # Ensure data is sorted by (cell_pos, year_pos) for matrix filling
  setorder(cell_data, cell_pos__, year_pos__)

  # ---------------------------------------------------------------
  # 3. For each source variable, compute neighbor max, min, mean
  #    Strategy:
  #      - Reshape variable to n_cells x n_years matrix V
  #      - MEAN:  (A %*% V) / degree  (sparse mat-mat multiply)
  #      - MAX/MIN: use the COO edges + data.table grouped aggregation
  # ---------------------------------------------------------------

  # Pre-extract edge list as data.table for max/min computation
  edge_dt <- data.table(from = ii, to = jj)

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing variable: %s\n", var_name))
    t0 <- proc.time()

    # 3a. Build n_cells x n_years matrix V from cell_data
    vals <- cell_data[[var_name]]
    V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    V[cbind(cell_data$cell_pos__, cell_data$year_pos__)] <- vals

    # 3b. MEAN via sparse matrix multiplication
    #     sum_neighbors = A %*% V   (n_cells x n_years)
    #     mean_neighbors = sum_neighbors / degree
    #     Where degree == 0, result is NaN -> convert to NA
    sum_V <- as.matrix(A %*% V)  # dense result, n_cells x n_years
    mean_V <- sum_V / degree     # vectorized division (recycles by column)
    mean_V[degree == 0, ] <- NA_real_
    # Also: if a neighbor exists but its value is NA, the sparse multiply
    # treats it as 0. We need to handle NAs properly.
    # Fix: count non-NA neighbors and non-NA sums separately.

    # --- Correct NA-aware mean ---
    # Replace NA with 0 in V for summation, and create indicator matrix
    V_nona <- V
    V_nona[is.na(V_nona)] <- 0
    V_ind <- (!is.na(V)) * 1  # indicator: 1 if not NA, 0 if NA

    sum_V    <- as.matrix(A %*% V_nona)       # sum of non-NA neighbor values
    count_V  <- as.matrix(A %*% V_ind)        # count of non-NA neighbors
    mean_V   <- sum_V / count_V               # NA-aware mean
    mean_V[count_V == 0] <- NA_real_          # no valid neighbors -> NA

    # 3c. MAX and MIN via edge list + data.table grouped aggregation
    #     For each year, look up neighbor values, group by 'from', compute max/min
    max_V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    min_V <- matrix(NA_real_, nrow = n_cells, ncol = n_years)

    for (yr_idx in seq_len(n_years)) {
      # Get the column of values for this year
      v_yr <- V[, yr_idx]

      # Look up neighbor values: for each edge (from, to), value = v_yr[to]
      edge_dt[, val := v_yr[to]]

      # Remove edges where neighbor value is NA
      valid <- edge_dt[!is.na(val)]

      if (nrow(valid) > 0L) {
        agg <- valid[, .(mx = max(val), mn = min(val)), by = from]
        max_V[agg$from, yr_idx] <- agg$mx
        min_V[agg$from, yr_idx] <- agg$mn
      }
    }

    # 3d. Map results back to cell_data row order
    idx_mat <- cbind(cell_data$cell_pos__, cell_data$year_pos__)

    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    set(cell_data, j = col_max,  value = max_V[idx_mat])
    set(cell_data, j = col_min,  value = min_V[idx_mat])
    set(cell_data, j = col_mean, value = mean_V[idx_mat])

    elapsed <- (proc.time() - t0)[3]
    cat(sprintf("    Done in %.1f seconds\n", elapsed))
  }

  # ---------------------------------------------------------------
  # 4. Clean up temporary columns
  # ---------------------------------------------------------------
  cell_data[, c("cell_pos__", "year_pos__") := NULL]

  # Remove temporary column from edge_dt
  edge_dt[, val := NULL]

  cat("Neighbor feature computation complete.\n")
  return(cell_data)
}


# ===================================================================
# USAGE
# ===================================================================
#
# # Load data (already in memory or from disk)
# # cell_data: data.frame/data.table with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# # id_order: vector of 344,208 cell IDs matching the nb object indexing
# # rook_neighbors_unique: spdep nb object (loaded via readRDS or similar)
# # rf_model: pre-trained Random Forest (loaded via readRDS)
#
# cell_data <- optimize_neighbor_pipeline(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Now cell_data has 15 new columns:
# #   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
# #   neighbor_max_ec, neighbor_min_ec, neighbor_mean_ec,
# #   ... etc.
#
# # Apply the pre-trained Random Forest (DO NOT retrain)
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **max** | `max(vals[neighbor_indices])` after removing NAs | `data.table` grouped `max` over the same edge set, same NA removal | âœ… Identical |
| **min** | `min(vals[neighbor_indices])` after removing NAs | `data.table` grouped `min` over the same edge set, same NA removal | âœ… Identical |
| **mean** | `mean(vals[neighbor_indices])` after removing NAs | `sum(non-NA values) / count(non-NA values)` via sparse matrix ops with NA-indicator correction | âœ… Identical (IEEE 754 floating point sum order may differ by â‰¤ machine epsilon; practically identical) |

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M string paste+match | Sparse matrix built once from integer COO |
| Mean computation | 6.46M R-level loops per variable | Single sparse matrix multiply (`A %*% V`) per variable |
| Max/Min computation | 6.46M R-level loops per variable | 28 `data.table` grouped aggregations per variable (~1.37M edges each) |
| Total loop iterations | ~32.3M R function calls | ~140 vectorized operations |
| Estimated time | **86+ hours** | **5â€“15 minutes** |
| RAM peak | Moderate (list of 6.46M vectors) | ~1â€“2 GB (sparse matrix + dense n_cellsÃ—n_years matrices) |
| RF model | Untouched | Untouched âœ… |
END_RESPONSE>>>
