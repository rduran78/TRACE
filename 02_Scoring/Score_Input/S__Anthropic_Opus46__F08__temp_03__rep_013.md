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
run_id: Anthropic_F08_T03_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The current implementation is extremely slow (~86+ hours) because of a fundamental design flaw: **it conflates static topology with dynamic data**.

### Root Cause Analysis

1. **`build_neighbor_lookup` rebuilds neighbor relationships per cell-year row (~6.46M rows) instead of per cell (~344K cells).** The rook-neighbor topology is purely spatial â€” it never changes across years. Yet the current code constructs a lookup of length 6.46M, where each entry re-discovers the same neighbors for the same cell, just keyed to a different year. This means the same neighbor-finding work is repeated 28 times per cell.

2. **`compute_neighbor_stats` indexes into the full 6.46M-row data frame using these year-specific row indices.** This is correct in principle but is built on the bloated lookup. Each of the ~6.46M entries in `neighbor_lookup` stores a vector of row indices into the full panel, and the `lapply` over 6.46M elements with per-element allocation is extremely slow in R.

3. **The string-pasting key construction** (`paste(id, year, sep="_")`) and named-vector lookups (`idx_lookup[neighbor_keys]`) over 6.46M keys are themselves very expensive â€” O(n) string operations and hash lookups repeated inside a per-row `lapply`.

4. **Memory pressure:** Storing 6.46M list elements (each a vector of ~4 neighbor row indices) plus intermediate string vectors causes significant GC overhead on a 16 GB laptop.

### Summary

| Aspect | Static (per-cell) | Dynamic (per-cell-year) |
|---|---|---|
| Neighbor topology | âœ… Same across all 28 years | â€” |
| Variable values (ntl, ec, â€¦) | â€” | âœ… Change every year |
| Current lookup granularity | âŒ Per cell-year (6.46M) | â€” |
| Optimal lookup granularity | âœ… Per cell (344K) | â€” |

The fix is to **separate the static neighbor graph (built once over 344K cells) from the dynamic variable values (sliced per year), and compute neighbor stats year-by-year using fast vectorized/matrix operations**.

---

## Optimization Strategy

### 1. Build the neighbor graph once, at the cell level (not cell-year level)

Convert `rook_neighbors_unique` (an `nb` object of length 344,208) into a **sparse adjacency matrix** (344K Ã— 344K). This is a one-time O(cells) operation.

### 2. Compute neighbor stats per year via sparse matrixâ€“vector multiplication

For each year and each variable:
- Extract the column vector **v** of length 344K (one value per cell for that year).
- **Neighbor mean:** `A %*% v / degree` where `A` is the binary adjacency matrix and `degree` is the number of neighbors per cell.
- **Neighbor max and min:** Use a row-wise sparse sweep. Since rook neighbors are â‰¤4 per cell, this is efficiently done with a small loop over neighbor columns or via `data.table` grouped operations.

Sparse matrix multiplication for mean is O(edges) â‰ˆ 1.37M operations per variable-year â€” trivial. Max/min require a grouped operation but over only 344K cells with â‰¤4 neighbors each.

### 3. Reassemble into the panel

Join the per-cell-year results back into the 6.46M-row data frame by `(id, year)`.

### Expected Speedup

| Step | Current | Optimized |
|---|---|---|
| Build lookup | ~6.46M string ops + hash | ~344K sparse matrix (once) |
| Neighbor stats (per var) | ~6.46M list iterations | 28 Ã— sparse matmul (344K Ã— ~4) |
| Total vars Ã— years | 5 Ã— 6.46M = 32.3M R-level iterations | 5 Ã— 28 = 140 vectorized ops |
| **Estimated time** | **86+ hours** | **~2â€“10 minutes** |

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from dynamic (yearly) variable values.
# Preserves the original numerical estimand (neighbor max, min, mean).
# Does NOT retrain or modify the Random Forest model.
# =============================================================================

library(Matrix)
library(data.table)

# ---- STEP 0: Ensure cell_data is a data.table for performance ----
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# ---- STEP 1: Build static sparse adjacency matrix (once) --------------------
# rook_neighbors_unique is an nb object of length n_cells (344,208).
# id_order is the vector of cell IDs corresponding to indices 1..n_cells.

build_adjacency_matrix <- function(neighbors_nb, n_cells) {
  # neighbors_nb: list of integer vectors (nb object), each entry gives
  #               indices of neighbors for that cell (1-based into id_order).
  # Returns: sparse binary adjacency matrix (n_cells x n_cells), class dgCMatrix.

  # Pre-count total edges for pre-allocation
  n_edges <- sum(vapply(neighbors_nb, length, integer(1)))

  # Build COO triplets
  row_idx <- integer(n_edges)
  col_idx <- integer(n_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_i <- neighbors_nb[[i]]
    len  <- length(nb_i)
    if (len > 0L) {
      row_idx[pos:(pos + len - 1L)] <- i
      col_idx[pos:(pos + len - 1L)] <- nb_i
      pos <- pos + len
    }
  }

  sparseMatrix(
    i    = row_idx,
    j    = col_idx,
    x    = 1,
    dims = c(n_cells, n_cells),
    giveCsparse = TRUE
  )
}

n_cells <- length(id_order)
cat("Building sparse adjacency matrix for", n_cells, "cells...\n")
A <- build_adjacency_matrix(rook_neighbors_unique, n_cells)

# Degree vector (number of neighbors per cell) â€” static
degree <- as.integer(rowSums(A))  # length n_cells

# Create a mapping from cell id to matrix row index (static)
id_to_matrow <- setNames(seq_len(n_cells), as.character(id_order))

# ---- STEP 2: Ensure cell_data has a matrix-row index column ----------------
cell_data[, mat_row := id_to_matrow[as.character(id)]]

# Sort by (year, mat_row) so we can work in contiguous year blocks
setkey(cell_data, year, mat_row)

# ---- STEP 3: Compute neighbor max, min, mean per variable per year ---------

# For MEAN: sparse matrix multiplication is exact and fast.
# For MAX / MIN: we iterate over the (small) adjacency list per cell.
#   With â‰¤4 neighbors per cell and 344K cells, this is fast in vectorized R.

# Pre-build adjacency list from the sparse matrix (once, static)
# This is just rook_neighbors_unique itself, but let's ensure clean integer lists.
adj_list <- lapply(seq_len(n_cells), function(i) {
  rook_neighbors_unique[[i]]
})

compute_neighbor_features_for_variable <- function(cell_dt, A, adj_list,
                                                   degree, n_cells,
                                                   var_name) {
  # Computes neighbor_max, neighbor_min, neighbor_mean for var_name
  # across all years. Adds three new columns to cell_dt (by reference).

  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Initialize result columns with NA_real_
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]

  years <- sort(unique(cell_dt$year))

  for (yr in years) {
    # Extract rows for this year (cell_dt is keyed by year, mat_row)
    yr_rows <- cell_dt[.(yr)]  # subset by year via key

    # Build a full-length vector for this variable, indexed by mat_row
    # Some cells may be missing in a given year; those stay NA.
    v <- rep(NA_real_, n_cells)
    v[yr_rows$mat_row] <- yr_rows[[var_name]]

    # --- MEAN via sparse matrix multiplication ---
    # Replace NA with 0 for multiplication, but track valid counts
    v_zero   <- v
    v_valid  <- as.numeric(!is.na(v))  # 1 if valid, 0 if NA
    v_zero[is.na(v_zero)] <- 0

    neighbor_sum   <- as.numeric(A %*% v_zero)    # sum of neighbor values
    neighbor_count <- as.numeric(A %*% v_valid)   # count of non-NA neighbors

    n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # --- MAX and MIN via adjacency list (vectorized per cell) ---
    # For cells with degree 0 or all-NA neighbors, result is NA.
    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)

    # Only process cells that exist in this year AND have neighbors
    active_cells <- yr_rows$mat_row
    for (ci in active_cells) {
      nb <- adj_list[[ci]]
      if (length(nb) == 0L) next
      nb_vals <- v[nb]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      n_max[ci] <- max(nb_vals)
      n_min[ci] <- min(nb_vals)
    }

    # Write results back into cell_dt for this year's rows
    # We need the row indices in cell_dt, not just the subset
    idx_in_dt <- cell_dt[.(yr), which = TRUE]
    mr        <- cell_dt$mat_row[idx_in_dt]

    set(cell_dt, i = idx_in_dt, j = col_max,  value = n_max[mr])
    set(cell_dt, i = idx_in_dt, j = col_min,  value = n_min[mr])
    set(cell_dt, i = idx_in_dt, j = col_mean, value = n_mean[mr])
  }

  invisible(cell_dt)
}

# ---- STEP 4: Run for all 5 neighbor source variables -----------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  t0 <- proc.time()
  compute_neighbor_features_for_variable(
    cell_dt  = cell_data,
    A        = A,
    adj_list = adj_list,
    degree   = degree,
    n_cells  = n_cells,
    var_name = var_name
  )
  elapsed <- (proc.time() - t0)[3]
  cat("  Done in", round(elapsed, 1), "seconds.\n")
}

# ---- STEP 5: Clean up helper column ---------------------------------------
cell_data[, mat_row := NULL]

# ---- STEP 6: Proceed to prediction with the pre-trained Random Forest -----
# The trained RF model is unchanged. cell_data now contains the same
# 15 neighbor feature columns (5 vars Ã— 3 stats) with identical values
# as the original implementation, ready for predict().
#
# Example (unchanged from original pipeline):
# predictions <- predict(trained_rf_model, newdata = cell_data)

cat("Neighbor feature computation complete. Ready for RF prediction.\n")
```

---

## Further Optimization: Vectorized Max/Min (Eliminating the Inner R Loop)

The per-cell R loop for max/min over `active_cells` (~344K iterations Ã— 28 years) may still take a few minutes. Here is a fully vectorized alternative using `data.table` grouped operations on an edge list:

```r
# ---- ALTERNATIVE: Fully vectorized max/min via edge-list approach ----------

# Build a static edge list (once) from the adjacency list
build_edge_dt <- function(adj_list, n_cells) {
  from <- rep(seq_len(n_cells), lengths(adj_list))
  to   <- unlist(adj_list, use.names = FALSE)
  data.table(from = from, to = to)
}

cat("Building static edge list...\n")
edge_dt <- build_edge_dt(adj_list, n_cells)
# edge_dt has ~1.37M rows: columns (from, to) are mat_row indices.

compute_neighbor_features_vectorized <- function(cell_dt, A, edge_dt,
                                                 n_cells, var_name) {
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]

  years <- sort(unique(cell_dt$year))

  for (yr in years) {
    # Full vector of values indexed by mat_row
    v <- rep(NA_real_, n_cells)
    idx_in_dt <- cell_dt[.(yr), which = TRUE]
    mr <- cell_dt$mat_row[idx_in_dt]
    v[mr] <- cell_dt[[var_name]][idx_in_dt]

    # --- MEAN via sparse matmul (same as before) ---
    v_zero <- v; v_zero[is.na(v_zero)] <- 0
    v_valid <- as.numeric(!is.na(v))
    neighbor_sum   <- as.numeric(A %*% v_zero)
    neighbor_count <- as.numeric(A %*% v_valid)
    n_mean <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)

    # --- MAX and MIN via edge_dt (fully vectorized) ---
    # Attach neighbor values to edges
    edge_dt[, nb_val := v[to]]

    # Group by 'from' cell, compute max and min (NA-aware)
    stats <- edge_dt[!is.na(nb_val),
                     .(nb_max = max(nb_val), nb_min = min(nb_val)),
                     by = from]

    n_max <- rep(NA_real_, n_cells)
    n_min <- rep(NA_real_, n_cells)
    n_max[stats$from] <- stats$nb_max
    n_min[stats$from] <- stats$nb_min

    # Write back
    set(cell_dt, i = idx_in_dt, j = col_max,  value = n_max[mr])
    set(cell_dt, i = idx_in_dt, j = col_min,  value = n_min[mr])
    set(cell_dt, i = idx_in_dt, j = col_mean, value = n_mean[mr])
  }

  # Clean up edge_dt temp column
  edge_dt[, nb_val := NULL]
  invisible(cell_dt)
}

# Run the vectorized version:
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features (vectorized) for:", var_name, "\n")
  t0 <- proc.time()
  compute_neighbor_features_vectorized(
    cell_dt  = cell_data,
    A        = A,
    edge_dt  = edge_dt,
    n_cells  = n_cells,
    var_name = var_name
  )
  elapsed <- (proc.time() - t0)[3]
  cat("  Done in", round(elapsed, 1), "seconds.\n")
}
```

---

## Summary

| | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-keyed list entries | 344K Ã— 344K sparse matrix (built once) |
| **Stats computation** | R-level `lapply` over 6.46M elements, per variable | Sparse matmul (mean) + `data.table` grouped edge ops (max/min), per year |
| **Total inner iterations** | ~32.3M R-level list evaluations | 140 vectorized operations (5 vars Ã— 28 years) |
| **Estimated runtime** | 86+ hours | **2â€“10 minutes** |
| **RAM usage** | High (6.46M-element list + strings) | Low (~50 MB sparse matrix + 1.37M-row edge table) |
| **RF model** | Unchanged âœ… | Unchanged âœ… |
| **Numerical results** | Original | Identical âœ… |
END_RESPONSE>>>
