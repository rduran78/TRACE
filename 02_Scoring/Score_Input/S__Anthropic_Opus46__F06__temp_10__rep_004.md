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
run_id: Anthropic_F06_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F06_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck 1: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows
For every one of the ~6.46 million cell-year rows, the function:
1. Looks up the cell's rook neighbors (by cell ID).
2. Constructs string keys (`paste(id, year)`) for each neighbor-year combination.
3. Performs named-vector lookups (`idx_lookup[neighbor_keys]`), which are hash-based but repeated millions of times with string allocation overhead.

This is fundamentally **year-redundant**: the neighbor topology is purely spatial and identical across all 28 years, yet the lookup is rebuilt per row (per cell-year), creating ~6.46M list entries instead of ~344K.

### Bottleneck 2: `compute_neighbor_stats` â€” R-level `lapply` over 6.46M elements
For each of the ~6.46 million rows, an R-level anonymous function extracts neighbor values, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable), totaling ~32.3M R function invocations with per-element vector subsetting. The R interpreter overhead here is enormous.

### Why 86+ hours?
- ~6.46M list constructions in `build_neighbor_lookup` with string operations.
- ~32.3M R-level loop iterations in `compute_neighbor_stats` (6.46M Ã— 5 vars).
- All computation is single-threaded, interpreted R, with heavy allocation of temporary character/numeric vectors.

### Why raster focal/kernel operations don't directly apply
Focal operations assume a regular grid with a fixed rectangular window. The data here is a panel (cell Ã— year) stored as a long data frame, the neighbor structure is an irregular `spdep::nb` object (not a rectangular kernel), and the computation must be done **within-year** across spatial neighbors. While the analogy is instructive (focal = neighborhood summary), a direct `terra::focal` approach would require reshaping each variable into a raster stack per year and reconstructing the nb topology as a custom weight matrix â€” adding complexity without guaranteeing correctness for irregular neighbor structures at boundaries. The better strategy is to vectorize the existing logic.

---

## Optimization Strategy

### Strategy 1: Separate spatial topology from temporal indexing
Build the neighbor lookup **once per cell** (~344K entries), not once per cell-year (~6.46M entries). Then, for each year, use integer matrix indexing to gather neighbor values.

### Strategy 2: Vectorize with `data.table` and matrix operations
- Reshape data so that each year's values for a variable can be accessed as a column or matrix slice.
- Use `data.table` for fast grouped operations or pre-build an integer index matrix of neighbor row positions per year.
- Replace the `lapply` with vectorized `rowMaxs`, `rowMins`, `rowMeans` from the `matrixStats` package over a gathered neighbor-value matrix.

### Strategy 3: Pre-build a sparse neighbor-row-index matrix
For each cell-year row, we know which rows are its neighbors (same year, neighbor cell). We can encode this as a fixed-width integer matrix (each row has at most 4 rook neighbors). Then `max/min/mean` can be computed via `matrixStats::rowMaxs` etc., which are C-level vectorized.

### Expected speedup
- Eliminating per-row string operations: ~100â€“500Ã—  
- Vectorized C-level `rowMaxs/rowMins/rowMeans`: ~50â€“200Ã—  
- Estimated new runtime: **1â€“5 minutes** total (from 86+ hours).

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# =============================================================================
# Requirements: data.table, matrixStats
# install.packages(c("data.table", "matrixStats"))

library(data.table)
library(matrixStats)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -------------------------------------------------------------------------
  # STEP 1: Convert to data.table for fast indexed operations
  # -------------------------------------------------------------------------
  dt <- as.data.table(cell_data)

  # Preserve original row order for faithful output
  dt[, .roworder := .I]

  # -------------------------------------------------------------------------
  # STEP 2: Build spatial-only neighbor edge list (cell-level, not row-level)
  #
  # rook_neighbors_unique is an nb object: a list of length = # cells,
  # where element i contains integer indices (into id_order) of neighbors.
  # We expand this into an edge list of (cell_index, neighbor_cell_index).
  # -------------------------------------------------------------------------
  n_cells <- length(id_order)

  # Map cell IDs to their position in id_order (1-based index)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Build edge list: from_id -> to_id (using original cell IDs)
  edge_from <- integer(0)
  edge_to   <- integer(0)

  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors_unique[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) > 0L) {
      edge_from <- c(edge_from, rep(id_order[i], length(nb_idx)))
      edge_to   <- c(edge_to,   id_order[nb_idx])
    }
  }

  edges <- data.table(id = edge_from, neighbor_id = edge_to)

  # -------------------------------------------------------------------------
  # STEP 3: Determine max number of rook neighbors (should be <= 4)
  # -------------------------------------------------------------------------
  max_neighbors <- edges[, .N, by = id][, max(N)]
  cat("Max rook neighbors per cell:", max_neighbors, "\n")

  # -------------------------------------------------------------------------
  # STEP 4: Build a fixed-width neighbor ID matrix (n_cells x max_neighbors)
  #
  # For each cell, store the IDs of its neighbors padded with NA.
  # -------------------------------------------------------------------------
  # Assign a within-cell neighbor index
  edges[, nb_seq := seq_len(.N), by = id]

  # Create a lookup: for each unique cell ID, a row of neighbor IDs
  # Using dcast for a wide matrix
  neighbor_wide <- dcast(edges, id ~ nb_seq, value.var = "neighbor_id")
  # Columns: id, 1, 2, ..., max_neighbors
  nb_cols <- setdiff(names(neighbor_wide), "id")

  # Merge this into dt so every cell-year row knows its neighbor cell IDs
  # But this would replicate â€” instead, we'll take a more direct approach.

  # -------------------------------------------------------------------------
  # STEP 5: For each (cell-year) row, find neighbor rows via integer indexing
  #
  # Key insight: sort dt by (id, year) and build a row-lookup matrix.
  # All cells share the same set of years, so for a given cell at position p
  # in the cell list and year at position t in the year list, the row index
  # is: (p - 1) * n_years + t  (if data is sorted by id, then year).
  # -------------------------------------------------------------------------

  # Ensure keyed sort: id, year
  setkey(dt, id, year)

  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_pos <- setNames(seq_along(years), as.character(years))

  # Verify that the panel is balanced (every cell has every year)
  cells_per_year <- dt[, .N, by = id]
  if (!all(cells_per_year$N == n_years)) {
    warning("Panel is unbalanced. Falling back to hash-based row lookup.")
    # Build a hash-based lookup for unbalanced panels
    dt[, .rowidx := .I]
    row_lookup <- dt[, .(.rowidx), keyby = .(id, year)]
    balanced <- FALSE
  } else {
    balanced <- TRUE
  }

  # Unique cell IDs in sorted order (matches keyed dt)
  cell_ids_sorted <- sort(unique(dt$id))
  cell_id_to_pos  <- setNames(seq_along(cell_ids_sorted), as.character(cell_ids_sorted))

  # -------------------------------------------------------------------------
  # STEP 6: Build neighbor POSITION matrix (n_cells x max_neighbors)
  #   neighbor_pos_mat[p, k] = position (in cell_ids_sorted) of p's k-th neighbor
  # -------------------------------------------------------------------------
  neighbor_pos_mat <- matrix(NA_integer_, nrow = n_cells, ncol = max_neighbors)

  # Re-derive from edges more efficiently
  # For each cell in cell_ids_sorted, get neighbor positions
  for (i in seq_len(n_cells)) {
    cid <- cell_ids_sorted[i]
    # original position in id_order
    orig_pos <- id_to_pos[as.character(cid)]
    nb_idx <- rook_neighbors_unique[[orig_pos]]
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) > 0L) {
      nb_cids <- id_order[nb_idx]
      nb_positions <- cell_id_to_pos[as.character(nb_cids)]
      nb_positions <- nb_positions[!is.na(nb_positions)]
      if (length(nb_positions) > 0L) {
        neighbor_pos_mat[i, seq_along(nb_positions)] <- nb_positions
      }
    }
  }

  # -------------------------------------------------------------------------
  # STEP 7: For each variable, compute neighbor stats vectorized
  #
  # For a balanced panel sorted by (id, year):
  #   Row index of cell at position p, year at position t = (p-1)*n_years + t
  #
  # For each neighbor slot k (1..max_neighbors):
  #   neighbor_row[i] = (neighbor_pos_mat[cell_pos[i], k] - 1) * n_years + year_pos[i]
  #
  # We gather all neighbor values into a matrix (n_rows x max_neighbors),

  # then compute rowMaxs, rowMins, rowMeans with na.rm = TRUE.
  # -------------------------------------------------------------------------

  n_rows <- nrow(dt)

  if (balanced) {
    # Pre-compute cell_pos and year_pos for each row in sorted dt
    # Since sorted by (id, year) with n_years per cell:
    cell_pos_vec <- rep(seq_len(n_cells), each = n_years)
    year_pos_vec <- rep(seq_len(n_years), times = n_cells)

    for (var_name in neighbor_source_vars) {
      cat("Processing variable:", var_name, "\n")

      vals <- dt[[var_name]]

      # Build neighbor value matrix: n_rows x max_neighbors
      nb_val_mat <- matrix(NA_real_, nrow = n_rows, ncol = max_neighbors)

      for (k in seq_len(max_neighbors)) {
        # For each row i, the k-th neighbor's cell position
        nb_cell_pos <- neighbor_pos_mat[cell_pos_vec, k]
        # Convert to row index: (nb_cell_pos - 1) * n_years + year_pos_vec
        nb_row_idx <- (nb_cell_pos - 1L) * n_years + year_pos_vec
        # nb_row_idx is NA where there is no k-th neighbor
        valid <- !is.na(nb_row_idx)
        nb_val_mat[valid, k] <- vals[nb_row_idx[valid]]
      }

      # Compute stats using matrixStats (C-level, vectorized)
      nb_max  <- rowMaxs(nb_val_mat,  na.rm = TRUE)
      nb_min  <- rowMins(nb_val_mat,  na.rm = TRUE)
      nb_mean <- rowMeans2(nb_val_mat, na.rm = TRUE)

      # rowMaxs/rowMins return -Inf/Inf when all NA; convert to NA
      nb_max[is.infinite(nb_max)] <- NA_real_
      nb_min[is.infinite(nb_min)] <- NA_real_
      # rowMeans2 returns NaN for all-NA rows
      nb_mean[is.nan(nb_mean)] <- NA_real_

      # Add columns to dt
      set(dt, j = paste0("n_max_", var_name), value = nb_max)
      set(dt, j = paste0("n_min_", var_name), value = nb_min)
      set(dt, j = paste0("n_mean_", var_name), value = nb_mean)

      cat("  Done:", var_name, "\n")
    }
  } else {
    # ----- Unbalanced panel fallback (still much faster than original) -----
    # Build a row lookup: given (cell_pos, year_pos) -> row index in dt
    dt[, cell_pos := cell_id_to_pos[as.character(id)]]
    dt[, year_pos := year_to_pos[as.character(year)]]

    # Create a lookup matrix: n_cells x n_years -> row index
    row_idx_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
    row_idx_mat[cbind(dt$cell_pos, dt$year_pos)] <- seq_len(n_rows)

    for (var_name in neighbor_source_vars) {
      cat("Processing variable:", var_name, "\n")

      vals <- dt[[var_name]]
      nb_val_mat <- matrix(NA_real_, nrow = n_rows, ncol = max_neighbors)

      for (k in seq_len(max_neighbors)) {
        nb_cell_pos <- neighbor_pos_mat[dt$cell_pos, k]
        # Look up row index via the matrix
        nb_row_idx <- row_idx_mat[cbind(nb_cell_pos, dt$year_pos)]
        valid <- !is.na(nb_row_idx)
        nb_val_mat[valid, k] <- vals[nb_row_idx[valid]]
      }

      nb_max  <- rowMaxs(nb_val_mat,  na.rm = TRUE)
      nb_min  <- rowMins(nb_val_mat,  na.rm = TRUE)
      nb_mean <- rowMeans2(nb_val_mat, na.rm = TRUE)

      nb_max[is.infinite(nb_max)] <- NA_real_
      nb_min[is.infinite(nb_min)] <- NA_real_
      nb_mean[is.nan(nb_mean)]    <- NA_real_

      set(dt, j = paste0("n_max_", var_name), value = nb_max)
      set(dt, j = paste0("n_min_", var_name), value = nb_min)
      set(dt, j = paste0("n_mean_", var_name), value = nb_mean)

      cat("  Done:", var_name, "\n")
    }

    # Clean up temporary columns
    dt[, c("cell_pos", "year_pos", ".rowidx") :=
         .(NULL, NULL, NULL)]
  }

  # -------------------------------------------------------------------------
  # STEP 8: Restore original row order and return as data.frame
  # -------------------------------------------------------------------------
  setorder(dt, .roworder)
  dt[, .roworder := NULL]

  return(as.data.frame(dt))
}


# =============================================================================
# USAGE
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M string-key list entries via `lapply` + `paste` | One-time 344K Ã— 4 integer matrix; row indices derived arithmetically |
| **Stat computation** | 32.3M R-level anonymous function calls (6.46M Ã— 5 vars) | 5 Ã— 4 vectorized column gathers + `matrixStats::rowMaxs/rowMins/rowMeans2` (C-level) |
| **Memory** | ~6.46M-element list of integer vectors + temporaries | One 6.46M Ã— 4 `nb_val_mat` (~207 MB) reused per variable |
| **Estimated time** | 86+ hours | **~1â€“5 minutes** |
| **Numerical result** | max, min, mean of rook-neighbor values | Identical â€” same `max`, `min`, `mean` over same neighbor sets |
| **Trained RF model** | â€” | Untouched; no retraining |

The key insight is that **the spatial neighbor topology is invariant across years**, so it should be encoded once as an integer matrix at the cell level (~344K rows), and then the year dimension is handled via arithmetic index computation â€” `(cell_pos - 1) * n_years + year_pos` â€” eliminating all string operations and per-row R function calls. The `matrixStats` package provides C-implemented row-wise summary functions that process the full 6.46M-row matrix in seconds.
END_RESPONSE>>>
