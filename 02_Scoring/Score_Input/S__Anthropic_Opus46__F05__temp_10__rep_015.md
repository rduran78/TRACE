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
run_id: Anthropic_F05_T10_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a much larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### The Core Problem: O(N) `lapply` with String Operations over 6.46M Rows

`build_neighbor_lookup` calls `lapply` over every row (`~6.46M` iterations). Inside each iteration it:

1. **Pastes** neighbor cell IDs with the current year to build string keys â€” `paste(neighbor_cell_ids, data$year[i], sep = "_")`.
2. **Looks up** those keys in a named character vector (`idx_lookup`) of length 6.46M.

Named-vector lookup in R is **O(n)** per lookup (linear scan or hash with overhead). With ~6.46M rows, each having ~4 rook neighbors on average (interior cells have 4, edge cells have 2â€“3), that's ~25.8M string constructions and ~25.8M named-vector lookups against a 6.46M-element vector. The string allocation and hashing overhead alone is enormous.

### Why It's a Broader Architectural Issue

The entire string-keying approach is unnecessary. The data is a **balanced panel** (344,208 cells Ã— 28 years). This means:

- Every cell appears in every year.
- Neighbors in year `t` are the same cells in year `t` â€” just at different row positions.
- If the data is sorted by `(year, id)` or `(id, year)`, neighbor row indices can be computed by **integer arithmetic** â€” no strings, no hash lookups.

The `compute_neighbor_stats` function is already vectorized once `neighbor_lookup` is built, so the bottleneck is entirely in `build_neighbor_lookup`.

### Quantified Impact

| Operation | Current | Optimized |
|---|---|---|
| String constructions | ~25.8M `paste()` calls | **0** |
| Hash lookups | ~25.8M named-vector lookups | **0** |
| Core index computation | String-based | **Integer arithmetic** |
| `build_neighbor_lookup` time | ~hours | **Seconds** |
| `compute_neighbor_stats` (Ã—5 vars) | Already vectorized | Further vectorizable with matrix ops |
| **Total estimated time** | **86+ hours** | **~2â€“5 minutes** |

---

## Optimization Strategy

### Strategy 1: Exploit Balanced Panel Structure with Integer Arithmetic

If data is sorted by `(year, id)` in a consistent order, then for year-block `t`, all cells appear at rows `((t-1)*N_cells + 1)` through `(t * N_cells)`, in the same `id_order`. A neighbor at position `j` in the id_order during year-block `t` is at row `(t-1)*N_cells + j`. No strings needed.

### Strategy 2: Vectorize `compute_neighbor_stats` with Matrix Column Indexing

Instead of `lapply` over 6.46M entries, we can build a neighbor-index matrix (padded to max neighbors), then use matrix subsetting to pull all neighbor values at once, and compute `max/min/mean` with `rowMeans`, `pmin`, `pmax` over columns.

### Strategy 3: Keep Everything Else Identical

The Random Forest model is already trained and takes the same column names â€” we only change how feature columns are computed, preserving exact numerical equivalence.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED FEATURE CONSTRUCTION
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns id, year, and predictor vars
#   - id_order: vector of unique cell IDs in the order used by the nb object
#   - rook_neighbors_unique: spdep::nb object (list of integer neighbor indices)
#   - The data must contain all combinations of id_order Ã— years (balanced panel)
# =============================================================================

library(data.table)

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # -------------------------------------------------------------------------
  # Convert to data.table for fast, controlled sorting
  # -------------------------------------------------------------------------
  dt <- as.data.table(data)
  
  # Build a mapping from cell id -> position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Add the spatial position index
  dt[, spatial_pos := id_to_pos[as.character(id)]]
  
  # Sort by (year, spatial_pos) so that within each year-block,
  # row i corresponds to spatial_pos i.
  setorder(dt, year, spatial_pos)
  
  # Record the permutation so we can map back to original row order later
  # We need to know: for each row in the ORIGINAL data, what row is it
  # in the sorted data?
  # We'll store the sorted order and the reverse mapping.
  
  N_cells <- length(id_order)
  years   <- sort(unique(dt$year))
  N_years <- length(years)
  
  stopifnot(nrow(dt) == N_cells * N_years)  # balanced panel check
  
  # In the sorted data, year-block t (0-indexed) spans rows
  #   (t * N_cells + 1) : ((t+1) * N_cells)
  # and within that block, row offset j corresponds to spatial_pos j.
  
  # -------------------------------------------------------------------------
  # Build padded neighbor matrix (N_cells Ã— max_neighbors)
  # Entry [i, k] = spatial_pos of k-th neighbor of cell i, or NA
  # -------------------------------------------------------------------------
  max_nb <- max(lengths(neighbors))
  
  # Pad neighbor lists to equal length
  nb_padded <- lapply(neighbors, function(x) {
    if (length(x) == 0) return(rep(NA_integer_, max_nb))
    c(as.integer(x), rep(NA_integer_, max_nb - length(x)))
  })
  nb_matrix <- do.call(rbind, nb_padded)  # N_cells Ã— max_nb
  # nb_matrix[i, k] = spatial_pos of k-th neighbor of cell at spatial_pos i
  
  # -------------------------------------------------------------------------
  # Build the full row-index neighbor lookup for the SORTED data
  # For sorted row r, its spatial_pos = ((r-1) %% N_cells) + 1
  # and its year-block offset = ((r-1) %/% N_cells) * N_cells
  # Neighbor sorted-rows = year_block_offset + nb_matrix[spatial_pos, ]
  # -------------------------------------------------------------------------
  
  # We'll return:
  #   1) The sorted data.table (to be used for feature computation)
  #   2) The neighbor matrix in terms of sorted-row indices (N_rows Ã— max_nb)
  #   3) A mapping to restore original row order
  
  # Compute neighbor row indices for ALL sorted rows at once (vectorized)
  spatial_pos_all   <- rep(seq_len(N_cells), times = N_years)
  year_block_offset <- rep(seq(0L, (N_years - 1L) * N_cells, by = N_cells),
                           each = N_cells)
  
  # nb_matrix[spatial_pos_all, ] gives an (N_rows Ã— max_nb) matrix
  # of neighbor spatial positions. Add year_block_offset to get sorted row idx.
  neighbor_row_matrix <- nb_matrix[spatial_pos_all, , drop = FALSE] +
                         year_block_offset
  # Where nb_matrix had NA, the result is NA (NA + integer = NA). Good.
  
  # -------------------------------------------------------------------------
  # Store original row indices for restoring order
  # dt was reordered; we need to map back.
  # Before sorting, we should have saved the original row index.
  # -------------------------------------------------------------------------
  
  list(
    sorted_dt            = dt,
    neighbor_row_matrix  = neighbor_row_matrix,   # N_rows Ã— max_nb (sorted-row indices)
    max_nb               = max_nb,
    N_cells              = N_cells,
    N_years              = N_years
  )
}


compute_neighbor_stats_fast <- function(vals, neighbor_row_matrix) {
  # -------------------------------------------------------------------------
  # vals: numeric vector of length N_rows (in sorted order)
  # neighbor_row_matrix: integer matrix N_rows Ã— max_nb
  # Returns: N_rows Ã— 3 matrix with columns max, min, mean
  # -------------------------------------------------------------------------
  
  max_nb <- ncol(neighbor_row_matrix)
  N_rows <- length(vals)
  
  # Build a matrix of neighbor values: N_rows Ã— max_nb
  # Use vals[neighbor_row_matrix] â€” this is a single vectorized index operation
  nb_vals <- matrix(vals[neighbor_row_matrix], nrow = N_rows, ncol = max_nb)
  # Where neighbor_row_matrix is NA, nb_vals is NA. Correct.
  
  # Compute row-wise max, min, mean ignoring NAs
  # For large matrices, rowwise operations are efficient in R / can use matrixStats
  
  # Check if matrixStats is available for speed; otherwise use base
  use_matrixStats <- requireNamespace("matrixStats", quietly = TRUE)
  
  if (use_matrixStats) {
    row_max  <- matrixStats::rowMaxs(nb_vals,  na.rm = TRUE)
    row_min  <- matrixStats::rowMins(nb_vals,  na.rm = TRUE)
    row_mean <- matrixStats::rowMeans2(nb_vals, na.rm = TRUE)
  } else {
    row_max  <- apply(nb_vals, 1, max,  na.rm = TRUE)
    row_min  <- apply(nb_vals, 1, min,  na.rm = TRUE)
    row_mean <- rowMeans(nb_vals, na.rm = TRUE)
  }
  
  # Handle rows where ALL neighbors are NA -> should return NA
  all_na <- rowSums(!is.na(nb_vals)) == 0L
  row_max[all_na]  <- NA_real_
  row_min[all_na]  <- NA_real_
  row_mean[all_na] <- NA_real_
  
  # Fix -Inf/Inf from max/min on empty sets (if matrixStats returns them)
  row_max[is.infinite(row_max)]  <- NA_real_
  row_min[is.infinite(row_min)]  <- NA_real_
  
  cbind(nb_max = row_max, nb_min = row_min, nb_mean = row_mean)
}


compute_and_add_neighbor_features_fast <- function(dt, var_name,
                                                   neighbor_row_matrix) {
  vals   <- dt[[var_name]]
  stats  <- compute_neighbor_stats_fast(vals, neighbor_row_matrix)
  
  dt[, paste0(var_name, "_nb_max")  := stats[, 1]]
  dt[, paste0(var_name, "_nb_min")  := stats[, 2]]
  dt[, paste0(var_name, "_nb_mean") := stats[, 3]]
  
  invisible(dt)
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

run_optimized_feature_construction <- function(cell_data, id_order,
                                               rook_neighbors_unique) {
  
  cat("Building fast neighbor lookup (integer arithmetic, no strings)...\n")
  t0 <- proc.time()
  
  # Save original row order for later restoration
  cell_data$.orig_row_idx <- seq_len(nrow(cell_data))
  
  lookup <- build_neighbor_lookup_fast(cell_data, id_order,
                                       rook_neighbors_unique)
  
  sorted_dt           <- lookup$sorted_dt
  neighbor_row_matrix <- lookup$neighbor_row_matrix
  
  t1 <- proc.time()
  cat(sprintf("  Neighbor lookup built in %.1f seconds.\n", (t1 - t0)[3]))
  
  # Compute neighbor features for each source variable
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Computing neighbor stats for '%s'...\n", var_name))
    tv0 <- proc.time()
    compute_and_add_neighbor_features_fast(sorted_dt, var_name,
                                           neighbor_row_matrix)
    tv1 <- proc.time()
    cat(sprintf("    Done in %.1f seconds.\n", (tv1 - tv0)[3]))
  }
  
  # Restore original row order
  cat("Restoring original row order...\n")
  setorder(sorted_dt, .orig_row_idx)
  sorted_dt[, c("spatial_pos", ".orig_row_idx") := NULL]
  
  t2 <- proc.time()
  cat(sprintf("Total feature construction time: %.1f seconds.\n", (t2 - t0)[3]))
  
  # Return as data.frame if the original was a data.frame
  return(sorted_dt[])
}


# =============================================================================
# USAGE (drop-in replacement for the original outer loop)
# =============================================================================
# cell_data <- run_optimized_feature_construction(
#   cell_data, id_order, rook_neighbors_unique
# )
#
# # Then predict with existing trained Random Forest as before:
# # predictions <- predict(rf_model, cell_data)
# =============================================================================
```

---

## Why This Preserves Numerical Equivalence

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | `nb_matrix[i, ]` contains exactly the same neighbor spatial positions as `neighbors[[i]]`. The year-block offset maps them to the correct year's rows â€” identical to the string-key join. |
| **Same aggregation** | `max`, `min`, `mean` with `na.rm = TRUE` over the same set of neighbor values produces identical results. |
| **Same output columns** | Column names follow the same `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` pattern. |
| **Original row order restored** | The `.orig_row_idx` column ensures the final output matches the input row order exactly. |
| **RF model untouched** | No model retraining; only feature column computation is changed. |

## Performance Summary

| Step | Original | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~hours (string paste + hash Ã—6.46M) | ~5â€“15 sec (integer arithmetic, vectorized) | ~1000Ã— |
| Stats for 1 variable | ~minutes (lapply over 6.46M) | ~3â€“8 sec (matrix indexing + `matrixStats`) | ~100Ã— |
| Stats for 5 variables | ~hours | ~15â€“40 sec | ~100Ã— |
| **Total** | **86+ hours** | **~1â€“3 minutes** | **~2000Ã—** |

**Memory**: The `neighbor_row_matrix` is `6.46M Ã— 4 integers â‰ˆ 103 MB`. Each `nb_vals` column matrix is `6.46M Ã— 4 doubles â‰ˆ 207 MB`. With 16 GB RAM this fits comfortably, even with the data.table copy overhead.
END_RESPONSE>>>
