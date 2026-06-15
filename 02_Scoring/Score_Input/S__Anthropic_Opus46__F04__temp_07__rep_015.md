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
run_id: Anthropic_F04_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String concatenation (`paste`), named-vector lookups by character key, and per-row `lapply` overhead repeated ~6.46 million times are extremely expensive in R. The `compute_neighbor_stats` function is a secondary bottleneck: it also loops over 6.46M entries in R-level `lapply`, computing `max/min/mean` with NA handling per iteration.

**Root causes, ranked by impact:**

1. **Row-level `lapply` in `build_neighbor_lookup`**: 6.46M R-level iterations with string operations and named-vector lookups (effectively O(n) hash lookups on large character vectors) dominate wall time.
2. **Row-level `lapply` in `compute_neighbor_stats`**: Another 6.46M R-level iterations per variable (Ã—5 variables = 32.3M iterations).
3. **Redundant work across years**: The neighbor *structure* is purely spatial (rook contiguity) and identical for every year, yet the lookup is rebuilt monolithically by string-keying `id_year`.
4. **No vectorization or use of sparse-matrix algebra**: The aggregation (mean, max, min of neighbors) is a classic sparse-matrixâ€“vector product pattern that can be fully vectorized.

## Optimization Strategy

**Core idea:** Represent the spatial neighbor graph as a sparse matrix **once**, then compute neighbor statistics for all cell-years simultaneously via vectorized sparse-matrix operations â€” eliminating all R-level row loops.

- **Step A**: Build a sparse adjacency matrix `W` of dimension `N_cells Ã— N_cells` from `rook_neighbors_unique` (344,208 Ã— 344,208, ~1.37M non-zeros â€” trivially small).
- **Step B**: Ensure `cell_data` is sorted by `(id, year)` so that all years for cell *i* occupy a contiguous block, and build a block-diagonal sparse matrix `W_panel` of dimension `N_rows Ã— N_rows` that replicates `W` identically for each year. This turns the 6.46M-row panel neighbor lookup into a single sparse structure.
- **Step C**: For each variable, compute `neighbor_mean` as a sparse matrixâ€“vector multiply (`W_panel %*% x / row_counts`), and `neighbor_max` / `neighbor_min` via a grouped operation over the sparse structure.

This replaces ~38.7M R-level loop iterations with a handful of vectorized C-level sparse operations. Expected speedup: **several hundred to a thousand fold** (minutes instead of 86+ hours).

## Working R Code

```r
# =============================================================================
# Optimized spatial neighbor feature construction
# Preserves the trained RF model and original numerical estimand exactly.
# =============================================================================

library(Matrix)   # for sparse matrices
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars) {

  # --------------------------------------------------------------------------
  # 0.  Convert to data.table for speed; record original row order
  # --------------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .orig_row := .I]

  # Ensure id and year columns exist
  stopifnot(all(c("id", "year") %in% names(dt)))

  # --------------------------------------------------------------------------
  # 1.  Build spatial sparse adjacency matrix  W  (N_cells x N_cells)
  #     from the spdep nb object.  id_order[i] is the cell id for nb index i.
  # --------------------------------------------------------------------------
  N_cells <- length(id_order)
  # Map cell id -> matrix row/col index (1-based, matching id_order position)
  cell_id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Construct COO triplets from the nb list
  from_idx <- integer(0)
  to_idx   <- integer(0)
  for (i in seq_along(rook_neighbors_unique)) {
    nb_i <- rook_neighbors_unique[[i]]
    # spdep nb encodes "no neighbours" as a single 0L
    if (length(nb_i) == 1L && nb_i == 0L) next
    from_idx <- c(from_idx, rep.int(i, length(nb_i)))
    to_idx   <- c(to_idx,   nb_i)
  }
  W <- sparseMatrix(i = from_idx, j = to_idx, x = 1,
                    dims = c(N_cells, N_cells))

  message(sprintf("Spatial adjacency matrix: %d cells, %d directed edges",
                  N_cells, length(from_idx)))

  # --------------------------------------------------------------------------
  # 2.  Sort panel data by (id, year) and build index mappings
  # --------------------------------------------------------------------------
  setkey(dt, id, year)
  unique_years <- sort(unique(dt$year))
  N_years      <- length(unique_years)
  year_to_slot <- setNames(seq_along(unique_years), as.character(unique_years))

  # For each row, find the spatial index of its cell id
  dt[, sp_idx := cell_id_to_idx[as.character(id)]]

  # --------------------------------------------------------------------------
  # 3.  Build block-diagonal panel adjacency matrix  W_panel
  #     Dimension: N_rows x N_rows

  #     Block t replicates W for year t, connecting only rows within that year.
  #
  #     Instead of literally constructing the huge block-diagonal, we work
  #     year-by-year (much more memory-friendly on a 16 GB laptop).
  # --------------------------------------------------------------------------

  # Pre-split row indices by year for fast access
  dt[, panel_row := .I]   # current row index after sorting
  year_rows <- split(dt$panel_row, dt$year)   # list keyed by year
  year_sp   <- split(dt$sp_idx,   dt$year)   # spatial indices per year

  # --------------------------------------------------------------------------
  # 4.  For each source variable, compute neighbor max, min, mean
  #     year-by-year using sparse matrix operations.
  # --------------------------------------------------------------------------

  compute_all_neighbor_stats <- function(dt, var_name, W, year_rows, year_sp,
                                         N_cells) {
    vals_all <- dt[[var_name]]
    n_rows   <- nrow(dt)

    col_mean <- rep(NA_real_, n_rows)
    col_max  <- rep(NA_real_, n_rows)
    col_min  <- rep(NA_real_, n_rows)

    for (yr_char in names(year_rows)) {
      rows_yr <- year_rows[[yr_char]]    # panel row indices for this year
      sp_yr   <- year_sp[[yr_char]]      # spatial indices (into W) for this year

      # Build a mapping: spatial_index -> position within this year's subset
      # Not all cells appear in every year, so we need the actual subset.
      n_yr <- length(rows_yr)

      # Values for this year (in spatial-index order of sp_yr)
      x <- vals_all[rows_yr]

      # We need the sub-matrix of W corresponding to cells present this year.
      # W_sub[i,j] = 1 iff cell sp_yr[j] is a rook neighbor of cell sp_yr[i].
      W_sub <- W[sp_yr, sp_yr, drop = FALSE]

      # --- Handle NAs: set NA values to 0 for summation, track validity ---
      not_na <- as.numeric(!is.na(x))   # 1 if valid, 0 if NA
      x_safe <- ifelse(is.na(x), 0, x)

      # Neighbor count (of non-NA neighbors)
      n_valid <- as.numeric(W_sub %*% not_na)

      # Neighbor sum  (of non-NA neighbors)
      n_sum <- as.numeric(W_sub %*% x_safe)

      # Neighbor mean
      yr_mean <- ifelse(n_valid > 0, n_sum / n_valid, NA_real_)

      # --- Neighbor max and min via row-wise operations on sparse matrix ---
      # For max/min we iterate over the sparse structure efficiently.
      # Extract the triplet form of W_sub.
      W_sub_T <- as(W_sub, "TsparseMatrix")   # 0-based i, j
      wi <- W_sub_T@i + 1L   # 1-based row
      wj <- W_sub_T@j + 1L   # 1-based col

      # Neighbor values corresponding to each edge
      edge_vals <- x[wj]

      # Remove edges pointing to NA values
      valid_edge <- !is.na(edge_vals)
      wi_v <- wi[valid_edge]
      edge_v <- edge_vals[valid_edge]

      # Compute grouped max and min using data.table for speed
      if (length(wi_v) > 0) {
        edge_dt <- data.table(row = wi_v, val = edge_v)
        agg <- edge_dt[, .(mx = max(val), mn = min(val)), by = row]
        yr_max <- rep(NA_real_, n_yr)
        yr_min <- rep(NA_real_, n_yr)
        yr_max[agg$row] <- agg$mx
        yr_min[agg$row] <- agg$mn
      } else {
        yr_max <- rep(NA_real_, n_yr)
        yr_min <- rep(NA_real_, n_yr)
      }

      col_mean[rows_yr] <- yr_mean
      col_max[rows_yr]  <- yr_max
      col_min[rows_yr]  <- yr_min
    }

    list(mean = col_mean, max = col_max, min = col_min)
  }

  # --------------------------------------------------------------------------
  # 5.  Main loop: compute and attach features for each source variable
  # --------------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_all_neighbor_stats(dt, var_name, W, year_rows, year_sp,
                                        N_cells)
    # Use the same column naming convention as the original pipeline
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = stats$max)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = stats$min)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = stats$mean)
  }

  # --------------------------------------------------------------------------
  # 6.  Restore original row order and return as data.frame
  # --------------------------------------------------------------------------
  setorder(dt, .orig_row)
  dt[, c(".orig_row", "panel_row", "sp_idx") := NULL]

  return(as.data.frame(dt))
}


# =============================================================================
# Usage  (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predictions proceed exactly as before:
# preds <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

## Why This Is Correct and Equivalent

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `mean` is computed as `sum / count` over the identical set of non-NA rook neighbors. `max` and `min` are computed over the same set. Sparse-matrix indexing `W[sp_yr, sp_yr]` reproduces exactly the neighbor sets from the `nb` object. |
| **NA handling** | Identical: NAs are excluded before aggregation; if all neighbors are NA (or no neighbors exist), the result is `NA`. |
| **Trained RF model** | Untouched. The code only reconstructs the 15 neighbor-derived feature columns (5 vars Ã— 3 stats) with identical names and values. |
| **Column naming** | Uses `paste0(var_name, "_neighbor_max/min/mean")` â€” adjust the suffix pattern if your original `compute_and_add_neighbor_features` uses a different convention. |

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level iterations with string ops | Eliminated entirely |
| `compute_neighbor_stats` (per variable) | ~6.46M R-level iterations | 28 sparse matrixâ€“vector multiplies (~230K rows each) + one grouped `data.table` aggregation per year |
| **Total neighbor feature time** | **~86+ hours** | **~5â€“15 minutes** (estimated) |
| Peak memory | Moderate (large list of 6.46M integer vectors) | ~1â€“2 GB (sparse matrix ~11 MB; year subsets ~230K rows; temporary edge `data.table`) |

The dominant cost becomes the 28 Ã— 5 = 140 sparse sub-matrix extractions and multiplies, each on ~230K Ã— 230K matrices with ~49K non-zeros per row on average â€” a trivial workload for `Matrix` package routines implemented in C.
END_RESPONSE>>>
