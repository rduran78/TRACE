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
run_id: Anthropic_F05_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local Inefficiency
Inside `build_neighbor_lookup`, for every one of the ~6.46 million rows, the code:
1. Calls `paste()` to build `neighbor_keys` (string allocation + concatenation).
2. Does a named-vector lookup (`idx_lookup[neighbor_keys]`), which is an O(n) hash probe per key, repeated across all rows.

This is expensive, but the **real** problem is structural.

### Broader Architectural Problem

1. **`build_neighbor_lookup` is an O(N Ã— K) `lapply` over 6.46M rows**, each doing string operations and hash lookups. With an average of ~4 rook neighbors per cell, that's ~26 million string constructions and hash probes â€” all in an interpreted R loop.

2. **The neighbor topology is static across years.** Rook neighbors don't change from year to year. Yet the code re-discovers the same spatial relationships for every year by encoding `(id, year)` pairs as strings. The correct mental model is: *within each year, the neighbor graph is identical*. This means you can compute neighbor statistics as a **vectorized within-year operation** using matrix indexing, eliminating all string work and the per-row `lapply` entirely.

3. **`compute_neighbor_stats` then loops over the same 6.46M rows again**, once per variable. Five variables Ã— 6.46M = 32.3M R-level loop iterations, each subsetting a numeric vector.

### Root Cause
The algorithm treats a **regular-grid spatial panel** as if neighbor relationships must be discovered row-by-row. In reality, the neighbor graph is a fixed sparse adjacency that can be applied to an entire year's data via sparse matrixâ€“dense matrix multiplication (or equivalent vectorized operations).

---

## Optimization Strategy

1. **Convert the `nb` object to a sparse adjacency matrix once.** This is a one-time cost.

2. **Reshape each variable into a (cells Ã— years) matrix**, where rows are grid cells in fixed order and columns are years.

3. **Compute neighbor sums, counts, max, and min using sparse matrix operations or vectorized column-wise grouped operations.** For mean: `neighbor_mean = (W %*% X) / (W %*% 1_valid)` where `W` is the row-standardized (or binary) sparse weight matrix and validity masking handles NAs. Max and min require a grouped approach but can be vectorized per-year.

4. **Unstack back to the long panel.**

This replaces ~6.46M interpreted iterations with ~28 sparse matrix multiplications per variable (one per year), all in optimized C/FORTRAN under the hood.

**Expected speedup:** From 86+ hours to minutes.

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature construction
# Preserves the exact numerical estimand (max, min, mean of non-NA neighbors)
# Preserves the trained Random Forest model (no retraining)
# =============================================================================

library(Matrix)   # for sparse matrices
library(spdep)    # for nb2listw / nb utilities (already used)
library(data.table)

# -------------------------------------------------------------------------
# STEP 0: One-time conversion of nb object to sparse binary adjacency matrix
# -------------------------------------------------------------------------
# id_order:              integer vector of cell IDs in the order used by rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

build_sparse_adjacency <- function(id_order, neighbors_nb) {
  n <- length(id_order)
  # Build COO triplets
  from <- integer(0)
  to   <- integer(0)
  for (i in seq_len(n)) {
    nb_i <- neighbors_nb[[i]]
    # spdep nb objects use 0L to denote "no neighbors"
    nb_i <- nb_i[nb_i > 0L]
    if (length(nb_i) > 0L) {
      from <- c(from, rep(i, length(nb_i)))
      to   <- c(to, nb_i)
    }
  }
  W <- sparseMatrix(i = from, j = to, x = 1, dims = c(n, n))
  W
}

# -------------------------------------------------------------------------
# STEP 1: Vectorized neighbor stats (max, min, mean) for one variable
# -------------------------------------------------------------------------
# This function takes:
#   dt        : data.table with columns id, year, and <var_name>
#   id_order  : integer vector of all cell IDs in adjacency-matrix row order
#   W         : sparse binary adjacency matrix (n_cells x n_cells)
#   var_name  : character, name of the variable
#
# Returns: dt with three new columns appended:
#   <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean

compute_neighbor_features_vectorized <- function(dt, id_order, W, var_name) {

  n_cells <- length(id_order)

  # --- Map cell IDs to matrix row indices (fixed order) ---
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))

  # --- Ensure data.table and keyed ---
  if (!is.data.table(dt)) dt <- as.data.table(dt)

  # --- Add matrix row index ---
  dt[, .row_idx := id_to_row[as.character(id)]]

  # --- Get unique sorted years ---
  years <- sort(unique(dt$year))

  # --- Pre-allocate result columns ---
  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]

  # --- For max and min, we need the actual neighbor list from W ---
  # Extract neighbor indices once (same structure as nb, but guaranteed consistent)
  # We use the sparse matrix column pointers for efficiency
  W_t <- t(W)  # transpose so that column j of W_t gives neighbors of cell j
  # Actually, for row i of W, nonzero columns are neighbors of i.
  # Let's extract neighbor lists from W directly:
  neighbor_list <- vector("list", n_cells)
  # Use the dgCMatrix structure of W
  Wc <- as(W, "dgCMatrix")
  for (i in seq_len(n_cells)) {
    # For row i of a dgCMatrix, we need to find nonzero entries.
    # It's faster to work with the transpose (dgCMatrix is column-oriented)
    # So we transpose and read columns.
    NULL
  }
  # More efficient: convert to dgRMatrix (row-oriented) or use summary()
  W_summary <- summary(W)  # returns data.frame with i, j, x columns
  # Split j by i
  neighbor_list <- split(W_summary$j, W_summary$i)
  # Fill in empty cells
  all_rows <- as.character(seq_len(n_cells))
  missing_rows <- setdiff(all_rows, names(neighbor_list))
  for (m in missing_rows) {
    neighbor_list[[m]] <- integer(0)
  }
  # Ensure numeric indexing
  neighbor_list_vec <- vector("list", n_cells)
  for (k in seq_len(n_cells)) {
    val <- neighbor_list[[as.character(k)]]
    neighbor_list_vec[[k]] <- if (is.null(val)) integer(0) else as.integer(val)
  }
  neighbor_list <- neighbor_list_vec
  rm(neighbor_list_vec, W_summary, Wc)

  # --- Process year-by-year (vectorized within each year) ---
  for (yr in years) {

    # Subset rows for this year
    yr_mask <- which(dt$year == yr)

    # Build a full-length vector for this variable aligned to matrix rows
    # (NA for cells not present in this year)
    vals_full <- rep(NA_real_, n_cells)
    rows_present <- dt$.row_idx[yr_mask]
    var_vals     <- dt[[var_name]][yr_mask]
    vals_full[rows_present] <- var_vals

    # --- MEAN via sparse matrix multiplication ---
    # Replace NA with 0 for summation, track valid counts
    valid_mask <- as.numeric(!is.na(vals_full))   # 1 if valid, 0 if NA
    vals_zero  <- vals_full
    vals_zero[is.na(vals_zero)] <- 0

    neighbor_sum   <- as.numeric(W %*% vals_zero)    # sum of neighbor values
    neighbor_count <- as.numeric(W %*% valid_mask)    # count of non-NA neighbors

    neighbor_mean_full <- ifelse(neighbor_count > 0,
                                 neighbor_sum / neighbor_count,
                                 NA_real_)

    # --- MAX and MIN via vectorized lapply over cells present this year ---
    # Only compute for cells actually in the data this year
    unique_rows_present <- unique(rows_present)

    # Pre-extract vals_full for speed (it's already a plain vector)
    max_full <- rep(NA_real_, n_cells)
    min_full <- rep(NA_real_, n_cells)

    # Vectorized batch: for each present cell, get neighbor values
    for (ci in unique_rows_present) {
      nb_idx <- neighbor_list[[ci]]
      if (length(nb_idx) == 0L) next
      nb_vals <- vals_full[nb_idx]
      nb_vals <- nb_vals[!is.na(nb_vals)]
      if (length(nb_vals) == 0L) next
      max_full[ci] <- max(nb_vals)
      min_full[ci] <- min(nb_vals)
    }

    # --- Write results back to dt ---
    set(dt, i = yr_mask, j = col_max,  value = max_full[rows_present])
    set(dt, i = yr_mask, j = col_min,  value = min_full[rows_present])
    set(dt, i = yr_mask, j = col_mean, value = neighbor_mean_full[rows_present])
  }

  # Clean up temporary column
  dt[, .row_idx := NULL]

  return(dt)
}

# -------------------------------------------------------------------------
# STEP 2: Further optimize max/min with pre-sorted neighbor value approach
# The inner loop above for max/min is still O(cells_per_year Ã— avg_neighbors).
# With ~344K cells and ~4 neighbors avg, that's ~1.4M ops per year, 28 years
# = ~39M ops total per variable. In R this is ~seconds, not hours.
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------

# Build adjacency matrix once
W <- build_sparse_adjacency(id_order, rook_neighbors_unique)

# Convert to data.table for efficiency
cell_dt <- as.data.table(cell_data)

# Neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Compute all neighbor features
for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  t0 <- Sys.time()
  cell_dt <- compute_neighbor_features_vectorized(cell_dt, id_order, W, var_name)
  cat("  Done in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")
}

# Convert back to data.frame if needed downstream
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is untouched â€” 
# these columns have identical names and identical numerical values
# as the original pipeline would produce.
```

---

## Further Optimization: Eliminate the Per-Cell Max/Min Loop Entirely

The remaining inner loop for max/min can be eliminated with a **long-form edge-table approach**:

```r
# -------------------------------------------------------------------------
# FULLY VECTORIZED max/min/mean â€” no R-level per-cell loops at all
# -------------------------------------------------------------------------

compute_neighbor_features_fully_vectorized <- function(dt, id_order, W, var_name) {

  if (!is.data.table(dt)) dt <- as.data.table(dt)

  n_cells <- length(id_order)
  id_to_row <- setNames(seq_along(id_order), as.character(id_order))
  dt[, .row_idx := id_to_row[as.character(id)]]

  # --- Build edge table from W (one-time per call, could also be passed in) ---
  W_summ <- summary(W)  # i, j, x
  edges <- data.table(from = W_summ$i, to = W_summ$j)  # "from" has neighbor "to"

  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")

  dt[, (col_max)  := NA_real_]
  dt[, (col_min)  := NA_real_]
  dt[, (col_mean) := NA_real_]

  years <- sort(unique(dt$year))

  for (yr in years) {

    yr_mask <- which(dt$year == yr)

    # Build lookup: matrix_row -> variable value (for cells present this year)
    vals_full <- rep(NA_real_, n_cells)
    vals_full[dt$.row_idx[yr_mask]] <- dt[[var_name]][yr_mask]

    # --- Join neighbor values via edge table (fully vectorized) ---
    # For each edge (from, to), attach the value at "to"
    edge_vals <- data.table(
      from   = edges$from,
      nb_val = vals_full[edges$to]
    )

    # Drop edges where neighbor value is NA
    edge_vals <- edge_vals[!is.na(nb_val)]

    # Aggregate by 'from' cell
    agg <- edge_vals[, .(
      nb_max  = max(nb_val),
      nb_min  = min(nb_val),
      nb_mean = mean(nb_val)
    ), by = from]

    # Map aggregated results back to full vector
    max_full  <- rep(NA_real_, n_cells)
    min_full  <- rep(NA_real_, n_cells)
    mean_full <- rep(NA_real_, n_cells)

    max_full[agg$from]  <- agg$nb_max
    min_full[agg$from]  <- agg$nb_min
    mean_full[agg$from] <- agg$nb_mean

    # Write back
    rows_present <- dt$.row_idx[yr_mask]
    set(dt, i = yr_mask, j = col_max,  value = max_full[rows_present])
    set(dt, i = yr_mask, j = col_min,  value = min_full[rows_present])
    set(dt, i = yr_mask, j = col_mean, value = mean_full[rows_present])
  }

  dt[, .row_idx := NULL]
  return(dt)
}

# -------------------------------------------------------------------------
# MAIN EXECUTION (fully vectorized version)
# -------------------------------------------------------------------------

W <- build_sparse_adjacency(id_order, rook_neighbors_unique)
cell_dt <- as.data.table(cell_data)

# Pre-compute edge table once (shared across all variables)
W_summ <- summary(W)
edges  <- data.table(from = W_summ$i, to = W_summ$j)

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "\n")
  t0 <- Sys.time()
  cell_dt <- compute_neighbor_features_fully_vectorized(cell_dt, id_order, W, var_name)
  cat("  Done in", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")
}

cell_data <- as.data.frame(cell_dt)
```

---

## Complexity Comparison

| | Original | Optimized |
|---|---|---|
| **String constructions** | ~6.46M Ã— (1 + avg_neighbors) â‰ˆ 32M | **0** |
| **Hash lookups** | ~26M named-vector probes | **0** |
| **R-level loop iterations** | 6.46M (build) + 6.46M Ã— 5 (stats) = ~38.8M | **28 Ã— 5 = 140** year-variable passes |
| **Core operation per year** | Interpreted per-row loop | data.table grouped aggregation on ~1.4M edge rows |
| **Estimated wall time** | 86+ hours | **~5â€“15 minutes** |
| **RAM overhead** | Moderate (string vectors) | ~edges DT â‰ˆ 1.4M Ã— 2 cols â‰ˆ 22 MB + sparse W â‰ˆ 33 MB |
| **Numerical output** | Reference | **Identical** (same max, min, mean of non-NA rook neighbors) |
| **Trained RF model** | â€” | **Untouched** |
END_RESPONSE>>>
