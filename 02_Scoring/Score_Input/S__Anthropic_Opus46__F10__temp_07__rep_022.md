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
run_id: Anthropic_F10_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F10_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **`build_neighbor_lookup` creates ~6.46 million list entries**, each requiring string-pasting, hash lookups via named vectors, and NA filtering. The `paste(..., sep="_")` key construction and `idx_lookup[neighbor_keys]` named-vector lookup are O(n) string operations repeated per row â€” totaling billions of character operations.

2. **`compute_neighbor_stats` iterates over 6.46 million list elements in an R-level `lapply`**, extracting subsets of a numeric vector per element. This is called 5 times (once per variable), yielding ~32.3 million R-level loop iterations with per-element subsetting, NA removal, and summary computation â€” all interpreted.

3. **The neighbor topology is year-invariant but the lookup is rebuilt as if it were year-specific.** Rook contiguity depends only on spatial position, not time. The current code re-resolves neighbor relationships per cell-year row instead of exploiting the fact that the same adjacency structure repeats identically across all 28 years.

**Root cause summary:** The algorithm is O(rows Ã— avg_neighbors) executed in interpreted R with string-key indirection, when it should be a sparse matrixâ€“vector multiplication executed in compiled C/C++ code.

---

## Optimization Strategy

### Core Insight
Neighbor-mean is a sparse-matrixâ€“vector product. Neighbor-max and neighbor-min are sparse row-wise aggregations. All three can be computed per-year by reusing a single sparse adjacency matrix built once from the `nb` object.

### Steps

1. **Build a single sparse adjacency matrix `W` (344,208 Ã— 344,208)** from `rook_neighbors_unique` using `spdep::nb2listw` â†’ `as_dgRMatrix_listw` or direct construction via `Matrix::sparseMatrix`. This is built **once**.

2. **For each year**, extract the column vector `x` of length 344,208 for each variable. Then:
   - **Mean**: `W %*% x / degree` (sparse mat-vec, microseconds for 344K nodes).
   - **Max / Min**: Use row-wise sparse aggregation. With the `Matrix` package, iterate over rows of `W` using the CSR structure, or â€” more practically â€” use `data.table` grouped aggregation on the edge list.

3. **Avoid string keys entirely.** Map cell IDs to integer indices once; use integer indexing throughout.

4. **Vectorize across years** using `data.table` split-apply-combine or a simple `for` loop over 28 years (trivial overhead when inner work is compiled).

### Expected Speedup
- Sparse mat-vec for mean: ~0.01s per variable-year â†’ 1.4s total for means.
- Grouped `data.table` aggregation for max/min: ~0.1s per variable-year â†’ 14s total.
- Total: **under 1 minute** vs. 86+ hours. Approximately **5,000Ã— speedup**.

---

## Optimized R Code

```r
# ==============================================================================
# Optimized Neighbor Feature Engineering
# Preserves numerical equivalence with original compute_neighbor_stats output.
# ==============================================================================

library(Matrix)
library(data.table)

# --------------------------------------------------------------------------
# STEP 1: Build sparse adjacency matrix ONCE from the nb object
# --------------------------------------------------------------------------

build_sparse_adjacency <- function(id_order, nb_obj) {
  # nb_obj is a list of length N where nb_obj[[i]] contains integer indices
  # of neighbors of node i (in id_order's positional space).
  # id_order is the vector of cell IDs in the order matching nb_obj.

  n <- length(id_order)
  stopifnot(length(nb_obj) == n)

  # Build COO triplets (row, col) for the adjacency matrix
  # nb objects use 0L to indicate no neighbors
  from <- rep(seq_len(n), times = vapply(nb_obj, function(x) {
    sum(x > 0L)
  }, integer(1)))

  to <- unlist(lapply(nb_obj, function(x) x[x > 0L]), use.names = FALSE)

  # Sparse binary adjacency matrix (rows = focal node, cols = neighbor node)
  W <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n, n),
    repr = "C"   # CSR format for fast row operations
  )

  # Degree vector (number of neighbors per node)
  degree <- diff(W@p)  # For dgCMatrix; for dgRMatrix, use rowSums
  # Actually, let's ensure we use dgCMatrix and compute degree via rowSums
  W <- as(W, "CsparseMatrix")
  degree <- as.integer(rowSums(W))

  list(W = W, degree = degree, n = n, id_order = id_order)
}

# --------------------------------------------------------------------------
# STEP 2: Build edge list (for max/min via data.table grouped ops)
# --------------------------------------------------------------------------

build_edge_dt <- function(nb_obj) {
  n <- length(nb_obj)
  from <- rep(seq_len(n), times = vapply(nb_obj, function(x) {
    sum(x > 0L)
  }, integer(1)))
  to <- unlist(lapply(nb_obj, function(x) x[x > 0L]), use.names = FALSE)
  data.table(from = from, to = to)
}

# --------------------------------------------------------------------------
# STEP 3: Compute neighbor stats for all variables, all years
# --------------------------------------------------------------------------

compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {

  # Convert to data.table if not already
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Build topology once ---
  cat("Building sparse adjacency structure...\n")
  adj    <- build_sparse_adjacency(id_order, nb_obj)
  W      <- adj$W
  degree <- adj$degree
  n      <- adj$n

  edge_dt <- build_edge_dt(nb_obj)

  # --- Build spatial index mapping: cell_id -> positional index (1..n) ---
  id_to_pos <- setNames(seq_len(n), as.character(id_order))

  # Add positional spatial index to cell_data
  cell_data[, spatial_idx := id_to_pos[as.character(id)]]

  # Verify all cells are mapped
  stopifnot(!anyNA(cell_data$spatial_idx))

  # --- Get sorted unique years ---
  years <- sort(unique(cell_data$year))
  cat(sprintf("Processing %d variables Ã— %d years = %d batches\n",
              length(neighbor_source_vars), length(years),
              length(neighbor_source_vars) * length(years)))

  # --- Pre-allocate output columns ---
  for (var_name in neighbor_source_vars) {
    col_max  <- paste0("neighbor_max_", var_name)
    col_min  <- paste0("neighbor_min_", var_name)
    col_mean <- paste0("neighbor_mean_", var_name)
    cell_data[, (col_max)  := NA_real_]
    cell_data[, (col_min)  := NA_real_]
    cell_data[, (col_mean) := NA_real_]
  }

  # --- Key the data for fast subsetting ---
  setkey(cell_data, year, spatial_idx)

  # --- Main loop: year Ã— variable ---
  for (yr in years) {
    # Extract this year's slice, ordered by spatial_idx
    yr_rows <- cell_data[.(yr)]  # keyed lookup
    # Ensure ordering by spatial_idx
    setorder(yr_rows, spatial_idx)

    # Verify complete spatial coverage for this year
    # (If some cells are missing in some years, we must handle gaps)
    yr_spatial_idx <- yr_rows$spatial_idx
    n_yr <- nrow(yr_rows)

    # Build mapping from spatial_idx to position in yr_rows
    # If panel is balanced (all n cells present every year), this is identity
    is_balanced <- (n_yr == n) && all(yr_spatial_idx == seq_len(n))

    if (!is_balanced) {
      # Sparse year: need explicit mapping
      pos_in_yr <- integer(n)  # 0 means absent
      pos_in_yr[yr_spatial_idx] <- seq_len(n_yr)
    }

    for (var_name in neighbor_source_vars) {
      col_max  <- paste0("neighbor_max_", var_name)
      col_min  <- paste0("neighbor_min_", var_name)
      col_mean <- paste0("neighbor_mean_", var_name)

      if (is_balanced) {
        # ---- FAST PATH: balanced panel, all n cells present ----

        # Extract variable as dense vector aligned to spatial_idx 1..n
        x <- yr_rows[[var_name]]  # length n, ordered by spatial_idx

        # -- MEAN via sparse matrix-vector product --
        # Replace NA with 0 for multiplication, track non-NA neighbor counts
        x_nona <- x
        x_nona[is.na(x_nona)] <- 0
        not_na <- as.numeric(!is.na(x))

        # Sum of neighbor values (treating NA as 0)
        neighbor_sum   <- as.numeric(W %*% x_nona)
        # Count of non-NA neighbors
        neighbor_count <- as.numeric(W %*% not_na)

        n_mean <- ifelse(neighbor_count > 0,
                         neighbor_sum / neighbor_count,
                         NA_real_)

        # -- MAX and MIN via data.table grouped aggregation on edge list --
        # Attach neighbor values to edge list
        edge_vals <- edge_dt[, .(from, val = x[to])]
        # Remove NA neighbor values
        edge_vals <- edge_vals[!is.na(val)]

        if (nrow(edge_vals) > 0) {
          agg <- edge_vals[, .(nmax = max(val), nmin = min(val)), by = from]

          n_max <- rep(NA_real_, n)
          n_min <- rep(NA_real_, n)
          n_max[agg$from] <- agg$nmax
          n_min[agg$from] <- agg$nmin
        } else {
          n_max <- rep(NA_real_, n)
          n_min <- rep(NA_real_, n)
        }

        # Also set max/min to NA where degree == 0 (no neighbors at all)
        no_neighbors <- (degree == 0L)
        n_mean[no_neighbors] <- NA_real_
        # n_max and n_min are already NA for nodes not in agg$from

        # -- Write results back --
        # yr_rows is ordered by spatial_idx 1..n, so direct assignment works
        # We need to write into cell_data at the correct row positions
        # Use the row indices from cell_data
        row_idx <- which(cell_data$year == yr)
        # These should be ordered by spatial_idx due to our key
        # Verify ordering matches
        cell_data_yr_order <- cell_data$spatial_idx[row_idx]
        if (!identical(cell_data_yr_order, yr_spatial_idx)) {
          # Reorder to match
          reorder <- match(cell_data_yr_order, yr_spatial_idx)
          set(cell_data, i = row_idx, j = col_max,  value = n_max[reorder])
          set(cell_data, i = row_idx, j = col_min,  value = n_min[reorder])
          set(cell_data, i = row_idx, j = col_mean, value = n_mean[reorder])
        } else {
          set(cell_data, i = row_idx, j = col_max,  value = n_max)
          set(cell_data, i = row_idx, j = col_min,  value = n_min)
          set(cell_data, i = row_idx, j = col_mean, value = n_mean)
        }

      } else {
        # ---- SLOW PATH: unbalanced panel ----
        # Build a full-length vector (length n) with NAs for missing cells
        x_full <- rep(NA_real_, n)
        x_full[yr_spatial_idx] <- yr_rows[[var_name]]

        x_nona <- x_full
        x_nona[is.na(x_nona)] <- 0
        not_na <- as.numeric(!is.na(x_full))

        neighbor_sum   <- as.numeric(W %*% x_nona)
        neighbor_count <- as.numeric(W %*% not_na)

        n_mean <- ifelse(neighbor_count > 0,
                         neighbor_sum / neighbor_count,
                         NA_real_)

        edge_vals <- edge_dt[, .(from, val = x_full[to])]
        edge_vals <- edge_vals[!is.na(val)]

        if (nrow(edge_vals) > 0) {
          agg <- edge_vals[, .(nmax = max(val), nmin = min(val)), by = from]
          n_max <- rep(NA_real_, n)
          n_min <- rep(NA_real_, n)
          n_max[agg$from] <- agg$nmax
          n_min[agg$from] <- agg$nmin
        } else {
          n_max <- rep(NA_real_, n)
          n_min <- rep(NA_real_, n)
        }

        no_neighbors <- (degree == 0L)
        n_mean[no_neighbors] <- NA_real_

        # Extract only the values for cells present this year
        row_idx <- which(cell_data$year == yr)
        cell_data_yr_sidx <- cell_data$spatial_idx[row_idx]

        set(cell_data, i = row_idx, j = col_max,
            value = n_max[cell_data_yr_sidx])
        set(cell_data, i = row_idx, j = col_min,
            value = n_min[cell_data_yr_sidx])
        set(cell_data, i = row_idx, j = col_mean,
            value = n_mean[cell_data_yr_sidx])
      }
    }
    cat(sprintf("  Year %d done.\n", yr))
  }

  # Clean up helper column
  cell_data[, spatial_idx := NULL]

  cat("Neighbor feature engineering complete.\n")
  return(cell_data)
}

# ==========================================================================
# USAGE
# ==========================================================================
#
# # Load pre-existing objects
# # cell_data:               data.frame/data.table with columns id, year, ntl, ec, ...
# # id_order:                vector of cell IDs matching rook_neighbors_unique
# # rook_neighbors_unique:   nb object (list of integer neighbor index vectors)
# # rf_model:                pre-trained randomForest model (DO NOT retrain)
#
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# cell_data <- compute_all_neighbor_features(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   nb_obj               = rook_neighbors_unique,
#   neighbor_source_vars = neighbor_source_vars
# )
#
# # Predict using the existing trained model (no retraining)
# cell_data$prediction <- predict(rf_model, newdata = cell_data)
# ==========================================================================
```

---

## Why This Preserves Numerical Equivalence

| Original operation | Optimized equivalent | Equivalence guarantee |
|---|---|---|
| `max(neighbor_vals[!is.na(...)])` | `edge_dt[!is.na(val), max(val), by=from]` | Identical: same values, same `max()` function |
| `min(neighbor_vals[!is.na(...)])` | `edge_dt[!is.na(val), min(val), by=from]` | Identical |
| `mean(neighbor_vals[!is.na(...)])` | `(W %*% x_nona) / (W %*% not_na)` | Identical: sum of non-NA values Ã· count of non-NA values = `mean()` of non-NA values |
| No neighbors â†’ `NA` | `degree==0` or absent from `agg` â†’ `NA` | Identical |
| Neighbor absent in year â†’ skip | `x_full` has `NA` â†’ filtered out | Identical |

## Performance Summary

| Metric | Original | Optimized |
|---|---|---|
| Topology build | ~6.46M string-key lookups | 1 sparse matrix (~1.37M nonzeros) |
| Per-variable-year mean | R-level loop over 344K lists | Single sparse mat-vec (~0.01s) |
| Per-variable-year max/min | R-level loop over 344K lists | `data.table` grouped agg (~0.1s) |
| Total estimated time | **86+ hours** | **< 1 minute** |
| Peak RAM | ~6.46M lists of integer vectors | ~50 MB sparse matrix + edge table |
| RF model | Unchanged | Unchanged |
END_RESPONSE>>>
