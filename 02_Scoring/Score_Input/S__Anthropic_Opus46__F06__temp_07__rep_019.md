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
run_id: Anthropic_F06_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F06
prompt_family_slug: RASTER_KERNEL_ANALOGY
family_label: Raster/kernel analogy
family_group: Raster/kernel bridge
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F06_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The current implementation has two critical bottlenecks:

### Bottleneck A: `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows

For every single cell-year row, the function:
1. Looks up the spatial cell's rook neighbors (fine â€” hash lookup).
2. Constructs `paste(neighbor_id, year)` key strings for each neighbor (expensive string allocation Ã—6.46M rows Ã— ~4 neighbors each).
3. Indexes into `idx_lookup` (a named character vector â€” each lookup is O(n) or at best O(log n) in the names hash).

This produces a **list of 6.46 million integer vectors**. The memory footprint of that list alone is enormous (each list element has R overhead of ~100+ bytes, so â‰¥600 MB just in list overhead, plus the index vectors).

### Bottleneck B: `compute_neighbor_stats` â€” `lapply` over 6.46M elements, repeated 5 times

For each of the 5 source variables, another `lapply` iterates over all 6.46M rows, subsets a numeric vector by index, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself very slow (millions of small vector allocations).

### Why raster focal/kernel operations are the right analogy but wrong implementation

The comment in the prompt is apt: this is conceptually a **focal (moving-window) operation** â€” compute summary statistics over each cell's neighborhood. On a regular raster grid, `terra::focal()` would do this in C in seconds. However, `spdep::nb` objects encode an **irregular** neighborhood structure (coastal cells, boundary cells have fewer neighbors; the grid may have holes). A rectangular kernel would compute over the wrong set of neighbors and would **not preserve the required numerical results**. So we must keep the explicit neighbor structure but vectorize the computation.

### Summary of the problem

| Component | Current complexity | Time estimate |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M string operations + hash lookups | ~30â€“40 min |
| `compute_neighbor_stats` Ã— 5 vars | ~32.3M R-level lapply iterations + rbind | ~80+ hours |
| **Total** | | **86+ hours** |

---

## 2. Optimization Strategy

### Strategy: Sparse-matrix vectorization (no loops, no lapply over rows)

**Key insight:** Since the panel is balanced (every cell appears in every year), the neighbor structure is **identical across years**. We can:

1. **Sort data by `(year, id)`** so that within each year-block, cells appear in the same spatial order.
2. **Build a sparse adjacency matrix** (344,208 Ã— 344,208) from the `nb` object â€” done once.
3. **Reshape each variable into a matrix** of shape (344,208 cells Ã— 28 years).
4. **Use sparse matrixâ€“dense matrix multiplication** to compute neighbor sums and counts, then derive max/min/mean.

For **mean**, sparse matrix multiplication gives us `sum` and we know `count` from the adjacency structure, so `mean = sum / count`.

For **max and min**, sparse matrix multiplication doesn't directly help, but we can use a grouped operation via `data.table` or, better yet, iterate over the (at most 4â€“6) neighbor positions in the `nb` list and take element-wise max/min across neighbor "layers." With only ~4 neighbors per cell on average, this is 4 passes over a 344K Ã— 28 matrix â€” trivial.

**Expected speedup:** From 86+ hours to **under 5 minutes**.

---

## 3. Working R Code

```r
library(data.table)
library(Matrix)

#' Optimized neighbor feature engineering for cell-year panel data.
#'
#' Preserves the exact same numerical results as the original
#' compute_neighbor_stats (max, min, mean of rook neighbors),
#' but replaces the 6.46M-row lapply with vectorized matrix operations.
#'
#' @param cell_data       data.frame/data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of unique cell IDs in the order matching rook_neighbors_unique
#' @param rook_nb         spdep::nb object (list of integer index vectors into id_order)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with new columns appended (e.g., ntl_max, ntl_min, ntl_mean, ...)

add_all_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_nb,
                                       neighbor_source_vars) {

  # ---- Convert to data.table for speed ----
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    was_df <- TRUE
  } else {
    was_df <- FALSE
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))
  n_years <- length(years)

  cat("Cells:", n_cells, " Years:", n_years, " Rows:", nrow(cell_data), "\n")

  # ---- Step 1: Establish consistent ordering ----
  # Map each cell id to its spatial index (1..n_cells)
  id_to_sidx <- setNames(seq_along(id_order), as.character(id_order))

  # Map each year to its temporal index (1..n_years)
  year_to_tidx <- setNames(seq_along(years), as.character(years))

  # Compute spatial and temporal indices for every row
  cell_data[, .spatial_idx := id_to_sidx[as.character(id)]]
  cell_data[, .temporal_idx := year_to_tidx[as.character(year)]]

  # ---- Step 2: Build the max number of neighbors & padded neighbor matrix ----
  # rook_nb is a list: rook_nb[[i]] gives integer vector of neighbor indices
  # (indices into id_order). spdep uses 0L for cells with no neighbors.
  n_neighbors <- vapply(rook_nb, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))
  max_k <- max(n_neighbors)

  cat("Max rook neighbors:", max_k, "\n")

  # Padded neighbor index matrix: n_cells x max_k

  # Pad with NA for cells with fewer than max_k neighbors
  nb_matrix <- matrix(NA_integer_, nrow = n_cells, ncol = max_k)
  for (i in seq_len(n_cells)) {
    nbrs <- rook_nb[[i]]
    if (length(nbrs) == 1L && nbrs[1] == 0L) next
    nb_matrix[i, seq_along(nbrs)] <- nbrs
  }

  # Also build a count vector for computing means
  nb_count <- n_neighbors  # integer vector length n_cells

  # ---- Step 3: For each variable, compute max/min/sum via neighbor layers ----
  for (var_name in neighbor_source_vars) {
    cat("Processing variable:", var_name, "...\n")
    t0 <- proc.time()

    # 3a. Reshape variable into matrix: n_cells x n_years
    #     val_mat[s, t] = value for spatial cell s in year t
    val_mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
    val_mat[cbind(cell_data$.spatial_idx, cell_data$.temporal_idx)] <- cell_data[[var_name]]

    # 3b. Initialize accumulator matrices
    max_mat  <- matrix(-Inf, nrow = n_cells, ncol = n_years)
    min_mat  <- matrix(Inf,  nrow = n_cells, ncol = n_years)
    sum_mat  <- matrix(0,    nrow = n_cells, ncol = n_years)
    count_mat <- matrix(0L,  nrow = n_cells, ncol = n_years)

    # 3c. Iterate over neighbor "slots" (max_k is typically 4 for rook)
    for (k in seq_len(max_k)) {
      nbr_idx <- nb_matrix[, k]          # length n_cells; NA if no k-th neighbor
      has_nbr <- !is.na(nbr_idx)         # logical mask

      if (!any(has_nbr)) next

      # Extract the neighbor's value matrix rows
      # nbr_vals[i, t] = val_mat[nbr_idx[i], t] for cells that have a k-th neighbor
      # We do this for ALL years at once (matrix row indexing)
      active_cells <- which(has_nbr)
      active_nbrs  <- nbr_idx[active_cells]

      nbr_vals <- val_mat[active_nbrs, , drop = FALSE]  # length(active_cells) x n_years
      not_na   <- !is.na(nbr_vals)                       # same shape, logical

      # Update max: element-wise max with current accumulator
      current_max <- max_mat[active_cells, , drop = FALSE]
      update_mask <- not_na & (nbr_vals > current_max)
      # Use direct replacement
      idx_update <- which(update_mask)
      if (length(idx_update) > 0) {
        # Convert to linear indices in the sub-matrix, then map back
        max_sub <- current_max
        max_sub[idx_update] <- nbr_vals[idx_update]
        max_mat[active_cells, ] <- max_sub
      }
      # Actually, simpler and faster with pmax:
      max_mat[active_cells, ] <- pmax(max_mat[active_cells, , drop = FALSE],
                                       ifelse(not_na, nbr_vals, -Inf))

      # Update min
      min_mat[active_cells, ] <- pmin(min_mat[active_cells, , drop = FALSE],
                                       ifelse(not_na, nbr_vals, Inf))

      # Update sum and count
      nbr_vals_zero <- nbr_vals
      nbr_vals_zero[!not_na] <- 0
      sum_mat[active_cells, ] <- sum_mat[active_cells, , drop = FALSE] + nbr_vals_zero
      count_mat[active_cells, ] <- count_mat[active_cells, , drop = FALSE] + as.integer(not_na)
    }

    # 3d. Compute mean
    mean_mat <- sum_mat / count_mat  # NaN where count == 0, will become NA

    # 3e. Replace -Inf/Inf/NaN with NA for cells with no valid neighbors
    no_valid <- (count_mat == 0L)
    max_mat[no_valid]  <- NA_real_
    min_mat[no_valid]  <- NA_real_
    mean_mat[no_valid] <- NA_real_

    # 3f. Map results back to cell_data rows using (.spatial_idx, .temporal_idx)
    linear_idx <- cell_data$.spatial_idx + (cell_data$.temporal_idx - 1L) * n_cells

    col_max  <- paste0(var_name, "_max")
    col_min  <- paste0(var_name, "_min")
    col_mean <- paste0(var_name, "_mean")

    set(cell_data, j = col_max,  value = max_mat[linear_idx])
    set(cell_data, j = col_min,  value = min_mat[linear_idx])
    set(cell_data, j = col_mean, value = mean_mat[linear_idx])

    elapsed <- (proc.time() - t0)[3]
    cat("  Done in", round(elapsed, 1), "seconds.\n")
  }

  # ---- Cleanup temporary columns ----
  cell_data[, .spatial_idx := NULL]
  cell_data[, .temporal_idx := NULL]

  if (was_df) cell_data <- as.data.frame(cell_data)

  return(cell_data)
}


# ==============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_nb              = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# Now cell_data has 15 new columns:
#   ntl_max, ntl_min, ntl_mean,
#   ec_max,  ec_min,  ec_mean,
#   pop_density_max, pop_density_min, pop_density_mean,
#   def_max, def_min, def_mean,
#   usd_est_n2_max,  usd_est_n2_min,  usd_est_n2_mean

# Proceed with prediction using the pre-trained Random Forest:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## 4. Why This Works and What It Preserves

| Concern | Resolution |
|---|---|
| **Numerical equivalence** | For each cell-year, the exact same set of rook-neighbor values are passed through `max`, `min`, and `mean`. The `pmax`/`pmin` element-wise approach over neighbor "slots" is algebraically identical to computing `max(vals[idx])` in the original code. |
| **NA handling** | Matches the original: NAs in neighbor values are excluded (`-Inf`/`Inf` sentinels are masked out where `count == 0`). Cells with zero valid neighbors get `NA` for all three statistics. |
| **Trained RF model** | Untouched â€” only the feature-engineering step is optimized. The 15 new columns have the same names and values, so `predict(model, newdata = cell_data)` produces identical results. |
| **Memory** | Peak memory: ~5 matrices of size 344,208 Ã— 28 Ã— 8 bytes â‰ˆ **385 MB** for the working matrices, well within 16 GB. The `nb_matrix` (344,208 Ã— 4) adds ~11 MB. |
| **Speed** | The inner loop runs `max_k` iterations (typically 4 for rook contiguity). Each iteration does vectorized matrix operations over 344K Ã— 28 elements. For 5 variables: **5 Ã— 4 Ã— ~0.3s â‰ˆ 6 seconds** total, plus overhead for reshaping (~30s). **Expected total: under 2 minutes** vs. the original 86+ hours â€” a **~2,500Ã— speedup**. |
| **No raster focal** | As noted, `terra::focal()` with a fixed rectangular kernel would apply to wrong neighbor sets at boundaries/holes. The padded neighbor-matrix approach respects the exact irregular `spdep::nb` topology. |
END_RESPONSE>>>
