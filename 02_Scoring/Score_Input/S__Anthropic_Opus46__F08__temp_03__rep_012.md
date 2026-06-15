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
run_id: Anthropic_F08_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **row-level** lookup of length ~6.46 million, where each entry maps a cell-year row to the row indices of its neighbors *in that same year*. This is enormously wasteful because:

1. **The neighbor topology is static.** Cell A's neighbors are the same in 1992 as in 2019. Yet the current code rebuilds the mapping for every cell-year combination â€” effectively duplicating the same spatial relationship 28 times per cell.

2. **String-key hashing is expensive at scale.** The function creates ~6.46M paste-based keys (`id_year`) and performs named-vector lookups inside a `lapply` over 6.46M rows. Named vector lookup in R is O(n) in the worst case for each access, and `paste` + string matching over millions of keys is slow.

3. **`compute_neighbor_stats` iterates row-by-row over 6.46M entries** using `lapply`, which is inherently slow in R even when the inner operation is trivial.

4. **The combination** of 6.46M-element `lapply` in `build_neighbor_lookup` and then again in `compute_neighbor_stats` (called 5 times, once per variable) produces the estimated 86+ hour runtime.

**Key insight:** Because the neighbor graph is year-invariant, we can split the problem into:
- A **static spatial lookup** (344K cells â†’ their neighbor cell indices), built once.
- A **year-level matrix operation** where, for each year, we pull the variable values for all cells and compute neighbor max/min/mean using fast vectorized or matrix operations.

## Optimization Strategy

1. **Build a cell-level neighbor lookup once** â€” a list of length 344,208 where each element contains the integer positions of that cell's neighbors within the cell-ID ordering. This is just a cleaned version of `rook_neighbors_unique` (the `nb` object) and costs essentially nothing.

2. **Reshape data so that each year's variable values are accessible as a simple numeric vector indexed by cell position.** We ensure `cell_data` is sorted by `(id, year)` or use a fast index so that for a given year, we can extract a vector of length 344,208 for any variable.

3. **For each variable and each year (28 iterations), vectorize the neighbor aggregation** using the static cell-level neighbor list. This reduces the inner loop from 6.46M iterations to 344K iterations Ã— 28 years = 9.63M, but each iteration is a trivial index-into-vector operation. We can further accelerate with `vapply` or, even better, by constructing a sparse adjacency matrix and using matrix multiplication for the mean, and row-wise operations for max/min.

4. **Sparse matrix approach for mean:** Construct a row-normalized sparse adjacency matrix `W` (344,208 Ã— 344,208). Then `neighbor_mean = W %*% values_vector` is a single sparse matrix-vector multiply per year per variable â€” extremely fast. For max and min, we use a compiled loop or a grouped operation.

5. **Result:** Instead of ~6.46M `lapply` iterations with string lookups, we get 28 sparse matrix-vector multiplies per variable (for mean) plus fast compiled max/min operations. Total wall-clock time drops from 86+ hours to minutes.

## Working R Code

```r
library(Matrix)
library(data.table)

# ==============================================================================
# STEP 1: Build static spatial structures (done ONCE)
# ==============================================================================

#' Build a sparse row-normalized adjacency matrix from an nb object,
#' plus a raw adjacency list for max/min.
#'
#' @param nb_obj   spdep::nb object (list of integer neighbor indices), length N_cells
#' @param n_cells  number of spatial cells
#' @return list with:
#'   - adj_list: the nb object cleaned (integer neighbor indices per cell)
#'   - W_mean:   sparse Matrix (dgCMatrix), row-normalized for computing means
#'   - W_adj:    sparse binary adjacency Matrix (dgCMatrix) for max/min helpers

build_static_spatial_structures <- function(nb_obj, n_cells) {

  # --- Build sparse adjacency matrix (binary) ---
  # Each entry (i, j) = 1 if cell j is a neighbor of cell i
  from <- rep(seq_len(n_cells), times = lengths(nb_obj))
  to   <- unlist(nb_obj)

  # Remove any 0-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(to) & to > 0L
  from  <- from[valid]
  to    <- to[valid]

  W_adj <- sparseMatrix(
    i    = from,
    j    = to,
    x    = 1,
    dims = c(n_cells, n_cells)
  )

  # --- Row-normalized version for mean computation ---
  row_sums <- rowSums(W_adj)
  row_sums[row_sums == 0] <- NA_real_   # islands get NA
  # Diagonal matrix of 1/row_sums
  D_inv <- Diagonal(x = ifelse(is.na(row_sums), 0, 1 / row_sums))
  W_mean <- D_inv %*% W_adj

  # --- Clean adjacency list (for max/min) ---
  adj_list <- lapply(nb_obj, function(x) {
    x <- as.integer(x)
    x[!is.na(x) & x > 0L]
  })

  list(
    adj_list = adj_list,
    W_mean   = W_mean,
    W_adj    = W_adj,
    n_neighbors = as.integer(row_sums)
  )
}

# ==============================================================================
# STEP 2: Compute neighbor max and min using the adjacency list
# ==============================================================================

#' Fast neighbor max and min for a single numeric vector (one year, one variable).
#' Uses vapply over the static adjacency list.
#'
#' @param values   numeric vector of length n_cells (one value per cell for one year)
#' @param adj_list list of integer vectors (neighbor indices per cell)
#' @return matrix of dimension (n_cells, 2): columns are max, min

neighbor_max_min <- function(values, adj_list) {
  result <- vapply(adj_list, function(idx) {
    if (length(idx) == 0L) return(c(NA_real_, NA_real_))
    nv <- values[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) return(c(NA_real_, NA_real_))
    c(max(nv), min(nv))
  }, numeric(2))
  t(result)  # transpose to n_cells x 2
}

# ==============================================================================
# STEP 3: Compute neighbor mean using sparse matrix multiplication
# ==============================================================================

#' Fast neighbor mean for a single numeric vector (one year, one variable).
#'
#' @param values numeric vector of length n_cells
#' @param W_mean row-normalized sparse adjacency matrix
#' @param n_neighbors integer vector of neighbor counts per cell
#' @return numeric vector of length n_cells

neighbor_mean_sparse <- function(values, W_mean, n_neighbors) {
  # Handle NAs in values: sparse matmul treats them as 0, so we need correction.
  # Strategy: compute sum of non-NA neighbors and count of non-NA neighbors.
  
  is_valid   <- !is.na(values)
  values_0   <- values
  values_0[!is_valid] <- 0  # replace NA with 0 for matmul

  neighbor_sum   <- as.numeric(W_mean %*% values_0) * n_neighbors
  neighbor_count <- as.numeric(W_mean %*% as.numeric(is_valid)) * n_neighbors

  result <- ifelse(neighbor_count > 0, neighbor_sum / neighbor_count, NA_real_)
  # Cells with 0 neighbors (islands)
  result[n_neighbors == 0 | is.na(n_neighbors)] <- NA_real_
  result
}

# ==============================================================================
# STEP 4: Main driver â€” compute all neighbor features for all variables
# ==============================================================================

#' Compute neighbor max, min, mean for all source variables across all years.
#' Adds columns: {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
#'
#' @param cell_data     data.frame/data.table with columns: id, year, and all source vars
#' @param id_order      integer vector of cell IDs in the order matching the nb object
#' @param nb_obj        spdep::nb object (rook_neighbors_unique)
#' @param source_vars   character vector of variable names to compute neighbor stats for
#' @return cell_data with new neighbor feature columns appended

compute_all_neighbor_features <- function(cell_data, id_order, nb_obj, source_vars) {

  # Convert to data.table for speed (non-destructive if already data.table)
  dt <- as.data.table(cell_data)
  n_cells <- length(id_order)
  years   <- sort(unique(dt$year))

  message("Building static spatial structures for ", n_cells, " cells...")
  spatial <- build_static_spatial_structures(nb_obj, n_cells)

  # Map cell IDs to their position in id_order (1-based index into adj_list / matrix rows)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Ensure data is keyed for fast subsetting
  setkey(dt, year)

  # Pre-allocate output columns
  for (var_name in source_vars) {
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")
    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]
  }

  # Process year by year
  for (yr in years) {
    message("  Processing year ", yr, " ...")

    # Get the subset of rows for this year
    yr_idx <- which(dt$year == yr)
    yr_data <- dt[yr_idx]

    # Map each row's cell ID to its position in the spatial ordering
    cell_positions <- id_to_pos[as.character(yr_data$id)]

    # Build a full-length vector (n_cells) for each variable, indexed by cell position
    for (var_name in source_vars) {
      # Initialize with NA
      full_vec <- rep(NA_real_, n_cells)
      full_vec[cell_positions] <- yr_data[[var_name]]

      # --- Neighbor mean via sparse matrix multiply ---
      n_mean <- neighbor_mean_sparse(full_vec, spatial$W_mean, spatial$n_neighbors)

      # --- Neighbor max and min via adjacency list ---
      mm <- neighbor_max_min(full_vec, spatial$adj_list)

      # Write results back to the data.table rows for this year
      # Map from full cell vector back to the rows present in this year
      col_max  <- paste0(var_name, "_neighbor_max")
      col_min  <- paste0(var_name, "_neighbor_min")
      col_mean <- paste0(var_name, "_neighbor_mean")

      set(dt, i = yr_idx, j = col_max,  value = mm[cell_positions, 1])
      set(dt, i = yr_idx, j = col_min,  value = mm[cell_positions, 2])
      set(dt, i = yr_idx, j = col_mean, value = n_mean[cell_positions])
    }
  }

  message("Neighbor feature computation complete.")

  # Return as same class as input
  if (is.data.table(cell_data)) {
    return(dt)
  } else {
    return(as.data.frame(dt))
  }
}

# ==============================================================================
# STEP 5: Usage â€” drop-in replacement for the original outer loop
# ==============================================================================

# Original code (86+ hours):
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# Optimized replacement (minutes):
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data    = cell_data,
  id_order     = id_order,
  nb_obj       = rook_neighbors_unique,
  source_vars  = neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Improvements

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M-element list with string key hashing | 344K-element static adjacency list + sparse matrix, built once |
| **Mean computation** | `lapply` over 6.46M rows, per variable | Sparse matrixâ€“vector multiply: 28 years Ã— 5 vars = 140 multiplies on 344K-dim vectors |
| **Max/Min computation** | Embedded in same 6.46M `lapply` | `vapply` over 344K cells Ã— 28 years Ã— 5 vars (compiled inner ops) |
| **Memory** | ~6.46M-element list of integer vectors + string keys | ~1.37M-nonzero sparse matrix (~33 MB) + 344K adjacency list |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes on a 16 GB laptop |
| **Numerical result** | â€” | Identical max, min, mean values per cell-year |
| **RF model** | â€” | Untouched; no retraining |
END_RESPONSE>>>
