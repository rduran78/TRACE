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
run_id: Anthropic_F08_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F08
prompt_family_slug: TOPOLOGY_INVARIANCE
family_label: Topology invariance
family_group: Topology cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F08_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The core performance problem is that `build_neighbor_lookup` constructs a **per-row** (per cell-year) lookup, producing ~6.46 million list entries, when the neighbor *topology* is identical across all 28 years. The function pastes cell IDs and years into string keys, performs hash lookups for every neighbor of every row, and repeats this identical structural work 28 times per cell. This results in:

1. **~6.46 million list elements** instead of ~344,208 (one per cell).
2. **String concatenation and hash-table lookups** (`paste`, named-vector indexing) on ~6.46M Ã— avg_neighbors scale â€” extremely slow in R.
3. **Redundant recomputation**: the neighbor graph is static, but the lookup is rebuilt as if it were year-varying.
4. **`compute_neighbor_stats`** then iterates over 6.46M list entries per variable â€” 5 variables Ã— 6.46M = ~32.3M R-level `lapply` iterations, each doing subsetting and summary stats.

The 86+ hour estimate is entirely explained by this O(cells Ã— years Ã— neighbors) string-key approach applied at the R interpreter level.

## Optimization Strategy

**Separate topology (static) from data (year-varying):**

1. **Build the neighbor lookup once, at the cell level only** (~344K entries). Map each cell to its position in `id_order`, and store neighbor *positions* (integer indices into `id_order`). This is a one-time O(cells Ã— avg_neighbors) operation with no string manipulation.

2. **Reshape each variable into a matrix**: rows = cells (in `id_order` order), columns = years. This gives O(1) column-vector access to all cells' values for a given year.

3. **Vectorized neighbor stats per year**: For each year-column, use the static cell-level neighbor list to compute max/min/mean of neighbor values. Critically, we can do this with a single `vapply` over ~344K cells (not 6.46M rows), repeated for 28 years â€” a ~18.7Ã— reduction in iterations. Each iteration is a simple integer-index subset of a numeric vector (no string ops).

4. **Flatten back** to the original cell-year row order and attach columns.

This reduces the dominant cost from ~6.46M Ã— 5 slow string-based lookups to ~344K Ã— 28 Ã— 5 fast integer-vector subsets, a roughly **500â€“1000Ã— speedup**, bringing runtime to minutes.

## Working R Code

```r
# =============================================================================
# OPTIMIZED NEIGHBOR FEATURE COMPUTATION
# Separates static topology from year-varying data
# =============================================================================

#' Build a cell-level neighbor index lookup (static, computed once).
#' 
#' @param id_order Integer vector of cell IDs in the order matching rook_neighbors_unique.
#' @param neighbors An spdep::nb object (list of integer index vectors into id_order).
#' @return A list of length length(id_order). Each element is an integer vector
#'         of positions (indices into id_order) of that cell's neighbors.
build_cell_neighbor_lookup <- function(id_order, neighbors) {
  # neighbors is already an nb object: neighbors[[i]] gives integer indices

# into id_order for the neighbors of id_order[i].
  # We just need to clean it (remove 0L entries that spdep uses for "no neighbors").
  n <- length(id_order)
  lookup <- vector("list", n)
  for (i in seq_len(n)) {
    nb <- neighbors[[i]]
    # spdep::nb uses 0L to indicate no neighbors
    nb <- nb[nb != 0L]
    lookup[[i]] <- nb
  }
  lookup
}

#' Build a mapping from (cell_id, year) in cell_data to (cell_position, year_index).
#' Returns the year vector (sorted unique years) and a matrix of variable values
#' with rows = cells (in id_order order) and columns = years.
#'
#' @param cell_data Data frame with columns id, year, and variable columns.
#' @param id_order Integer vector of cell IDs.
#' @param var_name Character: name of the variable to extract.
#' @return A list with:
#'   - years: sorted unique year vector
#'   - mat: numeric matrix [length(id_order) x length(years)]
build_variable_matrix <- function(cell_data, id_order, var_name) {
  years <- sort(unique(cell_data$year))
  n_cells <- length(id_order)
  n_years <- length(years)
  
  # Create mapping from cell id to row-position in matrix
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  # If IDs are not contiguous or max is very large, use a hash instead:
  # But for 344K cells this is fine (max ~344K integers = ~1.4 MB)
  
  # Create mapping from year to column-position
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  # Allocate matrix

  mat <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Fill matrix
  row_pos <- id_to_pos[cell_data$id]
  col_pos <- year_to_col[as.character(cell_data$year)]
  mat[cbind(row_pos, col_pos)] <- cell_data[[var_name]]
  
  list(years = years, mat = mat)
}

#' Compute neighbor max, min, mean for one variable across all cell-years.
#'
#' @param cell_data Data frame (original, with id and year columns).
#' @param id_order Integer vector of cell IDs.
#' @param cell_neighbor_lookup List from build_cell_neighbor_lookup.
#' @param var_name Character: source variable name.
#' @return A data frame with three columns: {var}_neighbor_max, {var}_neighbor_min,
#'         {var}_neighbor_mean, in the same row order as cell_data.
compute_neighbor_stats_optimized <- function(cell_data, id_order, 
                                              cell_neighbor_lookup, var_name) {
  # Step 1: Reshape variable into cell Ã— year matrix
  var_info <- build_variable_matrix(cell_data, id_order, var_name)
  mat      <- var_info$mat      # [n_cells x n_years]
  years    <- var_info$years
  n_cells  <- length(id_order)
  n_years  <- length(years)
  
  # Step 2: Allocate output matrices (same shape)
  out_max  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  out_min  <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  out_mean <- matrix(NA_real_, nrow = n_cells, ncol = n_years)
  
  # Step 3: For each year, compute neighbor stats using the static topology
  for (j in seq_len(n_years)) {
    vals_j <- mat[, j]  # all cell values for this year
    
    # vapply over cells â€” ~344K iterations per year
    stats_j <- vapply(cell_neighbor_lookup, function(nb_idx) {
      if (length(nb_idx) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      nv <- vals_j[nb_idx]
      nv <- nv[!is.na(nv)]
      if (length(nv) == 0L) return(c(NA_real_, NA_real_, NA_real_))
      c(max(nv), min(nv), mean(nv))
    }, numeric(3))
    # stats_j is [3 x n_cells]
    
    out_max[, j]  <- stats_j[1L, ]
    out_min[, j]  <- stats_j[2L, ]
    out_mean[, j] <- stats_j[3L, ]
  }
  
  # Step 4: Map results back to cell_data row order
  # Reconstruct the same positional mappings used in build_variable_matrix
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  year_to_col <- setNames(seq_along(years), as.character(years))
  
  row_pos <- id_to_pos[cell_data$id]
  col_pos <- year_to_col[as.character(cell_data$year)]
  linear_idx <- (col_pos - 1L) * n_cells + row_pos
  
  data.frame(
    nb_max  = out_max[linear_idx],
    nb_min  = out_min[linear_idx],
    nb_mean = out_mean[linear_idx]
  )
}

# =============================================================================
# MAIN EXECUTION â€” replaces the original outer loop
# =============================================================================

# Build the static cell-level neighbor lookup ONCE
cell_neighbor_lookup <- build_cell_neighbor_lookup(id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  
  stats_df <- compute_neighbor_stats_optimized(
    cell_data, id_order, cell_neighbor_lookup, var_name
  )
  
  # Attach columns with original naming convention
  cell_data[[paste0(var_name, "_neighbor_max")]]  <- stats_df$nb_max
  cell_data[[paste0(var_name, "_neighbor_min")]]  <- stats_df$nb_min
  cell_data[[paste0(var_name, "_neighbor_mean")]] <- stats_df$nb_mean
}

# cell_data now has all 15 neighbor feature columns (5 vars Ã— 3 stats)
# and is ready for predict(trained_rf_model, newdata = cell_data)
```

## Summary of Guarantees

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical `max`, `min`, `mean` computed on the same neighbor value sets â€” no approximation. |
| **Trained RF model preserved** | No retraining; code only prepares predictor columns for `predict()`. |
| **Column names preserved** | `{var}_neighbor_max`, `{var}_neighbor_min`, `{var}_neighbor_mean` â€” same as original `compute_and_add_neighbor_features` would produce. (Adjust the `paste0` suffix if your original convention differs.) |
| **Memory** | Peak: three `344K Ã— 28` float matrices (~230 MB) + the cell-level lookup (~50 MB) + `cell_data` itself. Well within 16 GB. |
| **Expected runtime** | ~344K cells Ã— 28 years Ã— 5 vars = ~48.2M `vapply` iterations of trivial integer-subset work. Estimated **5â€“15 minutes** on a standard laptop (vs. 86+ hours). |
END_RESPONSE>>>
