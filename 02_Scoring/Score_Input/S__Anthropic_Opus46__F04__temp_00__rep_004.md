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
run_id: Anthropic_F04_T00_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M string operations is extremely expensive in base R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` with per-element R function calls over 6.46M entries, each allocating small vectors.

**Root causes, ranked:**

| # | Cause | Impact |
|---|-------|--------|
| 1 | Per-row `paste()` + character key lookup in `build_neighbor_lookup` (~6.46M iterations, each creating multiple strings) | **Dominant** |
| 2 | Per-row anonymous function in `compute_neighbor_stats` with small vector allocations | **Major** |
| 3 | `do.call(rbind, result)` on a 6.46M-element list of 3-vectors | **Moderate** |
| 4 | Everything is single-threaded base R | **Compounding** |

## Optimization Strategy

1. **Eliminate all string key construction.** Replace the `paste(id, year)`-based lookup with integer arithmetic. Since years are contiguous (1992â€“2019, i.e., 28 years), we can map every `(id, year)` pair to a row index via a pre-built integer matrix indexed as `row_matrix[id_index, year_index]`. Lookup becomes a single integer matrix access â€” no strings, no hashing.

2. **Vectorize neighbor stat computation using `data.table` grouping or, better, a single pre-built sparse adjacency structure and matrix operations.** We build a sparse neighbor matrix (CSR-style, via two integer vectors: `neighbor_row_idx` and `target_row_idx`) and use vectorized grouped operations.

3. **Replace `lapply` + `do.call(rbind, ...)` with pre-allocated matrices.**

4. **Optionally parallelize** the five variables, but the vectorization alone should reduce runtime from 86+ hours to minutes.

**Expected speedup:** The dominant cost moves from ~6.46M Ã— k interpreted R calls with string ops to a single vectorized sparse-matrixâ€“style operation. Estimated wall time: **2â€“10 minutes** on a 16 GB laptop (down from 86+ hours).

## Optimized Working R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE CONSTRUCTION
# Drop-in replacement. Preserves the trained RF model and original estimand.
# =============================================================================

library(data.table)

build_neighbor_lookup_fast <- function(cell_data, id_order, rook_neighbors_unique) {
  # ---- Step 1: Create integer mappings (no strings) ----
  # Map each unique cell id to a contiguous integer index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map each year to a contiguous integer index
  years <- sort(unique(cell_data$year))
  year_to_idx <- setNames(seq_along(years), as.character(years))
  n_years <- length(years)
  n_ids   <- length(id_order)

  # ---- Step 2: Build (id_index, year_index) -> row number matrix ----
  # This replaces the paste-based idx_lookup entirely.
  # row_matrix[id_idx, year_idx] = row number in cell_data (or NA)
  row_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)

  cd_id_idx   <- id_to_idx[as.character(cell_data$id)]
  cd_year_idx <- year_to_idx[as.character(cell_data$year)]
  row_matrix[cbind(cd_id_idx, cd_year_idx)] <- seq_len(nrow(cell_data))

  # ---- Step 3: Expand neighbor pairs into (target_row, neighbor_row) ----
  # For each cell i (in id_order), rook_neighbors_unique[[i]] gives

  # the indices (into id_order) of its neighbors.
  # We need to expand this across all 28 years.

  # Build flat edge list at the id-index level
  n_neighbors <- lengths(rook_neighbors_unique)  # integer vector, length n_ids
  from_id_idx <- rep(seq_len(n_ids), times = n_neighbors)
  to_id_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)
  # from_id_idx[k] -> to_id_idx[k] is one directed neighbor relationship

  n_edges <- length(from_id_idx)
  cat(sprintf("Neighbor edges (id-level): %d\n", n_edges))
  cat(sprintf("Expanding across %d years -> %d edge-year pairs\n",
              n_years, n_edges * n_years))

  # Expand across years: each edge is replicated for every year
  # Use integer rep to avoid huge intermediate objects
  from_id_expanded <- rep(from_id_idx, times = n_years)
  to_id_expanded   <- rep(to_id_idx,   times = n_years)
  year_idx_expanded <- rep(seq_len(n_years), each = n_edges)

  # Map to row numbers in cell_data
  target_rows   <- row_matrix[cbind(from_id_expanded, year_idx_expanded)]
  neighbor_rows <- row_matrix[cbind(to_id_expanded,   year_idx_expanded)]

  # Remove pairs where either target or neighbor is missing
  valid <- !is.na(target_rows) & !is.na(neighbor_rows)
  target_rows   <- target_rows[valid]
  neighbor_rows <- neighbor_rows[valid]

  cat(sprintf("Valid (target_row, neighbor_row) pairs: %d\n", length(target_rows)))

  list(target_rows = target_rows, neighbor_rows = neighbor_rows,
       n_data_rows = nrow(cell_data))
}


compute_neighbor_stats_fast <- function(cell_data, edge_list, var_name) {
  # edge_list: list with target_rows, neighbor_rows, n_data_rows
  vals <- cell_data[[var_name]]
  neighbor_vals <- vals[edge_list$neighbor_rows]

  # Build a data.table for grouped aggregation (vectorized C code)
  dt <- data.table(
    target = edge_list$target_rows,
    nval   = neighbor_vals
  )

  # Remove edges where the neighbor value is NA
  dt <- dt[!is.na(nval)]

  # Grouped aggregation â€” single pass, highly optimized in data.table
  agg <- dt[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = target]

  # Allocate output columns (NA for rows with no valid neighbors)
  nb_max  <- rep(NA_real_, edge_list$n_data_rows)
  nb_min  <- rep(NA_real_, edge_list$n_data_rows)
  nb_mean <- rep(NA_real_, edge_list$n_data_rows)

  nb_max[agg$target]  <- agg$nb_max
  nb_min[agg$target]  <- agg$nb_min
  nb_mean[agg$target] <- agg$nb_mean

  data.frame(nb_max = nb_max, nb_min = nb_min, nb_mean = nb_mean)
}


compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_list) {
  stats <- compute_neighbor_stats_fast(cell_data, edge_list, var_name)
  cell_data[[paste0(var_name, "_nb_max")]]  <- stats$nb_max
  cell_data[[paste0(var_name, "_nb_min")]]  <- stats$nb_min
  cell_data[[paste0(var_name, "_nb_mean")]] <- stats$nb_mean
  cell_data
}


# =============================================================================
# MAIN EXECUTION (drop-in replacement for the original outer loop)
# =============================================================================

cat("Building optimized neighbor edge list...\n")
system.time({
  edge_list <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features for", length(neighbor_source_vars), "variables...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, edge_list)
  }
})

cat("Done. Neighbor features added to cell_data.\n")

# The trained Random Forest model is untouched.
# Column names match the original pattern: {var}_nb_max, {var}_nb_min, {var}_nb_mean
# Numerical results are identical (same max/min/mean over the same neighbor sets).
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Same neighbor sets** | We use the same `rook_neighbors_unique` nb object and the same `id_order`. The integer matrix lookup `row_matrix[id_idx, year_idx]` returns exactly the same row as the original `paste(id, year)` string lookup â€” it's just a different indexing scheme for the same mapping. |
| **Same statistics** | `max`, `min`, `mean` over the same non-NA neighbor values, grouped by the same target row. `data.table` aggregation is numerically identical to the base R computation. |
| **Same output columns** | Column names follow the same `{var}_nb_max/min/mean` pattern, so the trained Random Forest's `predict()` call will find the expected feature names. |
| **RF model untouched** | No retraining; we only reconstruct input features for prediction. |

## Complexity Comparison

| Operation | Original | Optimized |
|-----------|----------|-----------|
| Lookup construction | O(N) `paste` + named-vector hash lookups (N = 6.46M, each with k neighbors) | O(N) integer matrix writes + O(EÃ—T) integer matrix reads (all vectorized in C) |
| Stat computation | O(N) `lapply` with R-level function calls | O(EÃ—T) `data.table` grouped aggregation (radix sort + single C pass) |
| Result assembly | `do.call(rbind, 6.46M-element list)` | Pre-allocated vector indexed assignment |
| **Estimated time** | **86+ hours** | **2â€“10 minutes** |

The key insight: replacing millions of interpreted R-level string operations and per-row anonymous function calls with vectorized integer indexing and `data.table` grouped aggregation eliminates the bottleneck entirely.
END_RESPONSE>>>
