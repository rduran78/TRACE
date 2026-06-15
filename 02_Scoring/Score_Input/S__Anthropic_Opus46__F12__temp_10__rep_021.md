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
run_id: Anthropic_F12_T10_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Audit and Optimization of Spatial Neighbor Pipeline

## Diagnosis

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions:

### `compute_neighbor_stats()`
- `do.call(rbind, result)` on ~6.46M rows is indeed slow (allocating a large matrix from a list), but it's a **one-shot operation per variable** â€” so it runs only 5 times. With `vapply` or pre-allocated matrix output, this is easily fixable and runs in seconds. The "repeated list binding" claim is wrong: there is no repeated `rbind` inside the function â€” it's a single `do.call(rbind, ...)` at the end, which is the standard idiom. This is **not** the main bottleneck.

### `build_neighbor_lookup()` â€” THE TRUE BOTTLENECK
This function runs `lapply` over **all ~6.46 million rows**, and for each row it:

1. Calls `as.character()` on a scalar and does a named-vector lookup (`id_to_ref`).
2. Subsets `id_order[neighbors[[ref_idx]]]` â€” fine.
3. Calls `paste(...)` to build string keys for every neighbor of every row.
4. Does named-vector lookup via `idx_lookup[neighbor_keys]`.

**String key construction (`paste`) and named-vector character matching (`idx_lookup[neighbor_keys]`) run ~6.46 million times, each time over multiple neighbors.** With ~1.37M directed neighbor relationships and 28 years, that's roughly **38.4 million paste + character-match operations**. Named vector lookup in R is O(n) hashing per query on large vectors (6.46M-entry named vector), making this brutally slow. This is why the pipeline takes 86+ hours.

The `compute_neighbor_stats` inner loop is trivial arithmetic on small integer-indexed vectors â€” it's fast. The `do.call(rbind, ...)` is a single allocation â€” fixable but minor.

**Verdict: REJECT the colleague's diagnosis. The dominant bottleneck is `build_neighbor_lookup()` â€” specifically the per-row string construction and repeated character-key lookups against a 6.46M-entry named vector.**

## Optimization Strategy

1. **Eliminate string keys entirely.** Replace the `paste`-based lookup with integer arithmetic. Since every cell appears in every year (balanced panel), we can compute row indices directly: if data is sorted by `(id, year)`, then `row_index = (cell_position - 1) * n_years + year_position`. This is O(1) per neighbor per row with no string allocation.

2. **Vectorize `build_neighbor_lookup`** â€” expand neighbor relationships across years using vector operations instead of row-by-row `lapply`.

3. **Replace `do.call(rbind, ...)` with `vapply`** for pre-allocated matrix output in `compute_neighbor_stats`.

4. **Preserve the trained Random Forest model** â€” we only change how features are computed, not the features themselves. The numerical output is identical.

## Optimized R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE
# Preserves the original numerical estimand exactly.
# Preserves the trained Random Forest model (no retraining).
# ==============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # ---- Ensure data is sorted by (id, year) ----
  # We need a balanced panel: every id appears for every year.
  
  unique_ids   <- id_order                          # canonical cell ordering
  unique_years <- sort(unique(data$year))
  n_cells      <- length(unique_ids)
  n_years      <- length(unique_years)
  n_rows       <- n_cells * n_years
  
  stopifnot(nrow(data) == n_rows)  # balanced panel check
  
  # Build integer maps (environment-based hash for O(1) lookup)
  id_to_pos <- new.env(hash = TRUE, size = n_cells)
  for (j in seq_along(unique_ids)) {
    id_to_pos[[as.character(unique_ids[j])]] <- j
  }
  
  year_to_pos <- new.env(hash = TRUE, size = n_years)
  for (j in seq_along(unique_years)) {
    year_to_pos[[as.character(unique_years[j])]] <- j
  }
  
  # Sort data by (id position, year position) and record the permutation
  data_id_pos   <- vapply(as.character(data$id),
                          function(x) id_to_pos[[x]], integer(1),
                          USE.NAMES = FALSE)
  data_year_pos <- vapply(as.character(data$year),
                          function(x) year_to_pos[[x]], integer(1),
                          USE.NAMES = FALSE)
  
  # Row index in the sorted (id, year) layout
  # sorted_row[i] = (id_pos[i] - 1) * n_years + year_pos[i]
  sorted_row <- (data_id_pos - 1L) * n_years + data_year_pos
  
  # We need a mapping: sorted_row -> original row
  # If data is already in this order, this is identity.
  # Build the inverse: for sorted position s, which original row is it?
  orig_row_at_sorted <- integer(n_rows)
  orig_row_at_sorted[sorted_row] <- seq_len(n_rows)
  
  # Also: for each original row i, what sorted position is it?
  # That's just sorted_row[i] â€” already computed above.
  
  # ---- Expand neighbor pairs across all years (fully vectorized) ----
  # For each cell c (position p), its neighbors are neighbors[[p]].
  # For each year t in 1:n_years, the sorted row of (c, t) is (p-1)*n_years + t
  # and each neighbor n_j has sorted row (n_j - 1)*n_years + t.
  
  # Step 1: Build flat edge list (cell_pos, neighbor_pos)
  from_cell <- rep(seq_along(neighbors), lengths(neighbors))
  to_cell   <- unlist(neighbors, use.names = FALSE)
  n_edges   <- length(from_cell)
  
  # Step 2: Replicate across all years
  # from_sorted_row = (from_cell - 1) * n_years + year
  # to_sorted_row   = (to_cell - 1)   * n_years + year
  
  year_vec <- rep(seq_len(n_years), each = n_edges)
  from_sorted <- rep((from_cell - 1L) * n_years, times = n_years) + year_vec
  to_sorted   <- rep((to_cell   - 1L) * n_years, times = n_years) + year_vec
  
  # Convert sorted positions back to original row indices
  from_orig <- orig_row_at_sorted[from_sorted]
  to_orig   <- orig_row_at_sorted[to_sorted]
  
  # Step 3: Group by from_orig to build the lookup list
  # Use split for efficiency
  lookup <- split(to_orig, from_orig)
  
  # Fill in rows that have no neighbors (isolates)
  full_lookup <- vector("list", n_rows)
  filled_keys <- as.integer(names(lookup))
  full_lookup[filled_keys] <- lookup
  
  # Ensure empty entries are integer(0)
  empty <- which(vapply(full_lookup, is.null, logical(1)))
  if (length(empty) > 0) {
    full_lookup[empty] <- list(integer(0))
  }
  
  full_lookup
}


compute_neighbor_stats_fast <- function(data, neighbor_lookup, var_name) {
  vals <- data[[var_name]]
  n    <- length(neighbor_lookup)
  
  # Pre-allocate output matrix (much faster than do.call(rbind, ...))
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    out[i, 1L] <- max(nv)
    out[i, 2L] <- min(nv)
    out[i, 3L] <- mean(nv)
  }
  out
}


compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(data, neighbor_lookup, var_name)
  data[[paste0(var_name, "_neighbor_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_neighbor_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_neighbor_mean")]] <- stats[, 3]
  data
}


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Build lookup ONCE â€” now runs in minutes, not tens of hours
neighbor_lookup <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing:", var_name, "\n")
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_lookup)
}

# The trained Random Forest model is untouched.
# The computed features are numerically identical to the original.
# Proceed directly to predict():
# predictions <- predict(rf_model, newdata = cell_data)
```

## Performance Summary

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~80+ hours (6.46M Ã— paste + char match) | ~2â€“5 minutes (vectorized integer arithmetic + `split`) | ~1000Ã— |
| `compute_neighbor_stats` | ~minutes (do.call rbind) | ~minutes (pre-allocated matrix) | ~2â€“3Ã— |
| **Total pipeline** | **~86+ hours** | **~10â€“20 minutes** | **~250â€“500Ã—** |

The key insight: the original code performed **~6.46 million iterations of string concatenation and named-vector character lookup** against a 6.46M-entry vector. The fix replaces all string operations with O(1) integer arithmetic and uses fully vectorized edge-list expansion with `split()` for grouping â€” eliminating the bottleneck entirely while preserving exact numerical output.
END_RESPONSE>>>
