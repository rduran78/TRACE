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
run_id: Anthropic_F12_T07_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F12_Anthropic_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, result)` on ~6.46 million small 3-element vectors is not free, it is a single vectorized bind operation per variable (only 5 calls total). Each inner function returns a fixed-length numeric vector of length 3 â€” there is no repeated list growing or binding inside the function.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Row-level `paste()` and character-keyed lookups over 6.46M rows**: The function creates `idx_lookup` as a named vector with ~6.46M entries keyed by `paste(id, year, sep="_")`. Then, for *each* of the 6.46M rows, it performs `paste()` on neighbor IDs, and does named character vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) per lookup in the worst case (hashed, but still slow at scale with millions of character keys).

2. **Massive `lapply` over 6.46M iterations with per-row string operations**: Each iteration involves `as.character()`, `paste()`, named vector subsetting, and `is.na` filtering. With an average of ~4 rook neighbors per cell, this means ~25.8 million `paste()` calls and ~25.8 million hash lookups inside the loop â€” all in interpreted R.

3. **This function is called once, but it dominates wall time**: The 86+ hour runtime is overwhelmingly attributable to this single function. `compute_neighbor_stats()` is comparatively cheap â€” it does integer indexing into a numeric vector (fast) and simple `max`/`min`/`mean` on small neighbor sets.

**Summary**: The deep bottleneck is the O(N Ã— k) character-key construction and lookup inside `build_neighbor_lookup()`, where N â‰ˆ 6.46M and k â‰ˆ 4. The `compute_neighbor_stats()` function is already reasonably efficient.

---

## Optimization Strategy

1. **Replace character-keyed lookup with integer arithmetic**: Instead of `paste(id, year)` â†’ character key â†’ named vector lookup, encode the lookup as a 2D integer index: `(cell_ref, year_offset)` â†’ row number, stored in an integer matrix. Matrix indexing in R is O(1).

2. **Vectorize the neighbor lookup construction**: Pre-expand all neighbor relationships into a long-form data structure, compute all keys at once using vectorized integer arithmetic, and perform a single merge/match operation instead of 6.46M individual lookups.

3. **Vectorize `compute_neighbor_stats()`**: Replace the `lapply` + `do.call(rbind, ...)` with grouped vectorized operations using `data.table` or direct C-level vectorized code.

4. **Preserve the trained Random Forest model**: The output column names and numerical values remain identical; only the computational path changes.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup using integer-matrix indexing
# ==============================================================================
build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # Convert to data.table for speed (non-destructive)
  dt <- as.data.table(data)
  dt[, row_idx := .I]
  
  # Create integer mappings
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
  
  # Unique years as sorted integer vector; map year -> year_offset
  unique_years <- sort(unique(dt$year))
  year_to_offset <- setNames(seq_along(unique_years), as.character(unique_years))
  
  n_ids <- length(id_order)
  n_years <- length(unique_years)
  
  # Build a (cell_ref, year_offset) -> row_idx matrix
  # This replaces the expensive character-keyed named vector
  row_matrix <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
  
  cell_refs <- id_to_ref[as.character(dt$id)]
  year_offsets <- year_to_offset[as.character(dt$year)]
  row_matrix[cbind(cell_refs, year_offsets)] <- dt$row_idx
  
  # Now expand all neighbor relationships vectorized
  # For each row i: get its cell_ref, get neighbor cell_refs, pair with same year_offset
  
  # Step 1: For each row, get cell_ref and year_offset (already computed)
  # Step 2: Expand neighbors - build a long-form table of (row_i, neighbor_cell_ref)
  
  # Get neighbor lists for each cell_ref (not each row â€” only n_ids lists)
  # Then map to rows
  
  # Build: for each cell_ref, its neighbor cell_refs
  # neighbors is an nb object: list of length n_ids, each element is integer vector of neighbor indices
  
  # Expand: for each row, the neighbor cell refs
  n_neighbors <- lengths(neighbors)  # per cell_ref
  
  # Map each row to its cell_ref
  row_cell_refs <- cell_refs  # length = nrow(data)
  row_year_offs <- year_offsets  # length = nrow(data)
  
  # For each row i, neighbors are: neighbors[[row_cell_refs[i]]]
  # We need to look up row_matrix[ neighbor_cell_ref, row_year_offs[i] ]
  
  # Vectorized expansion:
  # rep each row index by the number of neighbors its cell has
  n_neigh_per_row <- n_neighbors[row_cell_refs]
  
  # Total directed neighbor-row pairs
  total_pairs <- sum(as.numeric(n_neigh_per_row))
  
  # Parent row indices (repeated)
  parent_rows <- rep.int(seq_len(nrow(dt)), n_neigh_per_row)
  
  # Neighbor cell refs (the actual neighbor indices into id_order)
  # For each cell_ref, unlist neighbors to get neighbor refs
  all_neighbor_refs <- unlist(neighbors[row_cell_refs], use.names = FALSE)
  
  # Year offsets for each pair (same year as parent)
  pair_year_offs <- row_year_offs[parent_rows]
  
  # Look up the row index of each neighbor in the same year
  neighbor_row_idx <- row_matrix[cbind(all_neighbor_refs, pair_year_offs)]
  
  # Remove NA pairs (neighbor doesn't exist in that year)
  valid <- !is.na(neighbor_row_idx)
  parent_rows_valid <- parent_rows[valid]
  neighbor_row_idx_valid <- neighbor_row_idx[valid]
  
  # Return as a data.table for grouped operations downstream
  list(
    parent_row = parent_rows_valid,
    neighbor_row = neighbor_row_idx_valid,
    n_data_rows = nrow(dt)
  )
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats using vectorized grouped operations
# ==============================================================================
compute_neighbor_stats_fast <- function(data, neighbor_info, var_name) {
  vals <- data[[var_name]]
  n <- neighbor_info$n_data_rows
  
  # Extract neighbor values
  neighbor_vals <- vals[neighbor_info$neighbor_row]
  parent_rows <- neighbor_info$parent_row
  
  # Remove pairs where the neighbor value is NA
  valid <- !is.na(neighbor_vals)
  parent_rows <- parent_rows[valid]
  neighbor_vals <- neighbor_vals[valid]
  
  # Grouped aggregation via data.table
  agg_dt <- data.table(pr = parent_rows, nv = neighbor_vals)
  stats <- agg_dt[, .(
    nmax = max(nv),
    nmin = min(nv),
    nmean = mean(nv)
  ), by = pr]
  
  # Initialize output matrix with NAs
  out <- matrix(NA_real_, nrow = n, ncol = 3)
  out[stats$pr, 1] <- stats$nmax
  out[stats$pr, 2] <- stats$nmin
  out[stats$pr, 3] <- stats$nmean
  
  out
}

# ==============================================================================
# OPTIMIZED compute_and_add_neighbor_features
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_info) {
  stats_mat <- compute_neighbor_stats_fast(data, neighbor_info, var_name)
  
  data[[paste0("n_max_", var_name)]]  <- stats_mat[, 1]
  data[[paste0("n_min_", var_name)]]  <- stats_mat[, 2]
  data[[paste0("n_mean_", var_name)]] <- stats_mat[, 3]
  
  data
}

# ==============================================================================
# MAIN PIPELINE (drop-in replacement)
# ==============================================================================

# Build the optimized neighbor lookup (once)
neighbor_info <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_info)
}

# The trained Random Forest model is untouched.
# Column names (n_max_*, n_min_*, n_mean_*) and numerical values are identical.
# Proceed directly to predict() with the existing model.
```

---

## Why This Works and Performance Expectations

| Step | Original | Optimized |
|---|---|---|
| **Key construction** | 6.46M Ã— `paste()` per row + character hash lookup | Single vectorized `cbind()` integer matrix index |
| **Neighbor expansion** | 6.46M `lapply` iterations, each with `paste()` + named vector subset | One `unlist()` + one vectorized matrix index (`cbind` on ~25.8M pairs) |
| **Stats computation** | 6.46M `lapply` iterations + `do.call(rbind, ...)` | Single `data.table` grouped aggregation on ~25.8M rows |
| **Expected runtime** | ~86+ hours | **~2â€“10 minutes** (depending on disk I/O and RAM pressure) |

**Key insight**: The original code performed ~6.46 million interpreted R loop iterations, each doing string concatenation and hash-table lookups. The optimized version replaces all of this with vectorized integer arithmetic and a single matrix indexing operation, reducing the algorithmic constant by roughly 3â€“4 orders of magnitude. The `compute_neighbor_stats` was never the real problem â€” it was the lookup construction that consumed virtually all of the 86+ hour runtime.
END_RESPONSE>>>
