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
run_id: Anthropic_F12_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not free, it is a single matrix assembly step per variable (5 calls total). Each call binds ~6.46M rows Ã— 3 columns â€” this completes in seconds.

**The true bottleneck is `build_neighbor_lookup()`.**

Here is the evidence from the code:

1. **Massive character key construction and named-vector lookup (O(n) repeated hashing):** For every one of the ~6.46 million rows, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into `idx_lookup`, a named character vector of length 6.46 million. Named vector lookup in R is **linear-time hashing per query**. With an average of ~4 rook neighbors per cell, this means ~25.8 million individual key lookups into a 6.46M-length named vector â€” each requiring a hash probe into a very large hash table.

2. **`as.character()` and `paste()` inside a per-row `lapply` over 6.46M rows:** String allocation, concatenation, and coercion are performed ~6.46 million times. This is extremely slow in R's single-threaded interpreted loop.

3. **The lookup is built once but costs hours:** The `lapply` over 6.46M rows, each doing string operations and named-vector lookups, is the dominant wall-clock cost. The 5 subsequent calls to `compute_neighbor_stats` are comparatively cheap (simple numeric indexing into a pre-extracted vector).

**Quantitative reasoning:** `build_neighbor_lookup` performs ~6.46M iterations Ã— (1 `as.character` + 1 named-vector lookup into `id_to_ref` + ~4 `paste` calls + ~4 named-vector lookups into `idx_lookup` of size 6.46M) = tens of millions of expensive string-hash operations. This is the 86-hour bottleneck.

## Optimization Strategy

1. **Replace all character/string-based lookups with integer arithmetic.** Since years are consecutive integers (1992â€“2019, i.e., 28 years) and cell IDs can be mapped to consecutive integers (1â€“344,208), every `(id, year)` pair maps to a unique integer row index via: `row_index = (cell_integer_index - 1) * 28 + (year - 1991)`. This eliminates all `paste()`, `as.character()`, and named-vector lookups.

2. **Vectorize the neighbor lookup construction** by expanding the neighbor list once using integer math, avoiding the per-row `lapply` entirely.

3. **Keep `compute_neighbor_stats` structure** but replace `do.call(rbind, result)` with direct matrix pre-allocation for marginal improvement.

4. **Preserve the trained Random Forest model** â€” we only change feature engineering speed, not values. The numerical results are identical (same max, min, mean computed on the same neighbor sets).

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup + compute_neighbor_stats
# Produces numerically identical results to the original code.
# =============================================================================

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # ---- Step 1: Build integer mappings (no strings) ----
  n_cells <- length(id_order)
  years   <- sort(unique(data$year))
  n_years <- length(years)
  
  # Map each id in id_order to its positional index (1..n_cells)
  # id_order is already ordered; we need id -> position
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_along(id_order)
  # If IDs are not guaranteed to be small integers, use a hash:
  # But for spatial grid cells they typically are. Fallback:
  if (max(id_order) > 1e8) {
    # Use environment-based hash for very large IDs
    id_to_pos_env <- new.env(hash = TRUE, size = n_cells)
    for (k in seq_along(id_order)) {
      id_to_pos_env[[as.character(id_order[k])]] <- k
    }
  }
  
  year_to_offset <- integer(max(years))
  year_to_offset[years] <- seq_along(years)  # year -> 1..28
  
  # ---- Step 2: Build row-index formula ----
  # We need data to be sorted by (id, year) so that:
  #   row(id, year) = (pos_of_id - 1) * n_years + offset_of_year
  # Check if data is already in this order; if not, create a mapping.
  
  # Compute expected row index for each actual row
  data_pos  <- id_to_pos[data$id]
  data_yoff <- year_to_offset[data$year]
  expected_row <- (data_pos - 1L) * n_years + data_yoff
  
  # If data is sorted by (id, year), expected_row == 1:nrow(data).
  # If not, we build a reindex map: expected_row[i] -> i
  n_rows <- nrow(data)
  is_sorted <- identical(expected_row, seq_len(n_rows))
  
  if (!is_sorted) {
    # Map from canonical index to actual row in data
    canonical_to_actual <- integer(n_cells * n_years)
    canonical_to_actual[expected_row] <- seq_len(n_rows)
    # Some canonical slots may be 0 (missing cell-year combos)
  } else {
    canonical_to_actual <- NULL
  }
  
  # ---- Step 3: For each row, find neighbor rows via integer math ----
  # Instead of lapply over 6.46M rows, vectorize:
  #   For each row i with cell position p and year offset y,
  #   neighbors are at positions neighbors[[p]], same year offset y.
  
  # Pre-expand neighbor list into a two-column matrix (cell_pos, neighbor_pos)
  # Then broadcast across years.
  
  n_neighbor_pairs <- sum(lengths(neighbors))  # ~1.37M directed pairs
  
  # Build edge list: (cell_position, neighbor_position)
  cell_positions <- rep(seq_along(neighbors), lengths(neighbors))
  neighbor_positions <- unlist(neighbors, use.names = FALSE)
  
  # For each year, each (cell_pos, neighbor_pos) pair maps to:
  #   source_canonical = (cell_pos - 1) * n_years + year_offset
  #   target_canonical = (neighbor_pos - 1) * n_years + year_offset
  
  # Expand across all years
  year_offsets <- seq_len(n_years)
  
  # Total entries: n_neighbor_pairs * n_years
  # ~1.37M * 28 = ~38.5M entries â€” fits in memory easily
  
  # Source row (canonical) for each (edge, year)
  src_canonical <- rep((cell_positions - 1L) * n_years, times = n_years) +
    rep(year_offsets, each = n_neighbor_pairs)
  
  tgt_canonical <- rep((neighbor_positions - 1L) * n_years, times = n_years) +
    rep(year_offsets, each = n_neighbor_pairs)
  
  # Map canonical to actual row indices
  if (!is_sorted) {
    src_actual <- canonical_to_actual[src_canonical]
    tgt_actual <- canonical_to_actual[tgt_canonical]
    # Remove entries where source or target doesn't exist in data
    valid <- src_actual > 0L & tgt_actual > 0L
    src_actual <- src_actual[valid]
    tgt_actual <- tgt_actual[valid]
  } else {
    src_actual <- src_canonical
    tgt_actual <- tgt_canonical
    valid_mask <- src_actual >= 1L & src_actual <= n_rows &
      tgt_actual >= 1L & tgt_actual <= n_rows
    src_actual <- src_actual[valid_mask]
    tgt_actual <- tgt_actual[valid_mask]
  }
  
  rm(src_canonical, tgt_canonical, cell_positions, neighbor_positions)
  gc()
  
  # ---- Step 4: Return as a sparse edge list (much faster than list-of-vectors) ----
  # Sort by source for grouped operations
  ord <- order(src_actual)
  list(
    src = src_actual[ord],
    tgt = tgt_actual[ord],
    n_rows = n_rows
  )
}


compute_neighbor_stats_fast <- function(data, neighbor_edge_list, var_name) {
  vals <- data[[var_name]]
  src  <- neighbor_edge_list$src
  tgt  <- neighbor_edge_list$tgt
  n    <- neighbor_edge_list$n_rows
  
  neighbor_vals <- vals[tgt]
  
  # Use data.table for ultra-fast grouped aggregation
  if (requireNamespace("data.table", quietly = TRUE)) {
    dt <- data.table::data.table(
      src = src,
      val = neighbor_vals
    )
    # Remove NA values before aggregation
    dt <- dt[!is.na(val)]
    
    agg <- dt[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = src]
    
    # Build result matrix (n_rows x 3), NA for rows with no valid neighbors
    result <- matrix(NA_real_, nrow = n, ncol = 3)
    result[agg$src, 1] <- agg$nb_max
    result[agg$src, 2] <- agg$nb_min
    result[agg$src, 3] <- agg$nb_mean
    
  } else {
    # Base R fallback using tapply
    valid <- !is.na(neighbor_vals)
    s <- src[valid]
    v <- neighbor_vals[valid]
    
    result <- matrix(NA_real_, nrow = n, ncol = 3)
    
    grp_max  <- tapply(v, s, max)
    grp_min  <- tapply(v, s, min)
    grp_mean <- tapply(v, s, mean)
    
    idx <- as.integer(names(grp_max))
    result[idx, 1] <- as.numeric(grp_max)
    result[idx, 2] <- as.numeric(grp_min)
    result[idx, 3] <- as.numeric(grp_mean)
  }
  
  result
}


compute_and_add_neighbor_features_fast <- function(data, var_name, neighbor_edge_list) {
  stats <- compute_neighbor_stats_fast(data, neighbor_edge_list, var_name)
  
  data[[paste0(var_name, "_nb_max")]]  <- stats[, 1]
  data[[paste0(var_name, "_nb_min")]]  <- stats[, 2]
  data[[paste0(var_name, "_nb_mean")]] <- stats[, 3]
  
  data
}


# =============================================================================
# MAIN EXECUTION â€” drop-in replacement for the original outer loop
# =============================================================================

# Build the fast integer-based edge list (runs in seconds, not hours)
neighbor_edge_list <- build_neighbor_lookup_fast(
  cell_data, id_order, rook_neighbors_unique
)

# Same 5 neighbor source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cell_data <- compute_and_add_neighbor_features_fast(
    cell_data, var_name, neighbor_edge_list
  )
}

# The trained Random Forest model is untouched.
# cell_data now has the same neighbor feature columns with identical values.
# Proceed to predict:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations of `paste()` + named-vector hash lookups into a 6.46M-entry table | Eliminated entirely |
| **Key technique** | Character string keys, per-row `lapply` | Integer arithmetic `(pos-1)*n_years + year_offset`, fully vectorized |
| **`compute_neighbor_stats`** | `lapply` over 6.46M rows + `do.call(rbind, ...)` (colleague's claim â€” minor cost) | `data.table` grouped aggregation on ~38.5M-row edge table |
| **Estimated runtime** | 86+ hours | **Minutes** (lookup build: ~10â€“30s; 5 variable aggregations: ~2â€“5 min total) |
| **Numerical output** | max, min, mean per cell-year | Identical max, min, mean per cell-year |
| **RF model** | Preserved | Preserved (unchanged) |

The colleague misidentified a **symptom** (`do.call(rbind, ...)`) as the **cause**. The real bottleneck is the O(n Ã— k) string-hashing inside `build_neighbor_lookup()`, where n = 6.46M rows and each iteration performs multiple `paste()` and named-vector lookups against a 6.46M-entry hash table. Replacing all string operations with integer arithmetic and vectorizing the neighbor expansion across years reduces the runtime from days to minutes.
END_RESPONSE>>>
