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
run_id: Anthropic_F12_T03_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F12_Anthropic_temp_03_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of 6.46 million small vectors is not ideal, it is a one-shot operation per variable (5 times total) and runs in seconds to low minutes at most. The `lapply` inside `compute_neighbor_stats` is also straightforward vectorized indexing â€” each iteration is O(k) where k is the small number of neighbors (~4 for rook contiguity).

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **`paste()` key construction for 6.46M rows**: `paste(data$id, data$year, sep = "_")` creates 6.46M character keys, and `setNames()` on that vector creates a named lookup. This is moderately expensive but not fatal.

2. **The critical `lapply` over 6.46M rows**: Inside the `lapply`, for *every single row* (6.46 million iterations), the function:
   - Calls `as.character(data$id[i])` â€” character conversion per row.
   - Looks up `id_to_ref[...]` â€” named vector lookup (hash-based, but still per-row overhead).
   - Extracts `neighbors[[ref_idx]]` â€” subset of the nb object.
   - Calls `id_order[neighbors[[ref_idx]]]` â€” integer subsetting.
   - Calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” **constructs new character keys per row for ~4 neighbors**.
   - Looks up `idx_lookup[neighbor_keys]` â€” **named vector lookup of character keys, 6.46M times**.
   - Filters NAs.

   This is **6.46 million R-level iterations** each involving character allocation, `paste()`, and named-vector hash lookups. The R interpreter overhead for this loop dwarfs everything else. With ~4 neighbors per cell and 28 years, this creates roughly **6.46M Ã— 4 = 25.8 million `paste` + hash-lookup operations**, all inside a slow R-level `lapply`.

3. **The lookup is year-redundant**: Every cell has the same neighbors across all 28 years. The function recomputes the *same* neighbor cell IDs for the same spatial cell 28 times (once per year). This means ~344K unique spatial lookups are inflated to 6.46M.

4. **Estimated time**: At even 0.05ms per iteration (conservative for the string operations), 6.46M iterations â‰ˆ 323 seconds â‰ˆ 5.4 minutes just for the lookup. But empirical R overhead for `paste` + named-vector lookup at this scale is much worse â€” likely 30â€“60+ minutes. And this is *before* `compute_neighbor_stats` runs. The 86-hour estimate suggests additional inefficiencies in the broader pipeline or repeated rebuilds, but `build_neighbor_lookup` is the dominant single bottleneck in the code shown.

**In contrast**, `compute_neighbor_stats` is a simple `lapply` doing `vals[idx]` (integer indexing â€” extremely fast), `max`, `min`, `mean` on ~4 values. Even 6.46M iterations of this are fast. The `do.call(rbind, result)` on a list of 6.46M length-3 vectors takes a few seconds.

## Optimization Strategy

1. **Eliminate per-row character operations entirely.** Replace the character-key lookup with integer-arithmetic indexing. Since the data has a regular panel structure (each cell Ã— each year), we can compute row indices arithmetically.

2. **Compute spatial neighbor lookup only once over the 344K unique cells**, then expand to all years via integer offset arithmetic. This reduces the core loop from 6.46M to 344K iterations (18.8Ã— reduction).

3. **Vectorize `compute_neighbor_stats`** using a pre-built sparse matrix or a single grouped C-level operation, replacing `lapply` + `do.call(rbind, ...)` with matrix operations.

4. **Preserve the trained Random Forest model** â€” we only change feature-engineering code, not model structure or predictions. The numerical output is identical.

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” drop-in replacement
# Preserves the trained RF model and original numerical estimand.
# =============================================================================

library(data.table)

# ---- Step 0: Convert to data.table for fast ordered operations ----
# Assume cell_data is a data.frame with columns: id, year, ntl, ec, pop_density, def, usd_est_n2, ...
# Assume id_order is the vector of unique cell IDs in the order matching rook_neighbors_unique (the nb object).
# Assume rook_neighbors_unique is the precomputed spdep::nb list (length = length(id_order)).

build_neighbor_lookup_fast <- function(data, id_order, neighbors) {
  # -------------------------------------------------------------------
  # KEY INSIGHT: build the lookup per unique cell (344K), not per row (6.46M).
  # Then expand to all years via integer arithmetic.
  # -------------------------------------------------------------------
  
  dt <- as.data.table(data)
  
  # Ensure data is sorted by (id, year) so we can use arithmetic indexing
  # Record original order to restore later if needed
  dt[, orig_row := .I]
  setkey(dt, id, year)
  
  # Unique years in sorted order
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_offset <- setNames(seq_along(years) - 1L, as.character(years))
  
  # Map each unique cell id to its block-start row in the sorted data.table
  # In a balanced panel (344208 cells Ã— 28 years), cell i occupies rows
  # ((i-1)*n_years + 1) : (i*n_years), IF every cell has all years.
  
  # Handle potentially unbalanced panels robustly:
  cell_info <- dt[, .(start_row = .I[1], n = .N, years_present = list(year)), by = id]
  
  # Build a fast map from id_order index to cell_info row
  id_to_cell_row <- setNames(seq_len(nrow(cell_info)), as.character(cell_info$id))
  
  # For each cell in id_order, find its neighbor cells' start rows and year vectors
  # We only iterate over 344K unique cells, not 6.46M rows
  n_cells <- length(id_order)
  
  # Pre-extract cell_info vectors for speed
  ci_id        <- as.character(cell_info$id)
  ci_start     <- cell_info$start_row
  ci_n         <- cell_info$n
  ci_years     <- cell_info$years_present  # list of year vectors
  
  # Build the full neighbor_lookup: a list of length nrow(dt),
  # where each element is the integer vector of row indices (in dt) of neighbors
  # sharing the same year.
  
  # Strategy: for each unique cell, get its neighbor cell indices,
  # then for each year that cell has, find matching rows in neighbor cells.
  
  # Pre-build: for each cell, a named vector mapping year -> row index in dt
  # This is 344K cells, each with up to 28 entries â€” very manageable.
  
  cat("Building per-cell year-to-row maps...\n")
  cell_year_maps <- vector("list", nrow(cell_info))
  for (c_idx in seq_len(nrow(cell_info))) {
    s <- ci_start[c_idx]
    n <- ci_n[c_idx]
    yrs <- ci_years[[c_idx]]
    cell_year_maps[[c_idx]] <- setNames(seq.int(s, s + n - 1L), as.character(yrs))
  }
  
  cat("Building neighbor lookup for", nrow(dt), "rows...\n")
  
  # Pre-compute neighbor cell_info indices for each cell in id_order
  neighbor_cell_indices <- vector("list", n_cells)
  for (j in seq_len(n_cells)) {
    nb_ids <- neighbors[[j]]  # indices into id_order
    if (length(nb_ids) == 0L || (length(nb_ids) == 1L && nb_ids[1] == 0L)) {
      neighbor_cell_indices[[j]] <- integer(0)
    } else {
      # Map id_order indices to cell_info row indices
      nb_cell_ids <- as.character(id_order[nb_ids])
      neighbor_cell_indices[[j]] <- as.integer(id_to_cell_row[nb_cell_ids])
    }
  }
  
  # Now build the full lookup: iterate over cells, then over their years
  # Total iterations: ~344K cells Ã— ~4 neighbors Ã— 28 years â‰ˆ 38.5M simple integer lookups
  # But the outer loop is only 344K, and inner work is vectorized.
  
  # Pre-allocate the result list
  lookup <- vector("list", nrow(dt))
  
  for (j in seq_len(n_cells)) {
    cell_id_char <- as.character(id_order[j])
    c_row <- id_to_cell_row[cell_id_char]
    if (is.na(c_row)) next
    
    my_year_map <- cell_year_maps[[c_row]]
    my_years_char <- names(my_year_map)
    my_rows <- as.integer(my_year_map)
    
    nb_c_indices <- neighbor_cell_indices[[j]]
    
    if (length(nb_c_indices) == 0L) {
      for (k in seq_along(my_rows)) {
        lookup[[my_rows[k]]] <- integer(0)
      }
      next
    }
    
    # For each year this cell has, gather neighbor rows for that same year
    for (k in seq_along(my_years_char)) {
      yr_char <- my_years_char[k]
      row_idx <- my_rows[k]
      
      nb_rows <- integer(length(nb_c_indices))
      for (m in seq_along(nb_c_indices)) {
        val <- cell_year_maps[[ nb_c_indices[m] ]][yr_char]
        nb_rows[m] <- if (is.null(val) || is.na(val)) NA_integer_ else as.integer(val)
      }
      lookup[[row_idx]] <- nb_rows[!is.na(nb_rows)]
    }
  }
  
  cat("Neighbor lookup complete.\n")
  
  # Return lookup AND the sort-mapping so we can reorder results back
  list(lookup = lookup, dt = dt)
}


compute_neighbor_stats_fast <- function(dt, neighbor_lookup, var_name) {
  # -------------------------------------------------------------------
  # Replace do.call(rbind, lapply(...)) with pre-allocated matrix
  # -------------------------------------------------------------------
  vals <- dt[[var_name]]
  n <- length(neighbor_lookup)
  
  result_mat <- matrix(NA_real_, nrow = n, ncol = 3)
  
  for (i in seq_len(n)) {
    idx <- neighbor_lookup[[i]]
    if (length(idx) == 0L) next
    nv <- vals[idx]
    nv <- nv[!is.na(nv)]
    if (length(nv) == 0L) next
    result_mat[i, 1] <- max(nv)
    result_mat[i, 2] <- min(nv)
    result_mat[i, 3] <- mean(nv)
  }
  
  result_mat
}


compute_and_add_neighbor_features_fast <- function(dt, var_name, neighbor_lookup) {
  stats <- compute_neighbor_stats_fast(dt, neighbor_lookup, var_name)
  col_max  <- paste0("neighbor_max_", var_name)
  col_min  <- paste0("neighbor_min_", var_name)
  col_mean <- paste0("neighbor_mean_", var_name)
  dt[, (col_max)  := stats[, 1]]
  dt[, (col_min)  := stats[, 2]]
  dt[, (col_mean) := stats[, 3]]
  dt
}


# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Build the optimized neighbor lookup (one-time cost)
result <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
neighbor_lookup <- result$lookup
dt <- result$dt

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Processing neighbor features for:", var_name, "\n")
  dt <- compute_and_add_neighbor_features_fast(dt, var_name, neighbor_lookup)
}

# Restore original row order so downstream RF predictions align
setorder(dt, orig_row)
dt[, orig_row := NULL]

# Convert back to data.frame if the RF predict method expects one
cell_data <- as.data.frame(dt)

# The trained Random Forest model is untouched â€” use it directly:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary of Changes and Expected Speedup

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M iterations with `paste()` + character hash lookups | Identified and fixed |
| **Lookup loop iterations** | 6.46M (every row) | 344K (unique cells only), with lightweight inner year loop |
| **Key mechanism** | Character `paste` + named-vector lookup per row | Integer arithmetic + pre-built per-cell yearâ†’row maps |
| **`compute_neighbor_stats`** | `do.call(rbind, list_of_vectors)` â€” allocates intermediate list | Pre-allocated matrix â€” zero intermediate allocation |
| **Estimated runtime** | 86+ hours | **~5â€“15 minutes** (lookup build ~3â€“8 min, stats ~1â€“2 min per variable) |
| **RF model** | Preserved âœ“ | Unchanged â€” no retraining |
| **Numerical output** | Preserved âœ“ | Identical `max`, `min`, `mean` values |

The 18.8Ã— reduction in outer-loop iterations (344K vs 6.46M), elimination of millions of `paste()` and character-hash operations, and pre-allocated output matrices collectively reduce the pipeline from days to minutes.
END_RESPONSE>>>
