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
run_id: Anthropic_F12_T10_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F12_Anthropic_temp_10_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague is **partially right but misses the deeper bottleneck**. Let me audit both functions against the data facts:

**`compute_neighbor_stats()`**: With ~6.46M rows, `do.call(rbind, result)` binds 6.46M three-element vectors into a matrix. This is O(n) and takes maybe a few seconds â€” R's `do.call(rbind, ...)` on a list of equal-length numeric vectors is relatively efficient. There is no "repeated list binding" inside the function; it uses `lapply` (pre-allocated list) then a single `rbind`. This is **not the main bottleneck**.

**`build_neighbor_lookup()` is the true deep bottleneck.** Here's why:

1. **`paste()` and named-vector lookup for every row**: For each of ~6.46M rows, it calls `as.character(data$id[i])`, indexes into `id_to_ref`, retrieves neighbor IDs, builds `paste(neighbor_ids, year, sep="_")` keys, and does named-vector lookup via `idx_lookup[neighbor_keys]`. Named vector lookup in R is **O(n) hash table probing per call**, and doing this 6.46 million times with an `idx_lookup` vector of length 6.46M is catastrophically slow.

2. **`lapply` over 6.46M rows with per-element R function calls**: Each iteration involves multiple R-level operations (string concatenation, subsetting, NA filtering). The overhead of 6.46M R function calls alone is enormous.

3. **The lookup is rebuilt once but takes the vast majority of the 86+ hours**. `compute_neighbor_stats` is called 5 times (once per variable) and is comparatively fast.

**Quantitative estimate**: ~6.46M iterations Ã— ~4 neighbor lookups each Ã— string operations â‰ˆ ~25M+ `paste` and named-vector indexing operations against a 6.46M-length named vector. This is the dominant cost.

## Optimization Strategy

1. **Replace string-key lookups with integer-arithmetic direct indexing.** Instead of `paste(id, year, sep="_")` â†’ named vector lookup, compute row indices directly: if data is sorted by (id, year), then for a given cell `id` at `year`, its row index is `(id_position - 1) * n_years + year_position`. This is O(1) per neighbor, no string ops needed.

2. **Vectorize `build_neighbor_lookup`** using `data.table` or pre-computed integer mappings, eliminating the per-row `lapply`.

3. **Vectorize `compute_neighbor_stats`** by unrolling the neighbor lookup into a two-column edge list (row, neighbor_row) and using grouped aggregation via `data.table`, eliminating per-row R function calls entirely.

4. **Preserve the trained Random Forest model and original numerical estimand** â€” we only change how features are computed, not what is computed. The output columns are numerically identical.

## Working R Code

```r
library(data.table)

# ============================================================
# OPTIMIZED build_neighbor_lookup + compute_neighbor_stats
# replaces both functions and the outer loop
# ============================================================

compute_all_neighbor_features_fast <- function(cell_data,
                                                id_order,
                                                rook_neighbors_unique,
                                                neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # ----------------------------------------------------------
  # 1. Build integer mappings (no string keys)
  # ----------------------------------------------------------
  # Map each unique id to its position in id_order
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Map each unique year to its position
  years_sorted <- sort(unique(dt$year))
  n_years      <- length(years_sorted)
  year_to_pos  <- setNames(seq_along(years_sorted), as.character(years_sorted))

  # Ensure dt is keyed by (id, year) so row index = (id_pos-1)*n_years + year_pos
  # We need dt sorted this way for direct integer indexing
  dt[, id_pos   := id_to_pos[as.character(id)]]
  dt[, year_pos := year_to_pos[as.character(year)]]
  setorder(dt, id_pos, year_pos)

  # After sorting, row i corresponds to (id_pos, year_pos) where:
  #   row_index = (id_pos - 1) * n_years + year_pos
  # Verify this mapping is consistent
  dt[, row_idx := .I]
  # Expected row index from the formula:
  dt[, expected_idx := (id_pos - 1L) * n_years + year_pos]

  # Handle cells that don't have all years (sparse panels)
  # Build a direct lookup: given (id_pos, year_pos) -> row in dt
  # Use a matrix for O(1) access
  n_ids <- length(id_order)
  # If panel is complete (n_ids * n_years == nrow(dt)), use formula;
  # otherwise, use a lookup matrix.
  use_formula <- (n_ids * n_years == nrow(dt))

  if (use_formula) {
    # Complete panel: row = (id_pos - 1)*n_years + year_pos
    message("Complete panel detected. Using direct formula indexing.")
    stopifnot(all(dt$row_idx == dt$expected_idx))
    row_lookup_fn <- function(id_positions, yr_pos) {
      (id_positions - 1L) * n_years + yr_pos
    }
  } else {
    # Sparse panel: build lookup matrix (n_ids x n_years), NA where missing
    message("Sparse panel detected. Using lookup matrix.")
    lookup_mat <- matrix(NA_integer_, nrow = n_ids, ncol = n_years)
    lookup_mat[cbind(dt$id_pos, dt$year_pos)] <- dt$row_idx
    row_lookup_fn <- function(id_positions, yr_pos) {
      lookup_mat[cbind(id_positions, rep(yr_pos, length(id_positions)))]
    }
  }

  # ----------------------------------------------------------
  # 2. Build edge list: (focal_row, neighbor_row) as integers
  #    This replaces build_neighbor_lookup entirely
  # ----------------------------------------------------------
  message("Building edge list...")

  # Convert nb object to an edge list of (focal_id_pos, neighbor_id_pos)
  # rook_neighbors_unique[[i]] gives the neighbor indices for id_order[i]
  edge_ids <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id_pos = i, neighbor_id_pos = as.integer(nb))
  }))

  # Now expand across years: for each year_pos, map (focal_id_pos, year_pos) and

  # (neighbor_id_pos, year_pos) to actual row indices
  message("Expanding edge list across years...")

  edge_list <- rbindlist(lapply(seq_len(n_years), function(yp) {
    el <- copy(edge_ids)
    if (use_formula) {
      el[, focal_row    := (focal_id_pos - 1L) * n_years + yp]
      el[, neighbor_row := (neighbor_id_pos - 1L) * n_years + yp]
    } else {
      el[, focal_row    := row_lookup_fn(focal_id_pos, yp)]
      el[, neighbor_row := row_lookup_fn(neighbor_id_pos, yp)]
    }
    el[!is.na(focal_row) & !is.na(neighbor_row), .(focal_row, neighbor_row)]
  }))

  message(sprintf("Edge list: %s rows", format(nrow(edge_list), big.mark=",")))

  # ----------------------------------------------------------
  # 3. Compute neighbor stats via vectorized grouped aggregation
  #    This replaces compute_neighbor_stats entirely
  # ----------------------------------------------------------
  message("Computing neighbor statistics...")

  for (var_name in neighbor_source_vars) {
    message(sprintf("  Processing variable: %s", var_name))

    # Attach the neighbor's value to each edge
    el <- copy(edge_list)
    el[, nval := dt[[var_name]][neighbor_row]]

    # Remove edges where the neighbor value is NA
    el <- el[!is.na(nval)]

    # Grouped aggregation: max, min, mean per focal_row
    stats <- el[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]

    # Initialize output columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    dt[stats$focal_row, (max_col)  := stats$nb_max]
    dt[stats$focal_row, (min_col)  := stats$nb_min]
    dt[stats$focal_row, (mean_col) := stats$nb_mean]
  }

  # ----------------------------------------------------------
  # 4. Clean up helper columns and return as data.frame
  # ----------------------------------------------------------
  dt[, c("id_pos", "year_pos", "row_idx", "expected_idx") := NULL]

  # Restore original row order if needed (by id, year as in original data)
  setorder(dt, id_pos)  # already dropped; order is (id, year) which is fine

  result <- as.data.frame(dt)
  message("Done.")
  return(result)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# The new columns (neighbor_max_*, neighbor_min_*, neighbor_mean_*)
# are numerically identical to the original implementation.
# Prediction proceeds as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **True bottleneck** | `build_neighbor_lookup`: 6.46M `paste()`+named-vector lookups | Eliminated entirely |
| **Indexing strategy** | String keys in named vector (hash probe per row) | Integer arithmetic: `(id_pos-1)*n_years + year_pos` â†’ O(1) |
| **Neighbor stats** | `lapply` over 6.46M rows + `do.call(rbind, ...)` | Vectorized `data.table` grouped aggregation on edge list |
| **Estimated time** | 86+ hours | **~2â€“10 minutes** (dominated by edge list expansion and 5 grouped aggregations) |
| **Memory** | Moderate (6.46M-element list of integer vectors) | Edge list: ~1.37M edges Ã— 28 years Ã— 2 cols â‰ˆ ~580MB (fits in 16 GB) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical output** | Original estimand | Identical âœ“ |

The colleague's diagnosis of `do.call(rbind, ...)` is a **surface-level concern** â€” it accounts for trivial time relative to the per-row string manipulation and named-vector hashing inside `build_neighbor_lookup()`, which is the true bottleneck consuming the vast majority of the 86+ hour runtime.
END_RESPONSE>>>
