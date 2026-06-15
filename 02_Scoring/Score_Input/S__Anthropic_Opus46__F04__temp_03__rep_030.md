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
run_id: Anthropic_F04_T03_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The suspicion is correct: **the spatial neighbor feature construction is the computational bottleneck**, not Random Forest inference. Here's why:

### `build_neighbor_lookup` â€” O(N) character-key lookups via named vectors

The function creates a named-vector lookup (`idx_lookup`) with **~6.46 million entries**, then for every one of those 6.46 million rows, it:

1. Converts the cell ID to character and looks it up in `id_to_ref` (named vector scan).
2. Builds character paste-keys for every neighbor.
3. Looks those keys up in the 6.46M-element named character vector `idx_lookup`.

Named vector lookup in R is **O(n)** linear search per query (R's named vectors use a linear-scan CHARSXP cache, not a hash table). With ~6.46M rows and an average of ~8 neighbors per cell (1,373,394 directed relationships / ~344K cells â‰ˆ 4 per cell, but rook adjacency on a grid is typically 4, yielding ~4 lookups per row), this produces roughly **6.46M Ã— 4 = ~25.8 million** individual named-vector lookups, each scanning a 6.46M-length vector. This is catastrophically slow â€” effectively **O(NÂ²)** behavior.

### `compute_neighbor_stats` â€” Repeated per-row R-level loops

The function calls `lapply` over 6.46M rows, executing R-level subsetting, `is.na` filtering, and three summary functions (`max`, `min`, `mean`) per row. This is called **5 times** (once per source variable). That's ~32.3 million R-level anonymous function invocations with small-vector allocations. The overhead is enormous.

### Summary of bottlenecks

| Component | Root Cause | Severity |
|---|---|---|
| `idx_lookup[neighbor_keys]` | Named-vector lookup is O(N) per query, not O(1) | **Critical** |
| `paste(...)` key construction | Millions of string allocations | High |
| `lapply` in `build_neighbor_lookup` | 6.46M R-level iterations with per-iteration allocations | High |
| `lapply` in `compute_neighbor_stats` | 6.46M Ã— 5 R-level iterations | High |
| `do.call(rbind, result)` | Binding 6.46M 3-element vectors into a matrix | Moderate |

---

## Optimization Strategy

### Principle: Replace character-key lookups with integer-arithmetic indexing; replace row-level `lapply` with vectorized `data.table` grouped operations.

**Key insight:** Since the panel is balanced (344,208 cells Ã— 28 years), and years are contiguous 1992â€“2019, we can compute the row index of any (cell, year) pair with pure integer arithmetic:

```
row_index = (cell_position - 1) * n_years + (year - min_year) + 1
```

This eliminates all string construction and all named-vector lookups entirely â€” going from O(N) per lookup to **O(1)**.

Then, instead of iterating row-by-row in R, we:

1. **Expand** the neighbor list into a long-form edge table (cell_position â†’ neighbor_position), which has ~1.37M rows.
2. **Cross-join** with years (28), yielding ~38.5M (source_row, neighbor_row) pairs.
3. **Join** the variable values by integer row index.
4. **Group-by** source_row and compute `max`, `min`, `mean` in one vectorized `data.table` operation.

This replaces ~32.3M R-level function calls with a single vectorized grouped aggregation.

### Expected speedup

| Operation | Before | After |
|---|---|---|
| Lookup construction | ~hours (O(NÂ²) string matching) | ~seconds (integer arithmetic) |
| Neighbor stats (per variable) | ~hours (6.46M `lapply` iterations) | ~30â€“60 seconds (`data.table` grouped agg) |
| **Total (5 variables)** | **86+ hours** | **~5â€“10 minutes** |

### Constraints preserved

- âœ… Trained Random Forest model is untouched.
- âœ… Numerical results are identical (same max, min, mean over same neighbor sets).
- âœ… Fits in 16 GB RAM (edge table ~38.5M rows Ã— 3 integer columns â‰ˆ ~900 MB peak).

---

## Working R Code

```r
# =============================================================================
# Optimized spatial neighbor feature construction
# =============================================================================
# Requirements: data.table
# install.packages("data.table") # if not already installed

library(data.table)

#' Compute and attach neighbor summary features for multiple variables.
#'
#' @param cell_data       Data.frame (or data.table) with columns: id, year,
#'                        and all columns named in `neighbor_source_vars`.
#'                        Must be a balanced panel (every id appears in every year).
#' @param id_order        Integer/character vector of unique cell IDs in the same
#'                        order used to construct the nb object.
#' @param neighbors       spdep nb object (list of integer index vectors),
#'                        aligned to `id_order`.
#' @param neighbor_source_vars Character vector of variable names to summarize.
#'
#' @return cell_data with new columns: {var}_neighbor_max, {var}_neighbor_min,
#'         {var}_neighbor_mean for each var in neighbor_source_vars.
add_neighbor_features_optimized <- function(cell_data,
                                            id_order,
                                            neighbors,
                                            neighbor_source_vars) {

  # -- Convert to data.table if needed; work on a copy to avoid side effects --
  dt <- as.data.table(cell_data)

  # -- Step 1: Establish integer-arithmetic row indexing -----------------------
  #
  # We sort by (id, year) so that row index = (cell_pos - 1) * n_years + year_pos.
  # This is the key optimization: O(1) row lookup via arithmetic, no strings.

  years_all  <- sort(unique(dt$year))
  n_years    <- length(years_all)
  min_year   <- min(years_all)

  # Map each id to its position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Add cell_pos and sort; this determines the row layout
  dt[, cell_pos := id_to_pos[as.character(id)]]
  setorder(dt, cell_pos, year)

  # After sorting, row i corresponds to:
  #   cell_pos = ((i-1) %/% n_years) + 1
  #   year     = min_year + ((i-1) %% n_years)
  # Verify the layout is correct (balanced panel check)
  stopifnot(
    nrow(dt) == length(id_order) * n_years,
    all(dt$year == rep(years_all, times = length(id_order)))
  )

  # -- Step 2: Build long-form edge table (cell_pos -> neighbor_pos) -----------
  #
  # ~1.37M rows. We strip the spdep nb 0-neighbor sentinel.

  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(cell_pos = i, neighbor_pos = nb)
  }))

  # -- Step 3: Expand edges across all years -----------------------------------
  #
  # For each (cell_pos, neighbor_pos) pair and each year offset, compute
  # the source row index and neighbor row index via arithmetic.
  #
  # source_row   = (cell_pos - 1) * n_years + year_offset
  # neighbor_row = (neighbor_pos - 1) * n_years + year_offset
  #
  # year_offset runs from 1 to n_years.

  year_offsets <- data.table(year_offset = seq_len(n_years))

  # Cross join: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
  edges_by_year <- edge_list[
    rep(seq_len(.N), each = n_years)
  ][, year_offset := rep(seq_len(n_years), times = nrow(edge_list))]

  edges_by_year[, `:=`(
    source_row   = (cell_pos - 1L) * n_years + year_offset,
    neighbor_row = (neighbor_pos - 1L) * n_years + year_offset
  )]

  # Free memory
  edge_list <- NULL

  # -- Step 4: For each variable, vectorized grouped aggregation ---------------

  for (var_name in neighbor_source_vars) {

    cat(sprintf("Computing neighbor features for: %s\n", var_name))

    # Extract the variable as a plain numeric vector aligned to dt's row order
    vals <- dt[[var_name]]

    # Attach neighbor values by integer index (vectorized, no lookup)
    edges_by_year[, nval := vals[neighbor_row]]

    # Grouped aggregation: max, min, mean per source_row, ignoring NAs
    agg <- edges_by_year[
      !is.na(nval),
      .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ),
      keyby = source_row
    ]

    # Initialize result columns with NA
    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")

    set(dt, j = col_max,  value = NA_real_)
    set(dt, j = col_min,  value = NA_real_)
    set(dt, j = col_mean, value = NA_real_)

    # Place aggregated values into the correct rows
    set(dt, i = agg$source_row, j = col_max,  value = agg$nb_max)
    set(dt, i = agg$source_row, j = col_min,  value = agg$nb_min)
    set(dt, i = agg$source_row, j = col_mean, value = agg$nb_mean)

    # Clean up temp column
    edges_by_year[, nval := NULL]

    cat(sprintf("  Done. Non-NA rows: %d / %d\n", nrow(agg), nrow(dt)))
  }

  # -- Step 5: Restore original row order and return ---------------------------

  # Remove helper column
  dt[, cell_pos := NULL]

  # Restore original row order: we need to map back.
  # The safest approach: add original row index before sorting, then re-sort.
  # Since we already sorted dt, we need to recover the original order.
  # REVISED: we should have saved it. Let's use id+year merge instead.


  # Actually, let's redo this more carefully by preserving original order
  # from the start. We'll use an index column.

  # --- IMPORTANT: the above code assumed we could sort dt in place. ---
  # --- Below is the corrected full version that preserves row order. ---

  # (The computations above are correct; we just need to map results back.)
  # Since we set() values by sorted-row indices, the results are in dt
  # which is sorted by (cell_pos, year). We restore original order:

  # We didn't save original order above, so we convert back via merge.
  # This is the simplest safe approach:
  as.data.frame(dt)
}


# =============================================================================
# CORRECTED FULL VERSION (preserves original row order cleanly)
# =============================================================================

add_neighbor_features_optimized <- function(cell_data,
                                            id_order,
                                            neighbors,
                                            neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  dt[, .orig_row := .I]

  years_all <- sort(unique(dt$year))
  n_years   <- length(years_all)

  # Map id -> position in id_order

  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # Sort and record the sorted-to-original mapping
  setorder(dt, cell_pos, year)
  sorted_to_orig <- dt$.orig_row  # sorted_to_orig[i] = original row of sorted row i

  # Verify balanced panel
  stopifnot(nrow(dt) == length(id_order) * n_years)

  # Build edge list (cell_pos -> neighbor_pos)
  edge_list <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(cell_pos = i, neighbor_pos = nb)
  }))

  # Expand across years
  n_edges <- nrow(edge_list)
  edges_by_year <- edge_list[rep(seq_len(n_edges), each = n_years)]
  edges_by_year[, year_offset := rep(seq_len(n_years), times = n_edges)]
  edge_list <- NULL  # free

  edges_by_year[, `:=`(
    source_row   = (cell_pos - 1L) * n_years + year_offset,
    neighbor_row = (neighbor_pos - 1L) * n_years + year_offset
  )]

  # Drop columns no longer needed to save memory
  edges_by_year[, c("cell_pos", "neighbor_pos", "year_offset") := NULL]

  # Compute features for each variable
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  [neighbor features] %s ...\n", var_name))

    vals <- dt[[var_name]]
    edges_by_year[, nval := vals[neighbor_row]]

    agg <- edges_by_year[
      !is.na(nval),
      .(nb_max = max(nval), nb_min = min(nval), nb_mean = mean(nval)),
      keyby = source_row
    ]

    col_max  <- paste0(var_name, "_neighbor_max")
    col_min  <- paste0(var_name, "_neighbor_min")
    col_mean <- paste0(var_name, "_neighbor_mean")

    set(dt, j = col_max,  value = NA_real_)
    set(dt, j = col_min,  value = NA_real_)
    set(dt, j = col_mean, value = NA_real_)

    set(dt, i = agg$source_row, j = col_max,  value = agg$nb_max)
    set(dt, i = agg$source_row, j = col_min,  value = agg$nb_min)
    set(dt, i = agg$source_row, j = col_mean, value = agg$nb_mean)

    edges_by_year[, nval := NULL]
  }

  # Restore original row order
  setorder(dt, .orig_row)
  dt[, c(".orig_row", "cell_pos") := NULL]

  as.data.frame(dt)
}


# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_optimized(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is then applied to cell_data as before.
# predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Concern | Guarantee |
|---|---|
| **Same neighbor sets** | The edge list is built from the identical `neighbors` (nb) object and expanded per year with the same year-matching logic as the original `paste(id, year)` key approach. |
| **Same aggregation functions** | `max`, `min`, `mean` are computed on the identical non-NA neighbor value subsets. |
| **Same NA handling** | Rows with zero valid neighbors receive `NA` for all three stats, exactly as the original. |
| **Row alignment** | The `.orig_row` index ensures the final data.frame has rows in the same order as the input, so downstream column binding or RF prediction is unaffected. |
| **RF model untouched** | No retraining; the code only constructs features and then the existing model's `predict()` is called. |

---

## Performance Summary

| Metric | Original | Optimized |
|---|---|---|
| Lookup method | Named-vector string match, O(N) per query | Integer arithmetic, O(1) per query |
| Row iteration | `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` grouped aggregation |
| Estimated wall time | **86+ hours** | **~5â€“10 minutes** |
| Peak RAM | Moderate (many small allocations, GC pressure) | ~2â€“3 GB (one 38.5M-row edge table) |
| Numerical equivalence | â€” | âœ… Identical results |
END_RESPONSE>>>
