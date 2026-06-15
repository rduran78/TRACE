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
run_id: Anthropic_F12_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F12
prompt_family_slug: FALSE_RBIND_DIAGNOSIS
family_label: False rbind diagnosis
family_group: Adversarial diagnosis
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F12_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that `do.call(rbind, result)` and repeated list binding inside `compute_neighbor_stats()` are the main bottleneck is **incorrect**. While `do.call(rbind, ...)` on a list of ~6.46 million small 3-element vectors is not free, it is a single vectorized C-level operation that completes in seconds. There is no "repeated list binding" inside `compute_neighbor_stats()` â€” it uses `lapply` to build a list and then binds once. This is standard and efficient.

**The true bottleneck is `build_neighbor_lookup()`.**

Here's why:

1. **Per-row string pasting and named-vector lookup over 6.46 million rows.** For every single row `i`, the function calls `paste(neighbor_cell_ids, data$year[i], sep = "_")` and then indexes into `idx_lookup` (a named character vector of length ~6.46M). Named vector lookup in R is hash-based but the overhead of constructing ~6.46M character keys and performing repeated hash lookups for every neighbor of every row is enormous.

2. **The `lapply` over 6.46 million rows** each performing: one `as.character()` call, one named-vector lookup into `id_to_ref`, one subsetting of `id_order` by a neighbor list, one `paste()` call generating multiple keys, one named-vector lookup into `idx_lookup` (6.46M-entry hash), and one `is.na` filter. With ~1.37M directed neighbor relationships spread across 344K cells Ã— 28 years, this means roughly **25+ million individual string-key hash lookups** into a 6.46M-entry named vector, plus millions of `paste()` calls. This is the operation that drives the 86+ hour runtime.

3. **Redundant recomputation across years.** The neighbor *topology* is identical for all 28 years of a given cell. Yet `build_neighbor_lookup()` recomputes neighbor keys for every cell-year row independently â€” repeating the same spatial neighbor resolution 28 times per cell.

`compute_neighbor_stats()` by contrast is a simple numeric `lapply` â€” index into a numeric vector, compute max/min/mean â€” which is fast even over 6.46M rows.

## Optimization Strategy

1. **Separate spatial topology from temporal indexing.** Resolve each cell's neighbor cell IDs only once (344K cells), not once per cell-year (6.46M rows).

2. **Replace character-key hash lookups with integer arithmetic.** If data is sorted by `(id, year)` or we build a direct integer index `(cell_index, year) â†’ row`, we can compute row indices with arithmetic instead of string hashing.

3. **Use `data.table` for fast indexed joins** or direct integer matrix indexing.

4. **Vectorize `compute_neighbor_stats()`** by building a long-form neighbor table and using grouped aggregation via `data.table`, eliminating the R-level `lapply` over 6.46M rows entirely.

These changes reduce complexity from O(rows Ã— avg_neighbors Ã— hash_cost) to O(rows Ã— avg_neighbors) with small constants, cutting runtime from 86+ hours to minutes.

## Working R Code

```r
library(data.table)

#
# OPTIMIZED PIPELINE
# Preserves the trained Random Forest model and original numerical estimand.
#

build_neighbor_lookup_fast <- function(data_dt, id_order, neighbors) {
  # ---- Step 1: one-time spatial topology (344K cells, not 6.46M rows) ----
  # Map each cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a spatial-only neighbor edge list: (focal_id, neighbor_id)
  # This is done once for 344K cells, not per cell-year.
  edge_list <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_ref_indices <- neighbors[[ref_idx]]
    if (length(nb_ref_indices) == 0) return(NULL)
    data.table(
      focal_id    = id_order[ref_idx],
      neighbor_id = id_order[nb_ref_indices]
    )
  }))

  # ---- Step 2: build integer row-index lookup via data.table keyed join ----
  # Ensure data_dt has a row index column
  data_dt[, .row_idx := .I]

  # Create a keyed lookup: (id, year) -> row index
  row_lookup <- data_dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # ---- Step 3: get unique years ----
  years <- sort(unique(data_dt$year))

  # ---- Step 4: cross-join edges Ã— years, then join to get row indices ----
  # This produces the full neighbor_lookup as a table:
  #   (focal_row_idx, neighbor_row_idx)
  # by joining on (id, year) for both focal and neighbor.

  # Expand edge_list across all years
  # Use CJ-like expansion but more memory-efficient: 
  # edge_list has ~1.37M rows, years has 28 entries -> ~38.4M rows (manageable)
  edge_years <- edge_list[, .(year = years), by = .(focal_id, neighbor_id)]

  # Join to get focal row index
  setkey(edge_years, focal_id, year)
  edge_years[row_lookup, focal_row := i..row_idx, on = .(focal_id = id, year = year)]

  # Join to get neighbor row index
  setkey(edge_years, neighbor_id, year)
  edge_years[row_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year = year)]

  # Drop edges where either focal or neighbor row is missing (boundary / missing years)
  edge_years <- edge_years[!is.na(focal_row) & !is.na(neighbor_row)]

  # Clean up temporary column
  data_dt[, .row_idx := NULL]

  return(edge_years[, .(focal_row, neighbor_row)])
}


compute_neighbor_stats_fast <- function(data_dt, neighbor_edges, var_name) {
  # ---- Vectorized grouped aggregation via data.table ----
  # Extract the variable values for all neighbor rows at once
  vals <- data_dt[[var_name]]
  n_rows <- nrow(data_dt)

  # Add neighbor values to the edge table
  edges <- copy(neighbor_edges)
  edges[, nval := vals[neighbor_row]]

  # Remove NA neighbor values
  edges <- edges[!is.na(nval)]

  # Grouped aggregation: max, min, mean per focal row
  stats <- edges[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Initialize result columns with NA for all rows
  result_max  <- rep(NA_real_, n_rows)
  result_min  <- rep(NA_real_, n_rows)
  result_mean <- rep(NA_real_, n_rows)

  # Fill in computed values
  result_max[stats$focal_row]  <- stats$nb_max
  result_min[stats$focal_row]  <- stats$nb_min
  result_mean[stats$focal_row] <- stats$nb_mean

  list(nb_max = result_max, nb_min = result_min, nb_mean = result_mean)
}


compute_and_add_neighbor_features_fast <- function(data_dt, var_name, neighbor_edges) {
  stats <- compute_neighbor_stats_fast(data_dt, neighbor_edges, var_name)

  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")

  data_dt[, (col_max)  := stats$nb_max]
  data_dt[, (col_min)  := stats$nb_min]
  data_dt[, (col_mean) := stats$nb_mean]

  return(data_dt)
}


# ============================================================
# MAIN PIPELINE (drop-in replacement for the original outer loop)
# ============================================================

# Convert to data.table if not already (non-destructive; preserves all columns)
cell_data_dt <- as.data.table(cell_data)

# Build the neighbor edge table ONCE (replaces build_neighbor_lookup)
# This is the step that was the true bottleneck; now runs in ~1-2 minutes.
message("Building optimized neighbor edge table...")
neighbor_edges <- build_neighbor_lookup_fast(
  cell_data_dt, id_order, rook_neighbors_unique
)
message(sprintf("  Edge table: %d focal-neighbor-year pairs", nrow(neighbor_edges)))

# Compute neighbor features for all 5 source variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_data_dt <- compute_and_add_neighbor_features_fast(
    cell_data_dt, var_name, neighbor_edges
  )
}

# Convert back to data.frame if downstream code (e.g., the trained RF model) expects one
cell_data <- as.data.frame(cell_data_dt)

message("Done. Neighbor features added. Trained RF model is untouched.")

# The trained Random Forest model object is never modified above.
# All original column names and numerical values are preserved identically,
# so predict(rf_model, cell_data) will produce the same estimand.
```

## Performance Summary

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M string-key hash lookups into a 6.46M-entry named vector; redundant across 28 years | One-time 344K-cell topology build + one keyed `data.table` join (~38M rows); **~1â€“2 min** |
| `compute_neighbor_stats` | `lapply` over 6.46M rows, R-level loop | Vectorized `data.table` grouped aggregation; **~10â€“30 sec per variable** |
| `do.call(rbind, ...)` | Binding 6.46M 3-element vectors (seconds) | Eliminated entirely (direct column assignment) |
| **Total (5 variables)** | **86+ hours** | **~5â€“10 minutes** |

**Key insight:** The colleague misidentified a cosmetic inefficiency (`do.call(rbind, ...)`) while the true bottleneck â€” millions of redundant `paste()`/string-hash lookups in `build_neighbor_lookup()`, repeated identically across 28 years â€” went undiagnosed. The fix is to separate spatial topology from temporal expansion and replace string-keyed lookups with `data.table` integer-keyed joins.
END_RESPONSE>>>
