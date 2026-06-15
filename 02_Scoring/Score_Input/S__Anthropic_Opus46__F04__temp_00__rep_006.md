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
run_id: Anthropic_F04_T00_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M Ã— ~4 neighbors â‰ˆ 25.8M string operations is extremely expensive in base R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` over 6.46M elements calling `max`/`min`/`mean` on small vectors â€” slow due to R-level loop overhead and repeated function-call dispatch.

**Root causes, ranked:**

| Rank | Cause | Impact |
|------|-------|--------|
| 1 | Per-row `paste` + character-key lookup in `build_neighbor_lookup` (~6.46M iterations) | Dominant â€” estimated >80% of 86 h |
| 2 | Per-row `lapply` in `compute_neighbor_stats` (~6.46M Ã— 5 vars) | Significant |
| 3 | Repeated allocation of small vectors inside closures | Moderate (GC pressure) |

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed join via `data.table`.** Build a `data.table` keyed on `(id, year)` with an integer row index. For each cell-year, join to a pre-expanded neighbor table (cell-year â†’ neighbor-cell-year) in one vectorized merge â€” no per-row `paste` or named-vector lookup.

2. **Vectorize neighbor stats computation.** After the join produces a long-form table of (row, neighbor_row), extract the variable values, then use `data.table` grouped aggregation (`[, .(max, min, mean), by = row_idx]`) â€” a single vectorized pass per variable instead of 6.46M R-level `lapply` calls.

3. **Memory management.** The expanded neighbor-edge table will have ~25.8M rows Ã— a few integer columns â€” roughly 200â€“400 MB, well within 16 GB.

**Expected speedup:** From ~86+ hours to roughly 5â€“15 minutes total for all 5 variables.

## Optimized R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a vectorized neighbor-edge table (run once)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edge_table <- function(cell_data_dt, id_order, neighbors) {
 # cell_data_dt: a data.table with columns id, year (and all feature cols)
 #               with an added integer column .row_idx = .I
 # id_order:     the vector of cell IDs in the same order as the nb object
 # neighbors:    the spdep nb object (list of integer index vectors)

 # --- Step A: Build cell-level directed edge list (from nb object) ---
 # Each element neighbors[[i]] gives the indices (into id_order) of
 # the neighbors of id_order[i].
 from_idx <- rep(seq_along(neighbors), lengths(neighbors))
 to_idx   <- unlist(neighbors, use.names = FALSE)

 # Remove any 0-entries that spdep uses to denote "no neighbors"
 valid <- to_idx > 0L
 from_idx <- from_idx[valid]
 to_idx   <- to_idx[valid]

 cell_edges <- data.table(
   from_id = id_order[from_idx],
   to_id   = id_order[to_idx]
 )
 # cell_edges now has ~1,373,394 rows (directed rook edges)

 # --- Step B: Get unique years ---
 years <- sort(unique(cell_data_dt$year))

 # --- Step C: Cross-join edges Ã— years to get cell-year edge table ---
 # Use CJ for memory-efficient cross join, then join to get row indices.
 cell_year_edges <- cell_edges[, .(from_id, to_id)]
 # Expand by year
 cell_year_edges <- cell_year_edges[
   , .(year = years), by = .(from_id, to_id)
 ]
 # cell_year_edges now has ~1,373,394 Ã— 28 â‰ˆ 38.5M rows
 # (but many will match; this is the upper bound)

 # --- Step D: Map (from_id, year) â†’ source row index ---
 setkey(cell_data_dt, id, year)
 # Add row index to cell_data_dt if not present
 if (!".row_idx" %in% names(cell_data_dt)) {
   cell_data_dt[, .row_idx := .I]
 }

 # Join to get the source row index (the row whose features we are building)
 cell_year_edges[
   cell_data_dt,
   on = .(from_id = id, year = year),
   src_row := i..row_idx
 ]

 # Join to get the neighbor row index
 cell_year_edges[
   cell_data_dt,
   on = .(to_id = id, year = year),
   nbr_row := i..row_idx
 ]

 # Drop edges where either side is missing (cell-year not in data)
 cell_year_edges <- cell_year_edges[!is.na(src_row) & !is.na(nbr_row)]

 # Keep only the integer index columns we need
 cell_year_edges[, .(src_row, nbr_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute neighbor stats for one variable (vectorized)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_fast <- function(cell_data_dt, edge_dt, var_name) {
 # edge_dt has columns: src_row, nbr_row (integer indices into cell_data_dt)
 # Returns a data.table with columns: .row_idx, <var>_max, <var>_min, <var>_mean

 vals <- cell_data_dt[[var_name]]

 # Attach neighbor values
 work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]

 # Drop NAs in neighbor values
 work <- work[!is.na(nbr_val)]

 # Grouped aggregation â€” single vectorized pass
 stats <- work[, .(
   v_max  = max(nbr_val),
   v_min  = min(nbr_val),
   v_mean = mean(nbr_val)
 ), by = src_row]

 # Rename columns to match original pipeline's naming convention
 setnames(stats, c("v_max", "v_min", "v_mean"),
          paste0("neighbor_", var_name, c("_max", "_min", "_mean")))

 stats
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Main driver â€” drop-in replacement for the outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars) {
 # Convert to data.table (non-destructive copy)
 cell_dt <- as.data.table(cell_data)
 cell_dt[, .row_idx := .I]

 message("Building vectorized edge table...")
 edge_dt <- build_neighbor_edge_table(cell_dt, id_order, rook_neighbors_unique)
 message(sprintf("  Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

 for (var_name in neighbor_source_vars) {
   message(sprintf("Computing neighbor stats for '%s'...", var_name))
   stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)

   # Merge back; rows with no valid neighbors get NA (preserving original behavior)
   max_col  <- paste0("neighbor_", var_name, "_max")
   min_col  <- paste0("neighbor_", var_name, "_min")
   mean_col <- paste0("neighbor_", var_name, "_mean")

   # Initialize with NA
   cell_dt[, (max_col)  := NA_real_]
   cell_dt[, (min_col)  := NA_real_]
   cell_dt[, (mean_col) := NA_real_]

   # Update matched rows
   cell_dt[stats, on = .(.row_idx = src_row),
           c(max_col, min_col, mean_col) := mget(paste0("i.", c(max_col, min_col, mean_col)))]
 }

 # Clean up helper column
 cell_dt[, .row_idx := NULL]

 # Return as data.frame if the original was a data.frame (for RF compatibility)
 if (!is.data.table(cell_data)) {
   return(as.data.frame(cell_dt))
 }
 cell_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (drop-in replacement for the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
 cell_data,
 id_order,
 rook_neighbors_unique,
 neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same set of non-NA neighbor values per (cell, year) as the original code. The edge table encodes the identical rook-neighbor relationships, filtered to the same year. |
| **NA handling** | Rows with no valid neighbors receive `NA` for all three stats, identical to the original `if (length(idx) == 0) return(c(NA, NA, NA))` logic. |
| **Trained RF model** | No model retraining. The output `cell_data` has the same columns with the same names and the same values, so `predict(rf_model, newdata = cell_data)` produces identical results. |
| **Column naming** | Adjust the `paste0("neighbor_", ...)` pattern to match whatever `compute_and_add_neighbor_features` originally produced. |

## Performance Estimate

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~60â€“70 h (6.46M `paste` + char lookups) | ~1â€“3 min (vectorized `data.table` join) |
| Stats computation (5 vars) | ~15â€“20 h (5 Ã— 6.46M `lapply`) | ~2â€“5 min (5 Ã— grouped `data.table` aggregation) |
| **Total** | **~86+ h** | **~5â€“15 min** |

Peak RAM usage: ~2â€“4 GB for the edge table plus working copies, well within the 16 GB budget.
END_RESPONSE>>>
