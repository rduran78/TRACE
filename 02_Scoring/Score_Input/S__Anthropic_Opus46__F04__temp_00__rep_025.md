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
run_id: Anthropic_F04_T00_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_025.md

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

1. **Replace string-key lookups with integer-indexed join via `data.table`.** Build a `data.table` keyed on `(id, year)` with an integer row-index column. Expand the neighbor graph into an edge-list `data.table` with columns `(id, neighbor_id)`. Join on `(neighbor_id, year)` to get all neighbor row indices in one vectorized operation â€” no per-row `paste` or named-vector lookup.

2. **Replace per-row `lapply` stats with grouped `data.table` aggregation.** Once we have an edge-list with `(focal_row, neighbor_row)`, pull the variable values by integer index and compute `max`/`min`/`mean` grouped by `focal_row` â€” fully vectorized in C via `data.table`.

3. **Process all 5 variables in one pass** over the edge-list to avoid redundant joins.

**Expected speedup:** From ~86 hours to **minutes** (the join is O(n log n); the grouped aggregation is O(n)).

**Preservation guarantees:**
- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of each neighbor variable) is identical to the original.

## Optimized R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {

  # --- Step 0: Convert to data.table, add row index ---
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- Step 1: Build edge-list from the nb object ---
  # rook_neighbors_unique is a list of integer vectors (indices into id_order).
  # Convert to a two-column data.table: (focal_id, neighbor_id).
  n_cells <- length(id_order)
  focal_indices <- rep(seq_len(n_cells),
                       times = lengths(rook_neighbors_unique))
  neighbor_indices <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove zero-neighbor entries (spdep uses integer(0) for islands)
  valid <- !is.na(neighbor_indices) & neighbor_indices != 0L
  edges <- data.table(
    focal_id    = id_order[focal_indices[valid]],
    neighbor_id = id_order[neighbor_indices[valid]]
  )

  # --- Step 2: Join edges with data to get (focal_row, neighbor_row) per year ---
  # Key the main table for fast join
  setkey(dt, id, year)

  # For each edge (focal_id, neighbor_id), we need every year.
  # Instead of a cross-join, join from the focal side first:
  # Get (focal_id, year, focal_row_idx) then join neighbor side.

  # Create a slim lookup: id -> year -> row_idx
  lookup <- dt[, .(id, year, .row_idx)]
  setkey(lookup, id, year)

  # Expand edges by year: join focal side to get all (focal_id, year) combos
  # But edges Ã— years would be huge. Instead, work per-row:
  # For each row in dt, find its neighbors' rows in the same year.


  # Efficient approach: merge edges with lookup on focal side, then neighbor side.
  # focal join: get year from focal
  focal_info <- dt[, .(focal_id = id, year, focal_row = .row_idx)]

  # Join: for each focal row, attach all its neighbor_ids
  setkey(edges, focal_id)
  setkey(focal_info, focal_id)

  # This is the key join â€” each focal row gets its neighbor IDs
  expanded <- edges[focal_info, on = .(focal_id),
                    allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded now has: focal_id, neighbor_id, year, focal_row

  # Now join to get the neighbor's row index in the same year
  setkey(expanded, neighbor_id, year)
  setkey(lookup, id, year)

  expanded[lookup,
           neighbor_row := i..row_idx,
           on = .(neighbor_id = id, year = year)]

  # Drop rows where neighbor had no data that year
  expanded <- expanded[!is.na(neighbor_row)]

  # --- Step 3: Compute grouped stats for each variable ---
  for (var_name in neighbor_source_vars) {
    # Pull neighbor values via integer indexing (vectorized)
    expanded[, nval := dt[[var_name]][neighbor_row]]

    # Remove NAs before aggregation
    agg <- expanded[!is.na(nval),
                    .(nb_max  = max(nval),
                      nb_min  = min(nval),
                      nb_mean = mean(nval)),
                    by = focal_row]

    # Assign back to dt by row index
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    dt[agg$focal_row, (max_col)  := agg$nb_max]
    dt[agg$focal_row, (min_col)  := agg$nb_min]
    dt[agg$focal_row, (mean_col) := agg$nb_mean]

    expanded[, nval := NULL]  # clean up for next iteration
  }

  # --- Step 4: Clean up and return ---
  dt[, .row_idx := NULL]
  return(as.data.frame(dt))
}

# ---- Usage (drop-in replacement for the original outer loop) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is used as before â€” no changes needed:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Complexity Comparison

| Step | Original | Optimized |
|------|----------|-----------|
| Neighbor lookup construction | O(N Ã— k) string ops in R loop (N=6.46M, kâ‰ˆ4) | O(E) vectorized edge-list build (Eâ‰ˆ1.37M) |
| Key matching (row finding) | O(N Ã— k) named-vector char lookups | O(N Ã— k) `data.table` binary-search join |
| Stats computation | O(N Ã— V) R-level `lapply` calls | O(N Ã— k Ã— V) grouped `data.table` agg in C |
| **Total estimated wall time** | **~86 hours** | **~5â€“15 minutes** |

*(N = 6.46M rows, k â‰ˆ 4 avg neighbors, V = 5 variables, E = 1.37M directed edges)*

## Memory Note

The `expanded` table will have approximately 6.46M Ã— 4 â‰ˆ 25.8M rows with a few integer/numeric columns â€” roughly **1â€“2 GB**, well within the 16 GB laptop constraint. If memory is tight, process one variable at a time and drop `nval` between iterations (as shown above).

## Summary

The original code's bottleneck was millions of R-level string constructions and character-key lookups inside `lapply`. The fix replaces all of this with `data.table` vectorized integer joins and grouped aggregation, preserving identical numerical output and leaving the trained Random Forest model completely untouched.
END_RESPONSE>>>
