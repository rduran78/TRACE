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
run_id: Anthropic_F04_T00_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_016.md

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
| 1 | Per-row `paste()` + character key lookup in `build_neighbor_lookup` (~6.46M iterations) | **Dominant** |
| 2 | Per-row anonymous function + small vector allocations in `compute_neighbor_stats` | **Major** |
| 3 | `do.call(rbind, result)` on a 6.46M-element list of 3-vectors | Moderate |
| 4 | Everything is single-threaded base R | Multiplier |

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed join.** Sort/group data by `(id, year)` and use `data.table` fast binary joins to map each row to its neighbor rows via integer indices â€” no `paste`, no named vectors.

2. **Vectorize `compute_neighbor_stats` entirely.** Expand the neighbor lookup into a long `data.table` of `(row_i, neighbor_row_j)`, join the variable values, and compute grouped `max/min/mean` with `data.table`'s optimized `by=` grouping. This replaces 6.46M R-level function calls with a single vectorized grouped aggregation.

3. **Process all 5 variables in one pass** over the long neighbor table instead of 5 separate `lapply` loops.

Expected speedup: from ~86+ hours to **minutes** (typically 2â€“10 minutes on 16 GB RAM).

## Optimized Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique,
                                         neighbor_source_vars) {
  # Convert to data.table if not already; preserve original row order
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]


  # -------------------------------------------------------------------
  # Step 1: Build a complete (row_i -> row_j) neighbor edge table

  #         using integer joins â€” no paste, no character lookups.
  # -------------------------------------------------------------------

  # Map each cell id to its integer position in id_order
  id_map <- data.table(id = id_order, ref_idx = seq_along(id_order))

  # Expand the nb object into a long edge list: (ref_idx_from, ref_idx_to)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(ref_from = i, ref_to = nb)
  }))

  # Translate ref indices back to cell ids
  edge_list[, id_from := id_order[ref_from]]
  edge_list[, id_to   := id_order[ref_to]]
  edge_list[, c("ref_from", "ref_to") := NULL]

  # For every row in dt, find its neighbor rows by joining on (id, year).
  # First, create a keyed lookup: for each (id, year) -> .row_id
  row_lookup <- dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)

  # Build the "from" side: each row's id and year
  from_side <- dt[, .(id_from = id, year, row_i = .row_id)]

  # Join from_side to edge_list to get neighbor cell ids
  setkey(edge_list, id_from)
  setkey(from_side, id_from)
  # This is a many-to-many join: each row_i Ã— its neighbor id_to values
  edges_with_year <- edge_list[from_side, on = "id_from", allow.cartesian = TRUE,
                               nomatch = NULL]
  # edges_with_year now has columns: id_from, id_to, year, row_i

  # Join to row_lookup to get row_j (the neighbor's row index in the same year)
  setnames(edges_with_year, "id_to", "id")
  setkey(edges_with_year, id, year)
  setkey(row_lookup, id, year)
  neighbor_edges <- row_lookup[edges_with_year, on = c("id", "year"), nomatch = NULL]
  # neighbor_edges has: id, year, .row_id (= row_j), id_from, row_i
  setnames(neighbor_edges, ".row_id", "row_j")
  neighbor_edges <- neighbor_edges[, .(row_i, row_j)]

  # -------------------------------------------------------------------
  # Step 2: Vectorized grouped aggregation for all source variables
  # -------------------------------------------------------------------

  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]
    neighbor_edges[, v := vals[row_j]]

    # Remove NAs before aggregation
    valid <- neighbor_edges[!is.na(v)]

    agg <- valid[, .(
      nb_max  = max(v),
      nb_min  = min(v),
      nb_mean = mean(v)
    ), by = row_i]

    # Build full-length columns (NA for rows with no valid neighbors)
    max_col  <- rep(NA_real_, nrow(dt))
    min_col  <- rep(NA_real_, nrow(dt))
    mean_col <- rep(NA_real_, nrow(dt))

    max_col[agg$row_i]  <- agg$nb_max
    min_col[agg$row_i]  <- agg$nb_min
    mean_col[agg$row_i] <- agg$nb_mean

    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)

    neighbor_edges[, v := NULL]
  }

  # -------------------------------------------------------------------
  # Step 3: Clean up and return
  # -------------------------------------------------------------------
  dt[, .row_id := NULL]

  # Return as data.frame if the input was a data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
# Original code replaced by a single call:
cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is untouched â€” just predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

### Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets with the same NA-removal logic. The output columns are numerically identical. |
| **Trained RF model** | No model retraining. The feature column names (`*_neighbor_max/min/mean`) match the originals. Only the feature *construction* path changes. |
| **Column naming** | Adjust the `paste0` suffix pattern to match whatever `compute_and_add_neighbor_features` originally produced (e.g., `"_nb_max"` vs `"_neighbor_max"`). |

### Memory Note (16 GB laptop)

The `neighbor_edges` table will have approximately 6.46M rows Ã— 4 neighbors â‰ˆ 25.8M rows Ã— 2 integer columns â‰ˆ **0.4 GB**. Adding one double column `v` at a time brings peak usage to roughly **2â€“3 GB** for the neighbor computation on top of the existing data, well within 16 GB. If memory is tight, process one variable at a time (as the loop already does) and call `gc()` between iterations.
END_RESPONSE>>>
