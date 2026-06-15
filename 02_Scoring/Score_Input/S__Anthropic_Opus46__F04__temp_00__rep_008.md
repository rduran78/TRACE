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
run_id: Anthropic_F04_T00_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_008.md

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
| 2 | Per-row `lapply` in `compute_neighbor_stats` (6.46M iterations Ã— 5 variables) | Significant |
| 3 | Repeated allocation of small vectors inside closures | Moderate (GC pressure) |

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed join via `data.table`.** Build a `data.table` keyed on `(id, year)` with an integer row-index column. Expand the neighbor graph into an edge-list `data.table` and do a single keyed merge to resolve all neighbor row indices at once â€” no per-row loop, no `paste`.

2. **Vectorize `compute_neighbor_stats`** by joining the edge-list to the data column and using `data.table` grouped aggregation (`[, .(max, min, mean), by = row_idx]`) â€” replaces 6.46M R-level iterations with a single C-level grouped operation.

3. **Process all 5 variables in one pass** over the edge-list join rather than 5 separate `lapply` calls.

These changes reduce algorithmic complexity from O(NÃ—k) R-level string operations to O(NÃ—k) C-level integer operations (via `data.table`), yielding an estimated **~200â€“500Ã— speedup** (minutes instead of days).

## Optimized R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build the neighbor edge-list (one-time, replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edgelist <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors into id_order
  # Expand to a data.table of (focal_id, neighbor_id)
  n_neighbors <- vapply(neighbors, length, integer(1))
  focal_idx   <- rep(seq_along(id_order), times = n_neighbors)
  neigh_idx   <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neigh_idx]
  )
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Compute and attach all neighbor features at once
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      neighbors,
                                      source_vars) {
  # Convert to data.table if needed (by reference if already one)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Assign a row index for later re-attachment
  cell_data[, .row_idx := .I]

  # --- Step 1: edge-list of cell-id pairs (spatial, time-invariant) ---
  edges <- build_neighbor_edgelist(id_order, neighbors)

  # --- Step 2: cross with years via keyed join ---
  # Create a slim lookup: (id, year) -> .row_idx  (and source var values)
  keep_cols <- c("id", "year", ".row_idx", source_vars)
  lookup    <- cell_data[, ..keep_cols]
  setkey(lookup, id, year)

  # Focal side: get (focal_id, year, focal_row_idx) for every cell-year
  focal <- cell_data[, .(focal_id = id, year, focal_row = .row_idx)]

  # Merge focal rows with edge-list to get neighbor_id for each focal row
  # This is the key step: one merge replaces 6.46M paste+lookup iterations
  setkey(edges, focal_id)
  setkey(focal, focal_id)
  expanded <- edges[focal, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # expanded now has columns: focal_id, neighbor_id, year, focal_row

  # Merge in neighbor data values by (neighbor_id, year)
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  merged <- lookup[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # merged has: neighbor_id, year, .row_idx (neighbor's), source_vars,
  #             focal_id, focal_row

  # --- Step 3: grouped aggregation (vectorised, C-level) ---
  agg_exprs <- list()
  for (v in source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("neighbor_max_",  v)]] <-
      bquote(as.numeric(max(.(sym_v),   na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_min_",  v)]] <-
      bquote(as.numeric(min(.(sym_v),   na.rm = TRUE)))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(sym_v), na.rm = TRUE))
  }

  # Evaluate all aggregations in one grouped pass
  stats <- merged[,
    eval(as.call(c(as.name("list"), agg_exprs))),
    by = focal_row
  ]

  # Replace Inf/-Inf (from max/min on all-NA) with NA
  inf_to_na <- function(x) { x[is.infinite(x)] <- NA_real_; x }
  stat_cols <- setdiff(names(stats), "focal_row")
  stats[, (stat_cols) := lapply(.SD, inf_to_na), .SDcols = stat_cols]

  # --- Step 4: attach back to cell_data by row index ---
  setkey(stats, focal_row)
  cell_data[stats, on = c(".row_idx" = "focal_row"),
            (stat_cols) := mget(stat_cols)]

  # Rows with no neighbors (e.g., islands) already have NA from nomatch

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  return(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Usage (drop-in replacement for the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The resulting cell_data now contains columns like:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... (15 new columns total, identical numerical values to original)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Predict with the EXISTING trained Random Forest (unchanged)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---------|-----------|
| **Numerical estimand** | `max`, `min`, `mean` with `na.rm = TRUE` are identical operations to the original; the join logic resolves the same (id, year) pairs. |
| **Trained RF model** | No retraining â€” the code only constructs the same 15 predictor columns the model expects, then calls `predict()`. |
| **Column names** | Naming convention (`neighbor_max_*`, etc.) should match whatever the original `compute_and_add_neighbor_features` produced; adjust the prefix if the original used a different convention. |

## Expected Performance

| Stage | Original | Optimized | Speedup |
|-------|----------|-----------|---------|
| Neighbor lookup construction | ~70+ hours (6.46M `paste` + char lookup) | ~10 seconds (one `data.table` merge) | ~25,000Ã— |
| Neighbor stats (5 vars) | ~16 hours (5 Ã— 6.46M `lapply`) | ~30â€“60 seconds (one grouped aggregation) | ~1,000Ã— |
| **Total neighbor features** | **~86 hours** | **~1â€“2 minutes** | **~3,000Ã—** |

Peak RAM for the expanded edge-list: ~1.37M edges Ã— 6.46M/344K years â‰ˆ 38.4M rows Ã— ~8 numeric columns â‰ˆ **2.5 GB**, well within 16 GB.
END_RESPONSE>>>
