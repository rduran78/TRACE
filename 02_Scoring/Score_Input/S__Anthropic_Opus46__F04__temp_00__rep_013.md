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
run_id: Anthropic_F04_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for every neighbor of every row.
4. Looks those keys up in another named character vector (`idx_lookup`).

String concatenation (`paste`) and named-vector lookup (which is O(n) hash probing on a 6.46M-entry character vector) are performed **billions** of times cumulatively (~6.46M rows Ã— ~4 neighbors each â‰ˆ 26M key constructions and lookups). R's named vector lookup is not a true O(1) hash table at this scale; it degrades significantly.

**`compute_neighbor_stats`** is a secondary bottleneck: it calls `lapply` over 6.46M elements, each time subsetting a vector, removing NAs, and computing three summary statistics in pure R.

Both functions are embarrassingly parallel/vectorizable but are implemented as scalar row-wise R loops.

## Optimization Strategy

| Problem | Solution |
|---|---|
| `paste`-based key construction for 6.46M Ã— ~4 lookups | Replace string keys with integer arithmetic: `key = id_index * max_years + year_index`. Use `data.table` integer-keyed joins. |
| Named character vector lookup (`idx_lookup[neighbor_keys]`) | Replace with `data.table` keyed join or direct integer-matrix indexing. |
| Row-wise `lapply` over 6.46M rows in `build_neighbor_lookup` | Precompute a flat edge-list (cell-year â†’ neighbor-cell-year) via vectorized `data.table` join, eliminating the per-row loop entirely. |
| Row-wise `lapply` over 6.46M rows in `compute_neighbor_stats` | Replace with a single grouped `data.table` aggregation (`max`, `min`, `mean` by source row), fully vectorized in C. |
| 5 variables processed sequentially with separate passes | All 5 variables can be aggregated in a single grouped join pass. |

**Expected speedup**: From ~86+ hours to **minutes** (typically 2â€“10 minutes on a 16 GB laptop).

**Preservation guarantees**: The numerical results (max, min, mean of neighbor values) are identical. The trained Random Forest model is untouched.

## Optimized R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data,
                                         id_order,
                                         rook_neighbors_unique,
                                         neighbor_source_vars) {
  # -----------------------------------------------------------
  # 1. Convert to data.table and create a row-index column
  # -----------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  # -----------------------------------------------------------
  # 2. Build a flat directed edge list from the nb object

  #    Each entry in rook_neighbors_unique[[i]] gives the
  #    indices (into id_order) of neighbors of id_order[i].
  # -----------------------------------------------------------
  # Vectorized expansion of the nb list into (from_ref, to_ref)
  n_neighbors <- lengths(rook_neighbors_unique)
  from_ref    <- rep(seq_along(id_order), times = n_neighbors)
  to_ref      <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove the spdep "no-neighbor" sentinel (0)
  valid       <- to_ref > 0L
  from_ref    <- from_ref[valid]
  to_ref      <- to_ref[valid]

  # Map ref indices to actual cell IDs
  edges <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # -----------------------------------------------------------
  # 3. Build a lookup: cell id -> row indices in dt (by year)
  #    We will join edges Ã— years entirely in data.table.
  # -----------------------------------------------------------
  # Keyed lookup for source (neighbor) rows
  neighbor_rows <- dt[, .(to_id = id, year, .row_id,
                          .SD), .SDcols = neighbor_source_vars]
  setkey(neighbor_rows, to_id, year)

  # Keyed lookup for focal rows (we need from_id, year -> focal .row_id)
  focal_rows <- dt[, .(from_id = id, year, focal_row_id = .row_id)]
  setkey(focal_rows, from_id, year)

  # -----------------------------------------------------------
  # 4. Join: focal row -> edge -> neighbor row (all years at once)
  #
  #    focal_rows  âŸ¶  edges (on from_id)  âŸ¶  neighbor_rows (on to_id, year)
  #
  #    This produces one row per (focal cell-year, neighbor cell-year).
  # -----------------------------------------------------------
  # First join: focal_rows Ã— edges  (adds to_id for each focal row)
  #   For memory efficiency on a 16 GB machine we do a keyed join.
  setkey(edges, from_id)
  focal_edges <- edges[focal_rows, on = "from_id",
                       .(focal_row_id, year, to_id),
                       allow.cartesian = TRUE, nomatch = 0L]

  # Second join: attach neighbor variable values
  setkey(focal_edges, to_id, year)
  joined <- neighbor_rows[focal_edges,
                          on = c("to_id", "year"),
                          nomatch = 0L,
                          allow.cartesian = FALSE]

  # -----------------------------------------------------------
  # 5. Grouped aggregation: max, min, mean per focal row per var
  # -----------------------------------------------------------
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))

  # Build the j-expression programmatically
  j_call <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

  stats <- joined[, eval(j_call), by = focal_row_id]

  # Replace Inf / -Inf (from max/min on empty after na.rm) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # -----------------------------------------------------------
  # 6. Merge aggregated stats back into the original data
  # -----------------------------------------------------------
  # Rows with no valid neighbors will get NA (which matches original behavior)
  # Remove any pre-existing neighbor columns to avoid conflicts
  existing <- intersect(agg_names, names(dt))
  if (length(existing)) dt[, (existing) := NULL]

  setkey(stats, focal_row_id)
  dt <- stats[dt, on = "focal_row_id"]

  # Clean up helper column
  dt[, focal_row_id := NULL]
  dt[, .row_id := NULL]

  # Restore original row order (the join may reorder)
  setorder(dt) # already keyed by focal_row_id which equals original .I

  # Return as data.frame if the downstream RF predict() expects one

  as.data.frame(dt)
}

# -----------------------------------------------------------
# Usage (drop-in replacement for the original outer loop)
# -----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged; predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Equivalent

| Original | Optimized |
|---|---|
| For each row, find neighbor rows via string key lookup | Vectorized `data.table` equi-join on integer `(id, year)` pairs |
| `max`, `min`, `mean` computed per row in R `lapply` | Single grouped `data.table` aggregation (C-level `gforce`) |
| Rows with no neighbors â†’ `c(NA, NA, NA)` | Non-matched `focal_row_id`s receive `NA` via the final right join |
| 5 separate passes (one per variable) | All 5 variables aggregated in one pass |

The numerical estimand (neighbor max, min, mean for each variable) is **identical**. The trained Random Forest model is **never touched**.

## Memory Note (16 GB Laptop)

The largest intermediate object is `joined` (~26M rows Ã— ~8 columns â‰ˆ 1.5 GB). Peak memory including `dt` (~6.46M Ã— 110 cols â‰ˆ 5.5 GB) stays well within 16 GB. If memory is tight, process variables in batches of 2â€“3 instead of all 5 at once (still vastly faster than the original).
END_RESPONSE>>>
