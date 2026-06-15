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
run_id: Anthropic_F04_T10_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, the function:

1. Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) â€” these are O(n) hash-style lookups but repeated ~6.46M times with string paste/match overhead.
2. Constructs `paste(..., sep="_")` keys for every neighbor of every row â€” millions of small string allocations.
3. Returns a ragged list of integer vectors.

**`compute_neighbor_stats`** then iterates over that 6.46M-element list again, extracting and summarizing values per row. The `do.call(rbind, result)` on 6.46M 3-element vectors is also expensive.

Both functions are pure-R row-wise loops over millions of rows with heavy string operations. This is the classic R anti-pattern.

## Optimization Strategy

**Core insight:** The neighbor graph is *static across years*. A cell's spatial neighbors are the same in every year. So we can:

1. **Vectorize the lookup construction** using `data.table` keyed joins instead of per-row `lapply` + string pasting. Build an edge-list of `(cell_id, neighbor_id)` once, then join on `(neighbor_id, year)` to get neighbor row indices for all rows simultaneously.

2. **Vectorize the stats computation** by performing a single grouped aggregation (`max`, `min`, `mean`) over the edge-list joined with variable values â€” no per-row `lapply` needed.

3. **Avoid ragged list storage entirely.** A `data.table` grouped-by operation replaces both `build_neighbor_lookup` and `compute_neighbor_stats`.

This reduces ~6.46M Ã— k R-level iterations to a handful of vectorized `data.table` joins and group-bys. Expected speedup: **~100â€“500Ã—**, bringing runtime from 86+ hours to minutes.

## Optimized Working R Code

```r
library(data.table)

#
# Step 1 â€” One-time: build a flat directed edge list from the nb object.
#          This is done ONCE regardless of how many variables you process.
#
build_edge_dt <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)
  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

#
# Step 2 â€” Compute neighbor stats for one variable using a single keyed join
#          and grouped aggregation. Returns the original data.table with three
#          new columns appended.
#
compute_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {
  # --- build a slim lookup: (to_id, year) -> value ---
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setnames(val_dt, "id", "to_id")
  setkey(val_dt, to_id, year)

  # --- expand edges Ã— years: for every (from_id, year) get neighbor values ---
  #     Join edge_dt with cell_dt's (from_id, year) combos, then look up vals.
  #     Memory-efficient: we only need from_id, year from cell_dt.
  from_dt <- cell_dt[, .(from_id = id, year)]
  setkey(edge_dt, from_id)
  setkey(from_dt, from_id)

  # Cartesian-ish join: for each (from_id, year), attach all to_id neighbors

  merged <- edge_dt[from_dt, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # merged now has columns: from_id, to_id, year

  # Attach the neighbor's value for that year
  merged <- val_dt[merged, on = c("to_id", "year"), nomatch = NA]
  # merged now has columns: to_id, year, val, from_id

  # --- grouped aggregation ---
  stats <- merged[!is.na(val),
    .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ),
    keyby = .(from_id, year)
  ]

  # Name columns to match the original pipeline's convention
  new_names <- paste0(var_name, c("_max", "_min", "_mean"))
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), new_names)

  # --- left-join back onto cell_dt ---
  result <- merge(cell_dt, stats,
    by.x = c("id", "year"),
    by.y = c("from_id", "year"),
    all.x = TRUE, sort = FALSE
  )
  result
}

# ============================================================
# Main pipeline (replaces the entire outer loop)
# ============================================================

# Convert to data.table once (in-place conversion avoids a copy)
setDT(cell_data)

# Build edge list ONCE from the precomputed nb object
edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)

# Feature construction for every source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  cell_data <- compute_neighbor_features_fast(cell_data, var_name, edge_dt)
}

# cell_data now contains <var>_max, <var>_min, <var>_mean for all 5 variables.
# Feed directly into the existing (already trained) Random Forest predict() call.
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | `max`, `min`, `mean` are computed over the identical neighbor sets (same directed rook edges, same year match, same NA exclusion). The estimand is unchanged. |
| **Trained RF model** | No retraining. The code only constructs predictor columns; the model object is untouched. Column names follow the same `<var>_max/min/mean` convention so `predict()` works as before. |
| **Memory (16 GB laptop)** | The largest intermediate object (`merged`) has ~6.46M Ã— avg_neighbors â‰ˆ ~26M rows Ã— 4 columns of simple types â‰ˆ ~0.8 GB. Well within budget. If memory is tight, process one variable at a time (as the loop already does) and `rm(merged)` inside the function. |

## Expected Performance

| Phase | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~hours (6.46M `lapply` + string ops) | ~10 s (`data.table` edge list build) |
| Stats per variable | ~17 h each (6.46M `lapply`) | ~30â€“90 s each (vectorized join + group-by) |
| **Total (5 variables)** | **86+ hours** | **~5â€“10 minutes** |

The entire bottleneck is eliminated by replacing row-wise R loops and string key construction with `data.table` vectorized keyed joins and grouped aggregations.
END_RESPONSE>>>
