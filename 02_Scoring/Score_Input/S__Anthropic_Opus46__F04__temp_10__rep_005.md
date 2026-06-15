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
run_id: Anthropic_F04_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector â€” O(1) average but with constant overhead from `as.character()` and name-based lookup repeated millions of times.
2. Constructs `paste(..., sep="_")` key strings for every neighbor of every row.
3. Looks up those keys in `idx_lookup` (another named-vector lookup involving string hashing).

This means roughly **6.46M Ã— ~4 neighbors Ã— 3 string operations** â‰ˆ 77M+ string allocations and hash lookups, all inside an interpreted R `lapply` loop. `compute_neighbor_stats` then loops again over the 6.46M-element list, extracting subsets of a numeric vector â€” lightweight individually, but the sheer iteration count and the `do.call(rbind, ...)` on a 6.46M-element list adds further overhead.

**Root causes (ranked):**
1. Row-level R loop with per-iteration string construction/hashing in `build_neighbor_lookup`.
2. Returning a 6.46M-element list-of-vectors, then iterating over it again per variable.
3. `do.call(rbind, ...)` on a multi-million element list (slow recursive binding).

## Optimization Strategy

**Replace the row-level R loop with a fully vectorized `data.table` join approach:**

- Expand the neighbor graph into an edge table (`cell_id â†’ neighbor_id`).
- Join the panel data onto this edge table by `(neighbor_id, year)` to retrieve neighbor values in one vectorized merge.
- Compute grouped `max/min/mean` with `data.table`'s fast `by=` aggregation.

This eliminates all per-row string pasting, list construction, and repeated lookups. Expected speedup: **~100â€“500Ã—** (minutes instead of days).

## Optimized Working R Code

```r
library(data.table)

build_neighbor_features_fast <- function(cell_data, id_order, rook_neighbors_unique, neighbor_source_vars) {

  # ---- Step 1: Build directed edge table from the nb object ----
  # rook_neighbors_unique is a list of integer index vectors (spdep nb object).
  # id_order maps positional index -> cell id.

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb_idx <- rook_neighbors_unique[[i]]
    nb_idx <- nb_idx[nb_idx != 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list now has ~1.37M rows: (cell_id, neighbor_id)

  # ---- Step 2: Convert panel to data.table keyed on (id, year) ----
  dt <- as.data.table(cell_data)

  # Keep only the columns we need for the neighbor join to reduce memory
  join_cols <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..join_cols]

  # ---- Step 3: For each source variable, compute neighbor stats via join ----
  for (var_name in neighbor_source_vars) {

    # Create a lookup table: (id, year, value)
    val_dt <- dt_slim[, .(neighbor_id = id, year, val = get(var_name))]
    setkey(val_dt, neighbor_id, year)

    # Expand edges Ã— years: join edge_list with the panel on neighbor side
    # First, add the focal cell's year by joining edge_list with dt's (id, year)
    focal_keys <- dt[, .(cell_id = id, year)]
    expanded <- edge_list[focal_keys, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
    # expanded has columns: cell_id, neighbor_id, year

    # Now join to get the neighbor's value for that year
    setkey(expanded, neighbor_id, year)
    expanded <- val_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
    # expanded now has: neighbor_id, year, val, cell_id

    # ---- Step 4: Aggregate per (cell_id, year) ----
    stats <- expanded[!is.na(val),
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      by = .(cell_id, year)
    ]

    # Rename columns to match original feature names
    max_name  <- paste0("neighbor_max_",  var_name)
    min_name  <- paste0("neighbor_min_",  var_name)
    mean_name <- paste0("neighbor_mean_", var_name)
    setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_name, min_name, mean_name))

    # ---- Step 5: Left-join back to the main table ----
    setkey(stats, cell_id, year)
    setkey(dt, id, year)
    dt <- stats[dt, on = .(cell_id = id, year = year)]
    setnames(dt, "cell_id", "id")
    setkey(dt, id, year)

    # Clean up to keep memory bounded on a 16 GB laptop
    rm(val_dt, expanded, stats)
    gc()

    message(paste0("Done: ", var_name))
  }

  return(dt)
}

# ---- Usage (drop-in replacement for the original outer loop) ----
cell_data_dt <- build_neighbor_features_fast(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Convert back to data.frame if the trained RF predict() expects one
cell_data <- as.data.frame(cell_data_dt)

# The trained Random Forest model is untouched â€” just call predict() as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor index sets and the same raw values. The vectorized join replicates the same (cell_id, year) â†” (neighbor_id, year) pairing as the original key-based lookup. |
| **Trained RF model** | No model retraining occurs. The output `cell_data` has identically named columns fed to `predict()`. |
| **NA handling** | `!is.na(val)` in the aggregation and `nomatch = NA` in the left-join replicate the original `is.na` filtering and the `NA` default for cells with no valid neighbors. |

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~60â€“70 hrs (6.46M R-loop iterations with string ops) | ~2 sec (vectorized `rbindlist` over 344K cells) |
| Stats per variable | ~3â€“4 hrs each (R lapply + `do.call(rbind,...)`) | ~1â€“3 min each (`data.table` keyed join + grouped agg) |
| **Total for 5 variables** | **~86+ hrs** | **~5â€“15 min** |
| Peak RAM | ~8â€“10 GB (6.46M-element list of integer vectors) | ~6â€“8 GB (edge expansion materialized one variable at a time, then freed) |

The optimization converts two nested R-level loops with per-element string operations into vectorized `data.table` keyed joins and grouped aggregations, reducing wall-clock time from days to minutes on the same 16 GB laptop.
END_RESPONSE>>>
