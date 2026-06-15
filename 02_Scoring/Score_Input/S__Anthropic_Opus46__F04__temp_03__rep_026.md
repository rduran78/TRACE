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
run_id: Anthropic_F04_T03_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) â€” these are hash-table lookups but done **6.46 million times in an interpreted R loop**.
2. Constructs paste-based string keys (`paste(neighbor_cell_ids, year, sep="_")`) for every neighbor of every row â€” creating and discarding millions of temporary character vectors.
3. Returns variable-length integer vectors, assembled into a list of 6.46M elements.

**`compute_neighbor_stats`** then iterates over that 6.46M-element list again, once per variable (Ã—5 variables = ~32.3M interpreted iterations).

The fundamental problem: **row-level interpreted R loops over millions of rows with per-element string operations and named-vector lookups**. The algorithmic complexity is fine (linear in edges Ã— years), but the constant factor in interpreted R is enormous.

## Optimization Strategy

**Replace the row-level R loop with vectorized operations using `data.table`.**

Key ideas:

1. **Expand the neighbor list into an edge table once** â€” a two-column `data.table` of `(cell_id, neighbor_id)` with ~1.37M rows.
2. **Join on `(neighbor_id, year)`** to get neighbor row indices or values directly â€” this is a single keyed merge, fully vectorized in C via `data.table`.
3. **Group-by aggregation** `[, .(max, min, mean), by = .(cell_id, year)]` replaces the per-row `lapply` in `compute_neighbor_stats`.
4. Do this once per variable (5 joins + group-bys instead of 32.3M R-level iterations).

This eliminates all interpreted loops, all string-key construction, and all per-row temporary allocations. Expected runtime: **minutes, not days**.

## Working R Code

```r
library(data.table)

#' Build a data.table edge list from an nb object.
#' Returns a two-column data.table: (id, neighbor_id)
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep nb object)
  n <- length(neighbors)
  # Pre-allocate: total number of directed edges
  from_idx <- rep.int(seq_len(n), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

#' Compute neighbor summary statistics for one variable,
#' returning a data.table with columns: id, year, <var>_max, <var>_min, <var>_mean
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {
  # Build a small lookup: (neighbor_id, year) -> value
  lookup <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(lookup, neighbor_id, year)

  # Join edges with all years: expand edges Ã— years
  # Instead of a full cross join (expensive in memory), join through cell_dt's (id, year) pairs
  # Step 1: get all (id, year) pairs
  id_year <- cell_dt[, .(id, year)]
  setkey(id_year, id)
  setkey(edge_dt, id)

  # Step 2: merge to get (id, year, neighbor_id) â€” one row per neighbor per cell-year

  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, neighbor_id, year

  # Step 3: look up the neighbor's value in that year
  setkey(expanded, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # expanded now has: neighbor_id, year, val, id

  # Step 4: aggregate
  max_name  <- paste0(var_name, "_max")
  min_name  <- paste0(var_name, "_min")
  mean_name <- paste0(var_name, "_mean")

  stats <- expanded[
    !is.na(val),
    .(V_max = max(val), V_min = min(val), V_mean = mean(val)),
    by = .(id, year)
  ]
  setnames(stats, c("V_max", "V_min", "V_mean"), c(max_name, min_name, mean_name))
  stats
}

#' Main driver: add neighbor features for all source variables to cell_data.
add_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                      neighbor_source_vars) {
  cell_dt <- as.data.table(cell_data)

  # 1. Build edge table once (~1.37M rows, tiny)
  edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

  # 2. For each variable, compute stats and merge back

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)
    # Left-join stats back onto the main table; unmatched rows get NA (preserves original behavior)
    cell_dt <- stats[cell_dt, on = .(id, year)]
  }

  # 3. Return as data.frame to preserve downstream compatibility
  as.data.frame(cell_dt)
}

# ---- Usage (drop-in replacement for the original outer loop) ----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched; predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets and values. NA handling is identical (NAs excluded before aggregation; rows with no valid neighbors get NA via the left join). |
| **Trained RF model** | No model retraining. The output columns have the same names and semantics, so `predict(rf_model, newdata = cell_data)` works unchanged. |
| **Column naming** | `paste0(var_name, "_max")` etc. matches the original `compute_and_add_neighbor_features` convention. |

## Expected Performance

| Step | Original | Optimized |
|---|---|---|
| Build lookup / edge table | ~6.46M R-loop iterations with string ops | One `rep.int` + `unlist` (~1.37M rows, <1 sec) |
| Neighbor stats (per var) | ~6.46M `lapply` iterations | One keyed join + one `group-by` in C (est. 30â€“90 sec per variable) |
| **Total (5 vars)** | **~86+ hours** | **~5â€“10 minutes** |

Memory peak for the expanded join table: ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows Ã— 3 columns â‰ˆ ~0.9 GB, well within 16 GB RAM.
END_RESPONSE>>>
