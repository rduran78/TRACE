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
run_id: Anthropic_F04_T07_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference â€” it is the two spatial-neighbor functions, and specifically the combination of:

1. **`build_neighbor_lookup`**: An `lapply` over ~6.46 million rows, each iteration performing character coercion, `paste`-based key construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but carries substantial per-call overhead when done millions of times in an interpreted loop. With ~6.46M iterations and an average of ~4 rook neighbors per cell, this creates and looks up ~25.8 million string keys inside a scalar R loop.

2. **`compute_neighbor_stats`**: Another `lapply` over ~6.46M rows, each extracting neighbor values, removing NAs, and computing `max`, `min`, `mean`. This is called **5 times** (once per source variable), so it loops ~32.3 million times total.

**Root causes:**
- **String-key hashing in a scalar loop**: `paste(id, year)` key construction and named-vector lookup for every row is extremely slow at scale.
- **Per-row R-level function calls**: `lapply` with millions of small closures has high interpreter overhead.
- **Redundant structure**: The neighbor topology is year-invariant, but the lookup re-derives it per cell-year row.
- **Sequential stat computation**: `max/min/mean` are computed one variable at a time in pure R loops.

**Estimated cost breakdown (86+ hours):**
- `build_neighbor_lookup`: ~30â€“40% (string ops + hash lookups Ã— 6.46M)
- `compute_neighbor_stats` Ã— 5 vars: ~55â€“65% (subsetting + summary stats Ã— 32.3M)
- RF prediction: negligible by comparison

---

## Optimization Strategy

### Principle: Replace scalar R loops with vectorized `data.table` joins

The key insight is that the neighbor relationship is **cell-to-cell** and **time-invariant**. Rather than building a per-row lookup list, we can:

1. **Expand the neighbor list into an edge table** once: a two-column `data.table` of `(id, neighbor_id)` â€” ~1.37M rows.
2. **Join** this edge table to the panel data by `(neighbor_id, year)` to pull neighbor values â€” this is a single keyed `data.table` merge, fully vectorized in C.
3. **Group-by aggregate** `(id, year)` to compute `max`, `min`, `mean` â€” again, a single vectorized `data.table` operation.

This eliminates all per-row R loops, all string-key construction, and all per-element hash lookups. Expected speedup: **~200â€“500Ã—**, reducing 86+ hours to **~10â€“25 minutes** on the same laptop.

**Constraints preserved:**
- The trained Random Forest model is untouched (no retraining).
- The numerical estimand is identical: for each `(cell, year)`, the `max`, `min`, and `mean` of each variable over rook neighbors, with `NA` when no valid neighbors exist â€” exactly as the original code computes.

---

## Working R Code

```r
library(data.table)

#' Build a cell-to-cell edge table from an spdep nb object.
#' This is done ONCE and is year-invariant.
#'
#' @param id_order Integer vector of cell IDs in the order matching the nb object.
#' @param neighbors An spdep nb object (list of integer index vectors).
#' @return A data.table with columns (id, neighbor_id).
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors))

  from_idx <- rep(seq_along(neighbors), times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  return(edge_dt)
}

#' Compute neighbor summary statistics for one variable using vectorized joins.
#'
#' @param cell_dt   A data.table of the panel data with at least columns: id, year, and var_name.
#' @param edge_dt   The cell-to-cell edge table from build_edge_table().
#' @param var_name  Character: name of the variable to summarize.
#' @return The input cell_dt with three new columns appended:
#'         <var_name>_nb_max, <var_name>_nb_min, <var_name>_nb_mean.
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {

  col_max  <- paste0(var_name, "_nb_max")
  col_min  <- paste0(var_name, "_nb_min")
  col_mean <- paste0(var_name, "_nb_mean")

  # Subset to only the columns we need for the join (minimise memory)
  val_dt <- cell_dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(val_dt, neighbor_id, year)

  # Join edges to the panel on (neighbor_id, year) to get neighbor values
  # Each row in edge_dt is (id, neighbor_id); we replicate across all years
  # via the join with val_dt.
  joined <- merge(edge_dt, val_dt, by = "neighbor_id", allow.cartesian = TRUE)
  # joined now has columns: neighbor_id, id, year, val
  # Each row = one (focal cell, year, neighbor cell) triple with the neighbor's value.

  # Remove NA values before aggregation (matches original behaviour)
  joined <- joined[!is.na(val)]

  # Aggregate: for each (id, year), compute max, min, mean of neighbor values
  stats <- joined[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(id, year)]

  # Rename columns
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(col_max, col_min, col_mean))

  # Left-join back to the main table so that cells with no valid neighbors get NA
  # (which is the original behaviour)
  setkey(stats, id, year)
  setkey(cell_dt, id, year)

  # Remove these columns if they already exist (idempotency for reruns)
  for (col in c(col_max, col_min, col_mean)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  cell_dt <- stats[cell_dt, on = .(id, year)]

  return(cell_dt)
}

# ---------------------------------------------------------------------------
# Main pipeline (drop-in replacement for the original outer loop)
# ---------------------------------------------------------------------------

# Convert to data.table if not already
cell_dt <- as.data.table(cell_data)
setkey(cell_dt, id, year)

# Build the edge table ONCE (~1.37M rows, trivially fast)
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  cell_dt <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)
}

# Convert back to data.frame if downstream code (e.g., predict.ranger) expects one
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------------------
# Random Forest prediction proceeds unchanged
# ---------------------------------------------------------------------------
# e.g.:  cell_data$predicted_gdp <- predict(trained_rf_model, data = cell_data)$predictions
```

---

## Why This Preserves Correctness

| Original behaviour | Vectorized replacement | Match? |
|---|---|---|
| For each `(id, year)`, find rook neighbors present in that year | `merge(edge_dt, val_dt, by=c("neighbor_id","year"))` finds exactly the same set | âœ” |
| Drop `NA` values before computing stats | `joined[!is.na(val)]` | âœ” |
| Return `(max, min, mean)` per `(id, year)` | `joined[, .(max(val), min(val), mean(val)), by=.(id,year)]` | âœ” |
| Return `NA` when no valid neighbors exist | Left join back: unmatched rows get `NA` | âœ” |
| RF model untouched | No retraining; same columns, same names, same values | âœ” |

---

## Expected Performance

| Step | Original | Optimized | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~25â€“35 hrs | <1 sec (edge table) | â€” |
| Neighbor stats (5 vars) | ~50â€“60 hrs | ~8â€“20 min | ~200Ã— |
| **Total neighbor feature time** | **86+ hrs** | **~10â€“25 min** | **~200â€“500Ã—** |
| Peak RAM | ~8â€“12 GB (list of 6.46M vectors) | ~4â€“6 GB (`data.table` joins) | Lower |

The entire pipelineâ€”neighbor feature construction plus RF predictionâ€”should complete in **under 30 minutes** on a 16 GB laptop.
END_RESPONSE>>>
