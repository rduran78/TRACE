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
run_id: Anthropic_F11_T10_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly shows the bottleneck is in the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, performing per-row string pasting (`paste()`), character-to-index hash lookups, and NA filtering. That's 6.46 million R-level iterations with string allocation and named-vector lookups â€” an O(n Ã— k) operation where k is the average neighbor count, all in interpreted R with no vectorization.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over the same 6.46 million rows via `lapply`, subsetting vectors, removing NAs, and computing `max/min/mean`. That's another ~32.3 million R-level loop iterations total, each with allocation overhead.

3. **The `do.call(rbind, result)` call** inside `compute_neighbor_stats` binds 6.46 million 3-element vectors into a matrix â€” a notoriously slow pattern in R.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single call to `predict()` on a matrix of ~6.46M Ã— 110 features. Packages like `ranger` or `randomForest` execute this in optimized C/C++ and typically complete in seconds to a few minutes, even on millions of rows. Loading a serialized model (`readRDS`) is also fast. Writing predictions is a single vectorized write. There is no loop, no string manipulation, no per-row R overhead.

**Conclusion:** The bottleneck is the neighbor feature engineering: building the lookup (one-time, ~6.46M R iterations with string ops) and computing neighbor stats (5 Ã— 6.46M R iterations). The estimated 86+ hour runtime is attributable to these operations, not to RF inference.

---

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace per-row `paste` and named-vector lookup with a single vectorized merge/join using `data.table`. Pre-build an edge list of (row_i, row_j) pairs by joining neighbor relationships against a cell-year index table. This eliminates all per-row string operations.

2. **Vectorize `compute_neighbor_stats()`**: Once we have the edge list, group-by-row aggregation (`max`, `min`, `mean`) can be done in a single `data.table` operation per variable â€” fully vectorized in C, no R-level `lapply`.

3. **Eliminate `do.call(rbind, ...)`**: The `data.table` approach produces columnar results directly.

4. **Leave the Random Forest model and prediction code untouched**: The trained model is preserved, and the numerical estimand (the same neighbor features fed to the same model producing the same predictions) is unchanged.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Build a vectorized edge list (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: spdep nb object (list of integer neighbor indices)

  # Step A: Build directed edge list at the cell level: (focal_id, neighbor_id)
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  neighbor_idx <- unlist(rook_neighbors_unique)

  cell_edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # Step B: Add row_index to cell_data_dt
  cell_data_dt[, row_idx := .I]

  # Step C: Create a keyed lookup: (id, year) -> row_idx
  lookup <- cell_data_dt[, .(id, year, row_idx)]

  # Step D: Expand cell-level edges to cell-year-level edges
  #   For every (focal_id, neighbor_id) pair and every year,
  #   map both to their row indices.
  #   Because every cell appears in every year (balanced panel), we cross-join
  #   cell_edges with the unique years, then join twice to get row indices.

  years <- sort(unique(cell_data_dt$year))

  # Cross join edges Ã— years
  cell_year_edges <- cell_edges[, CJ(year = years), by = .(focal_id, neighbor_id)]

  # Join focal row index
  setkey(lookup, id, year)
  setkey(cell_year_edges, focal_id, year)
  cell_year_edges[lookup, focal_row := i.row_idx, on = .(focal_id = id, year = year)]

  # Join neighbor row index
  setkey(cell_year_edges, neighbor_id, year)
  cell_year_edges[lookup, neighbor_row := i.row_idx, on = .(neighbor_id = id, year = year)]

  # Remove any edges where either side is missing
  cell_year_edges <- cell_year_edges[!is.na(focal_row) & !is.na(neighbor_row)]

  # Return only the row-index edges (compact)
  cell_year_edges[, .(focal_row, neighbor_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Vectorized neighbor stats (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_vec <- function(cell_data_dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row, neighbor_row
  # Pull the variable values for neighbor rows
  edge_dt[, val := cell_data_dt[[var_name]][neighbor_row]]

  # Remove NAs in val
  valid <- edge_dt[!is.na(val)]

  # Aggregate per focal row
  agg <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Build full result aligned to all rows
  n <- nrow(cell_data_dt)
  result_max  <- rep(NA_real_, n)
  result_min  <- rep(NA_real_, n)
  result_mean <- rep(NA_real_, n)

  result_max[agg$focal_row]  <- agg$nb_max
  result_min[agg$focal_row]  <- agg$nb_min
  result_mean[agg$focal_row] <- agg$nb_mean

  # Clean up temporary column
  edge_dt[, val := NULL]

  list(nb_max = result_max, nb_min = result_min, nb_mean = result_mean)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Full pipeline (replaces the outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
run_neighbor_feature_pipeline <- function(cell_data, id_order, rook_neighbors_unique) {

  cell_data_dt <- as.data.table(cell_data)

  message("Building vectorized edge list...")
  edge_dt <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list has %s rows.", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_vec(cell_data_dt, edge_dt, var_name)

    col_max  <- paste0(var_name, "_nb_max")
    col_min  <- paste0(var_name, "_nb_min")
    col_mean <- paste0(var_name, "_nb_mean")

    set(cell_data_dt, j = col_max,  value = stats$nb_max)
    set(cell_data_dt, j = col_min,  value = stats$nb_min)
    set(cell_data_dt, j = col_mean, value = stats$nb_mean)
  }

  # Remove helper column if present
  if ("row_idx" %in% names(cell_data_dt)) {
    cell_data_dt[, row_idx := NULL]
  }

  message("Neighbor features complete.")
  cell_data_dt
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. Usage (drop-in replacement for the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# cell_data <- run_neighbor_feature_pipeline(cell_data, id_order, rook_neighbors_unique)
#
# # Then proceed to RF prediction exactly as before â€” no changes needed:
# model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(model, data = cell_data)$predictions
# # (or however predict is called for your specific RF package)
```

---

## Summary of Expected Improvement

| Component | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-level `lapply` iterations with `paste()` + named vector lookup | Single vectorized `data.table` join (~seconds) |
| `compute_neighbor_stats` (Ã—5 vars) | ~32.3M R-level iterations, each with subsetting + `max/min/mean` | 5 vectorized `data.table` group-by aggregations (~seconds each) |
| `do.call(rbind, ...)` (Ã—5) | Binding 6.46M 3-element vectors per call | Eliminated entirely |
| **Estimated total** | **86+ hours** | **Minutes** (dominated by the edge-list cross-join memory allocation) |
| RF inference | Fast (unchanged) | Fast (unchanged) |

The trained Random Forest model is never modified. The same 15 neighbor-derived features (5 variables Ã— {max, min, mean}) are computed with identical numerical values, preserving the original estimand.
END_RESPONSE>>>
