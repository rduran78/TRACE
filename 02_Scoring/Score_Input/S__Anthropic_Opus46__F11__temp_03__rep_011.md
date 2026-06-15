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
run_id: Anthropic_F11_T03_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential R loop.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million R-level loop iterations** total.

3. By contrast, Random Forest **prediction** on a pre-trained model with ~110 predictors and ~6.46M rows is a single vectorized call to `predict()` backed by compiled C/C++ code. It is fast â€” typically minutes, not hours.

**The 86+ hour runtime is dominated by the ~38.7 million R-interpreter-level loop iterations in the neighbor feature engineering, not by the RF predict call.**

## Optimization Strategy

1. **Replace the row-level `lapply` in `build_neighbor_lookup`** with a vectorized, `data.table`-based merge/join approach. Instead of building a per-row list, construct an edge-list data.table of `(row_index, neighbor_row_index)` pairs using keyed joins â€” eliminating millions of `paste` + named-vector lookups.

2. **Replace the row-level `lapply` in `compute_neighbor_stats`** with a grouped `data.table` aggregation over the edge list. Compute `max`, `min`, and `mean` per row in one vectorized, C-backed pass per variable.

3. **Leave the Random Forest predict step untouched** â€” it is already efficient.

This converts O(N Ã— k) interpreted R operations into a small number of vectorized `data.table` joins and group-by aggregations, reducing runtime from 86+ hours to likely **minutes**.

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a vectorized edge list (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edgelist_dt <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Map each cell ID to its position in id_order
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )

  # Build directed edge list at the cell level: (focal_id, neighbor_id)
  # Each element neighbors[[j]] is an integer vector of indices into id_order
  edge_cell <- rbindlist(lapply(seq_along(neighbors), function(j) {
    nb <- neighbors[[j]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[j], neighbor_id = id_order[nb])
  }))

  # Add a row-index column to the data
  data_dt[, row_idx := .I]

  # Key the data for fast joins
  setkey(data_dt, id, year)

  # Expand edges across all years by joining focal side
  # For every (focal_id, neighbor_id) pair, we need every year that the
  # focal cell appears in the data, then look up the neighbor's row in
  # that same year.

  # Focal rows: join edge_cell to data on focal_id == id
  focal_join <- data_dt[, .(focal_row_idx = row_idx, id, year)]
  setnames(focal_join, "id", "focal_id")
  setkey(focal_join, focal_id)
  setkey(edge_cell, focal_id)

  # Merge: gives (focal_id, neighbor_id, year, focal_row_idx)
  edges_with_year <- edge_cell[focal_join, on = "focal_id",
                               allow.cartesian = TRUE, nomatch = 0L]

  # Now look up the neighbor's row index for the same year
  neighbor_key <- data_dt[, .(neighbor_row_idx = row_idx, id, year)]
  setnames(neighbor_key, "id", "neighbor_id")
  setkey(neighbor_key, neighbor_id, year)
  setkey(edges_with_year, neighbor_id, year)

  edgelist <- neighbor_key[edges_with_year, on = c("neighbor_id", "year"),
                           nomatch = NA]

  # Keep only the columns we need
  edgelist <- edgelist[, .(focal_row_idx, neighbor_row_idx)]

  # Remove edges where the neighbor row was not found (boundary / missing year)
  edgelist <- edgelist[!is.na(neighbor_row_idx)]

  return(edgelist)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_neighbor_stats_dt <- function(data_dt, edgelist, var_name) {
  # edgelist: data.table with columns focal_row_idx, neighbor_row_idx
  # Attach the neighbor's value
  el <- copy(edgelist)
  el[, val := data_dt[[var_name]][neighbor_row_idx]]

  # Drop NAs in the variable
  el <- el[!is.na(val)]

  # Grouped aggregation
  stats <- el[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row_idx]

  # Prepare output columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[stats$focal_row_idx]  <- stats$nb_max
  out_min[stats$focal_row_idx]  <- stats$nb_min
  out_mean[stats$focal_row_idx] <- stats$nb_mean

  # Name columns to match original pipeline conventions
  suffix_max  <- paste0("nb_max_",  var_name)
  suffix_min  <- paste0("nb_min_",  var_name)
  suffix_mean <- paste0("nb_mean_", var_name)

  data_dt[, (suffix_max)  := out_max]
  data_dt[, (suffix_min)  := out_min]
  data_dt[, (suffix_mean) := out_mean]

  invisible(data_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Full optimized pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model) {
  # Convert to data.table (no copy if already data.table)
  cell_dt <- as.data.table(cell_data)

  # --- Neighbor feature engineering (the ACTUAL bottleneck, now fast) ---
  message("Building vectorized edge list...")
  edgelist <- build_neighbor_edgelist_dt(cell_dt, id_order, rook_neighbors_unique)

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    compute_neighbor_stats_dt(cell_dt, edgelist, var_name)
  }

  # Clean up the temporary row index
  cell_dt[, row_idx := NULL]

  # --- Random Forest inference (already fast, preserved as-is) ---
  message("Running Random Forest predictions...")
  predictor_cols <- setdiff(names(cell_dt),
                            c("id", "year", "outcome"))
  predictions <- predict(rf_model, newdata = cell_dt[, ..predictor_cols])

  cell_dt[, prediction := predictions]

  message("Done.")
  return(cell_dt)
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` iterations with `paste` + named-vector hash lookups | Single `data.table` keyed join producing an edge list |
| `compute_neighbor_stats` | ~6.46M Ã— 5 = ~32.3M `lapply` iterations with per-row subsetting | 5 grouped `data.table` aggregations (C-backed) |
| RF `predict()` | Vectorized C/C++ call | **Unchanged** â€” not the bottleneck |
| Estimated runtime | 86+ hours | Minutes (edge list build) + minutes (aggregations) + minutes (predict) |
| Numerical results | Original estimand | **Preserved** â€” identical `max`, `min`, `mean` logic |
| Trained RF model | â€” | **Preserved** â€” no retraining |
END_RESPONSE>>>
