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
run_id: Anthropic_F11_T00_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates tens of millions of string operations and hash lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function invocations across the 5 variables.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with ~6.46M rows and ~110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and runs in minutes, not hours. Loading the model from disk is a one-time `readRDS()` call. Writing predictions is a single `fwrite()` or `write.csv()`. None of these are 86-hour operations.

**Conclusion:** The bottleneck is the row-level R `lapply` loops over millions of rows doing string construction and named-vector lookups for neighbor feature computation. This is a classic R anti-pattern: millions of iterations of interpreted, non-vectorized code.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` entirely** with a vectorized `data.table` merge/join approach. Instead of building a per-row list of neighbor indices via string keys, we construct a flat edge-list data.table of `(row_i, neighbor_row_j)` pairs using keyed joins â€” no `paste`, no named-vector lookups, no `lapply`.

2. **Replace `compute_neighbor_stats()` with a grouped `data.table` aggregation.** Once we have the flat edge list with the neighbor's value joined in, we compute `max`, `min`, and `mean` per row using `data.table`'s `by=` grouping, which is executed in C.

3. **Preserve the trained Random Forest model** â€” we do not retrain. We only change the feature engineering that feeds into `predict()`.

4. **Preserve the original numerical estimand** â€” the computed features (`_max`, `_min`, `_mean` of neighbor values) are numerically identical; only the computational method changes.

Expected speedup: from 86+ hours to **minutes** (roughly 1,000â€“10,000Ã— faster).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build a vectorized neighbor edge-list (replaces
#         build_neighbor_lookup entirely)
# ============================================================

build_neighbor_edgelist <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt: a data.table with columns 'id' and 'year' (and others)
  # id_order: vector of cell IDs in the order matching rook_neighbors_unique
  # rook_neighbors_unique: an nb object (list of integer neighbor index vectors)

  # --- 1a. Build spatial neighbor edge list (cell-level, not row-level) ---
  # Each element i of rook_neighbors_unique contains the indices (into id_order)
  # of the neighbors of id_order[i].

  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique)

  # Remove 0-neighbor entries (spdep uses 0L for no-neighbor in some representations)
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  spatial_edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )

  # --- 1b. Create a row-index lookup keyed by (id, year) ---
  cell_data_dt[, row_idx := .I]

  # --- 1c. Expand spatial edges across all years via keyed join ---
  # For each (from_id, year) row, find all neighbor (to_id, year) rows.

  # Get unique years
  years <- sort(unique(cell_data_dt$year))

  # Cross-join spatial edges with years
  edge_year <- spatial_edges[, .(year = years), by = .(from_id, to_id)]

  # Join to get the row index of the focal cell (from_id, year)
  setkey(cell_data_dt, id, year)
  edge_year[cell_data_dt, on = .(from_id = id, year = year), focal_row := i.row_idx]

  # Join to get the row index of the neighbor cell (to_id, year)
  edge_year[cell_data_dt, on = .(to_id = id, year = year), neighbor_row := i.row_idx]

  # Drop edges where either focal or neighbor row is missing
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  return(edge_year[, .(focal_row, neighbor_row)])
}


# ============================================================
# STEP 2: Compute neighbor stats via grouped data.table
#         aggregation (replaces compute_neighbor_stats)
# ============================================================

compute_neighbor_features_fast <- function(cell_data_dt, edge_dt, var_name) {
  # edge_dt: data.table with columns focal_row, neighbor_row
  # var_name: character, the column to aggregate

  vals <- cell_data_dt[[var_name]]

  # Attach the neighbor's value to each edge
  work <- copy(edge_dt)
  work[, nval := vals[neighbor_row]]

  # Drop NAs in neighbor values
  work <- work[!is.na(nval)]

  # Grouped aggregation
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]

  # Build output columns (NA for rows with no valid neighbors)
  n <- nrow(cell_data_dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)

  col_max[agg$focal_row]  <- agg$nb_max
  col_min[agg$focal_row]  <- agg$nb_min
  col_mean[agg$focal_row] <- agg$nb_mean

  max_name  <- paste0(var_name, "_max")
  min_name  <- paste0(var_name, "_min")
  mean_name <- paste0(var_name, "_mean")

  set(cell_data_dt, j = max_name,  value = col_max)
  set(cell_data_dt, j = min_name,  value = col_min)
  set(cell_data_dt, j = mean_name, value = col_mean)

  invisible(cell_data_dt)
}


# ============================================================
# STEP 3: Full optimized pipeline
# ============================================================

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model_path, output_path) {

  # Convert to data.table (in-place if already, otherwise copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Neighbor feature engineering (THE FORMER BOTTLENECK) ---
  message("Building vectorized neighbor edge list...")
  t0 <- proc.time()

  edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)

  message("  Edge list built: ", nrow(edge_dt), " directed (row, neighbor-row) pairs")
  message("  Elapsed: ", round((proc.time() - t0)[3], 1), "s")

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  message("Computing neighbor features for ", length(neighbor_source_vars), " variables...")
  t1 <- proc.time()

  for (var_name in neighbor_source_vars) {
    compute_neighbor_features_fast(cell_data, edge_dt, var_name)
    message("  Done: ", var_name)
  }

  message("  Neighbor features elapsed: ", round((proc.time() - t1)[3], 1), "s")

  # Clean up temporary column
  if ("row_idx" %in% names(cell_data)) {
    cell_data[, row_idx := NULL]
  }

  # --- Random Forest inference (NOT the bottleneck) ---
  message("Loading trained Random Forest model...")
  t2 <- proc.time()
  rf_model <- readRDS(rf_model_path)
  message("  Model loaded: ", round((proc.time() - t2)[3], 1), "s")

  message("Running predict()...")
  t3 <- proc.time()
  predictions <- predict(rf_model, newdata = cell_data)
  message("  Prediction elapsed: ", round((proc.time() - t3)[3], 1), "s")

  cell_data[, prediction := predictions]

  # --- Write output ---
  message("Writing predictions...")
  fwrite(cell_data, output_path)
  message("Pipeline complete.")

  return(cell_data)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows; `paste` + named-vector hash lookups | Vectorized `data.table` keyed join; zero `lapply` |
| `compute_neighbor_stats` | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R function calls | `data.table` grouped `by=` aggregation in C; 5 passes |
| String operations | ~25.8M `paste()` calls + ~25.8M named-vector lookups | Zero `paste()`; integer joins only |
| Estimated runtime | 86+ hours | Minutes (typically 5â€“20 min on 16 GB laptop) |
| RF model | Preserved (unchanged) | Preserved (unchanged) |
| Numerical results | Original `max`/`min`/`mean` | Identical `max`/`min`/`mean` |

The true bottleneck was never Random Forest inference â€” it was millions of interpreted R loop iterations performing string construction and hash-table lookups for spatial neighbor feature engineering.
END_RESPONSE>>>
