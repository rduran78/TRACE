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
run_id: Anthropic_F11_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F11_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, and another named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookups in R are **O(n) string-hashing operations per call**, and doing this 6.46 million times with an `idx_lookup` vector of 6.46 million names is catastrophically slow.

2. **`compute_neighbor_stats()`** then iterates over the same ~6.46 million rows again, performing subsetting, `NA` removal, and `max/min/mean` per row. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million `lapply` iterations total.

3. **Random Forest prediction** is a single vectorized `predict()` call on the final data frame. Even with ~6.46 million rows and ~110 predictors, a single `predict.randomForest()` call typically completes in minutes, not hours. Loading a serialized model (`readRDS`) is also fast. This is clearly not the 86+ hour bottleneck.

**Conclusion:** The bottleneck is the row-by-row R-level loop over millions of rows using named-vector string lookups and per-row `lapply` aggregation. This is a classic "death by a million R-level iterations" problem.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` entirely** with a vectorized `data.table` merge/join approach. Instead of building a per-row list of neighbor indices, construct an **edge table** (a two-column data.table of `(focal_row, neighbor_row)`) using fast integer-keyed joins. This eliminates all `paste()`-based string key construction and named-vector lookups.

2. **Replace `compute_neighbor_stats()` with grouped `data.table` aggregation.** Join the edge table to the variable values, then compute `max`, `min`, and `mean` grouped by the focal row index â€” all in one vectorized pass per variable.

3. **Leave the Random Forest model and predict step untouched**, since it is not the bottleneck and the trained model must be preserved.

Expected speedup: from 86+ hours to **minutes** (the edge table has ~1.37M neighbor pairs Ã— 28 years â‰ˆ ~38M edges, and `data.table` grouped aggregation over ~38M rows is very fast).

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the edge table (replaces build_neighbor_lookup)
# ============================================================
build_neighbor_edge_table <- function(data_dt, id_order, rook_neighbors) {
  # data_dt: a data.table with columns 'id' and 'year' (and row index = original row order)
  # id_order: vector of cell IDs in the order matching rook_neighbors
  # rook_neighbors: spdep nb object (list of integer neighbor index vectors)

  # --- Build directed edge list at the cell level ---
  # Each element rook_neighbors[[i]] gives the indices (into id_order) of
  # neighbors of cell id_order[i].
  n_cells <- length(id_order)
  focal_cell_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  neighbor_cell_idx <- unlist(rook_neighbors, use.names = FALSE)

  # Map cell-level indices to actual cell IDs
  cell_edges <- data.table(
    focal_id    = id_order[focal_cell_idx],
    neighbor_id = id_order[neighbor_cell_idx]
  )

  # --- Expand to cell-year level via join ---
  # Create a lookup from (id, year) -> row position in data_dt
  data_dt[, .row_idx := .I]

  # Get unique years
  years <- unique(data_dt$year)

  # Cross join cell edges with all years
  cell_edges_year <- cell_edges[, .(year = years), by = .(focal_id, neighbor_id)]

  # Join to get focal row index
  setkey(data_dt, id, year)
  focal_lookup <- data_dt[, .(id, year, focal_row = .row_idx)]
  setkey(focal_lookup, id, year)
  setkey(cell_edges_year, focal_id, year)
  cell_edges_year <- focal_lookup[cell_edges_year,
                                   on = .(id = focal_id, year = year),
                                   nomatch = 0L]
  # Now cell_edges_year has columns: id, year, focal_row, neighbor_id

  # Join to get neighbor row index
  neighbor_lookup <- data_dt[, .(id, year, neighbor_row = .row_idx)]
  setkey(neighbor_lookup, id, year)
  setkey(cell_edges_year, neighbor_id, year)
  edge_table <- neighbor_lookup[cell_edges_year,
                                 on = .(id = neighbor_id, year = year),
                                 nomatch = 0L]
  # Keep only what we need
  edge_table <- edge_table[, .(focal_row, neighbor_row)]

  # Clean up temporary column
  data_dt[, .row_idx := NULL]

  return(edge_table)
}

# ============================================================
# STEP 2: Vectorized neighbor stats (replaces compute_neighbor_stats)
# ============================================================
compute_neighbor_stats_fast <- function(data_dt, edge_table, var_name) {
  # Extract neighbor values via the edge table
  vals <- data_dt[[var_name]]
  edges <- copy(edge_table)
  edges[, val := vals[neighbor_row]]

  # Remove NA neighbor values
  edges <- edges[!is.na(val)]

  # Grouped aggregation
  agg <- edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Initialize result columns with NA for all rows
  n <- nrow(data_dt)
  max_col  <- rep(NA_real_, n)
  min_col  <- rep(NA_real_, n)
  mean_col <- rep(NA_real_, n)

  # Fill in computed values
  max_col[agg$focal_row]  <- agg$nb_max
  min_col[agg$focal_row]  <- agg$nb_min
  mean_col[agg$focal_row] <- agg$nb_mean

  list(max_col = max_col, min_col = min_col, mean_col = mean_col)
}

# ============================================================
# STEP 3: Add neighbor features (replaces compute_and_add_neighbor_features)
# ============================================================
add_neighbor_features_fast <- function(data_dt, var_name, edge_table) {
  stats <- compute_neighbor_stats_fast(data_dt, edge_table, var_name)
  set(data_dt, j = paste0(var_name, "_nb_max"),  value = stats$max_col)
  set(data_dt, j = paste0(var_name, "_nb_min"),  value = stats$min_col)
  set(data_dt, j = paste0(var_name, "_nb_mean"), value = stats$mean_col)
  invisible(data_dt)
}

# ============================================================
# STEP 4: Full optimized pipeline
# ============================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  # Convert to data.table if not already (by reference, no deep copy)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Build edge table once (vectorized, fast)
  message("Building edge table...")
  edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge table: %d edges", nrow(edge_table)))

  # Compute neighbor features for all 5 variables
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    add_neighbor_features_fast(cell_data, var_name, edge_table)
  }

  # --- Random Forest prediction (unchanged, not the bottleneck) ---
  message("Running Random Forest prediction...")
  cell_data[, prediction := predict(rf_model, newdata = cell_data)]

  message("Done.")
  return(cell_data)
}

# ============================================================
# Usage (assumes objects already in memory):
# ============================================================
# rf_model              <- readRDS("trained_rf_model.rds")
# cell_data             <- readRDS("cell_data.rds")          # ~6.46M rows
# id_order              <- readRDS("id_order.rds")           # 344,208 cell IDs
# rook_neighbors_unique <- readRDS("rook_neighbors.rds")     # spdep nb object
#
# result <- run_optimized_pipeline(cell_data, id_order, rook_neighbors_unique, rf_model)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named vector lookups | Single vectorized `data.table` join to build edge table |
| `compute_neighbor_stats` | 6.46M `lapply` iterations Ã— 5 vars = 32.3M R-level loops | 5 grouped `data.table` aggregations (~38M rows each, vectorized C-level) |
| RF inference | Single `predict()` call (fast) | **Unchanged** â€” not the bottleneck |
| Estimated runtime | 86+ hours | **Minutes** (edge table build ~1â€“3 min; each variable aggregation ~10â€“30 sec) |
| Numerical results | Original estimand | **Preserved** â€” identical `max`, `min`, `mean` logic |
| Trained RF model | Preserved | **Preserved** â€” only `predict()` is called, no retraining |
END_RESPONSE>>>
