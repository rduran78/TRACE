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
run_id: Anthropic_F11_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, and another named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookups in R are hash-table-like but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates tens of millions of string operations and named lookups. This single function likely accounts for the majority of the 86+ hour runtime.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46M rows via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level function calls total.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with ~6.46M rows and ~110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes â€” orders of magnitude faster than the feature engineering above.

**The bottleneck is the row-level R `lapply` loops with per-element string construction and named-vector lookups across 6.46 million rows, repeated for 5 variables.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` entirely** with a vectorized `data.table` merge/join approach. Instead of building a per-row list of neighbor indices via string keys, we construct a flat edge-list data.table of `(row_i, neighbor_row_j)` pairs using keyed joins â€” eliminating all `paste()`, `as.character()`, and named-vector lookups.

2. **Replace `compute_neighbor_stats()` with a single vectorized `data.table` group-by aggregation** per variable. Join the edge list to the data values, then aggregate by source row using `data.table`'s optimized `max`, `min`, `mean` â€” all in C.

3. **Leave the Random Forest predict step untouched**, since it is not the bottleneck.

This reduces the estimated runtime from 86+ hours to roughly **minutes**.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Vectorized neighbor lookup construction (replaces build_neighbor_lookup)
# ---------------------------------------------------------------
build_neighbor_edge_list <- function(data_dt, id_order, rook_neighbors) {

  # data_dt: a data.table with columns 'id', 'year', and a row index 'row_i'

  # id_order: vector of cell IDs in the order matching rook_neighbors
  # rook_neighbors: spdep nb object (list of integer neighbor indices)

  # Step A: Build a flat edge list of (focal_id, neighbor_id) from the nb object
  n_cells <- length(id_order)
  focal_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  neighbor_idx <- unlist(rook_neighbors)

  # Remove zero-neighbor entries (spdep uses 0L for no-neighbor cells)
  valid <- neighbor_idx != 0L
  focal_idx <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edges <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # Step B: Cross-join edges with years present in the data
  #   For each (focal_id, neighbor_id) pair, the relationship holds for every year.
  #   We join to the data to get row indices for both focal and neighbor.

  # Create a keyed lookup: (id, year) -> row_i
  row_lookup <- data_dt[, .(id, year, row_i)]
  setkey(row_lookup, id, year)

  # Get all unique years
  years <- unique(data_dt$year)

  # Expand edges across all years
  edges_expanded <- edges[, .(year = years), by = .(focal_id, neighbor_id)]

  # Join to get focal row index
  setnames(edges_expanded, "focal_id", "id")
  edges_expanded <- row_lookup[edges_expanded, on = .(id, year), nomatch = 0L]
  setnames(edges_expanded, c("id", "row_i"), c("focal_id", "focal_row"))

  # Join to get neighbor row index
  setnames(edges_expanded, "neighbor_id", "id")
  edges_expanded <- row_lookup[edges_expanded, on = .(id, year), nomatch = 0L]
  setnames(edges_expanded, c("id", "row_i"), c("neighbor_id", "neighbor_row"))

  return(edges_expanded[, .(focal_row, neighbor_row)])
}

# ---------------------------------------------------------------
# 2. Vectorized neighbor stats (replaces compute_neighbor_stats)
# ---------------------------------------------------------------
compute_neighbor_stats_vec <- function(data_dt, edge_dt, var_name) {
  # edge_dt has columns: focal_row, neighbor_row
  # Attach the neighbor's value for var_name
  neighbor_vals <- data_dt[[var_name]][edge_dt$neighbor_row]

  agg_dt <- data.table(
    focal_row = edge_dt$focal_row,
    val       = neighbor_vals
  )

  # Remove NAs before aggregation
  agg_dt <- agg_dt[!is.na(val)]

  # Aggregate
  stats <- agg_dt[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = focal_row]

  return(stats)
}

# ---------------------------------------------------------------
# 3. Add neighbor features to the dataset
# ---------------------------------------------------------------
compute_and_add_neighbor_features_vec <- function(data_dt, var_name, edge_dt) {
  stats <- compute_neighbor_stats_vec(data_dt, edge_dt, var_name)

  # Initialize columns with NA
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  data_dt[, (max_col)  := NA_real_]
  data_dt[, (min_col)  := NA_real_]
  data_dt[, (mean_col) := NA_real_]

  # Fill in computed values by row index
  data_dt[stats$focal_row, (max_col)  := stats$nb_max]
  data_dt[stats$focal_row, (min_col)  := stats$nb_min]
  data_dt[stats$focal_row, (mean_col) := stats$nb_mean]

  return(data_dt)
}

# ---------------------------------------------------------------
# 4. Full optimized pipeline
# ---------------------------------------------------------------
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model) {
  # Convert to data.table if needed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Add row index

  cell_data[, row_i := .I]

  # Build the edge list ONCE (vectorized, replaces build_neighbor_lookup)
  message("Building vectorized neighbor edge list...")
  edge_dt <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge list built: %s edges", format(nrow(edge_dt), big.mark = ",")))

  # Compute neighbor features for all 5 variables
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    cell_data <- compute_and_add_neighbor_features_vec(cell_data, var_name, edge_dt)
  }

  # Clean up helper column
  cell_data[, row_i := NULL]

  # ---- Random Forest inference (untouched, not the bottleneck) ----
  message("Running Random Forest predictions...")
  predictions <- predict(rf_model, newdata = cell_data)
  cell_data[, prediction := predictions]

  message("Pipeline complete.")
  return(cell_data)
}

# ---------------------------------------------------------------
# Usage (preserves trained RF model and original numerical estimand):
# ---------------------------------------------------------------
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model          # pre-trained, never retrained
# )
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows; `paste`/named-vector lookups | Single vectorized `data.table` keyed join; edge list built once |
| **Neighbor stats** | `lapply` over 6.46M rows Ã— 5 vars = 32.3M R calls | `data.table` grouped aggregation in C, 5 calls |
| **RF inference** | `predict()` â€” already fast | Unchanged |
| **Estimated runtime** | 86+ hours | Minutes |
| **Trained RF model** | Preserved | Preserved |
| **Numerical results** | Original `max`/`min`/`mean` | Identical `max`/`min`/`mean` â€” estimand preserved |

The true bottleneck is the R-level row-by-row neighbor feature engineering, not Random Forest inference. The fix is vectorization via `data.table` joins and grouped aggregation.
END_RESPONSE>>>
