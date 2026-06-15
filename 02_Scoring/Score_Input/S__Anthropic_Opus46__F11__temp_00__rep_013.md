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
run_id: Anthropic_F11_T00_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** as the dominant bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a base-R `lapply` over **~6.46 million rows**, and inside each iteration it performs `as.character()` coercion, `paste()` string construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash-table probing per call, and doing this 6.46 million times with string key construction each time is extremely expensive. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** then iterates over the 6.46 million-element lookup list again, computing `max`, `min`, and `mean` for each row's neighbors. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million `lapply` iterations total with per-element subsetting and aggregation.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46 million rows and 110 predictors, a single `predict.randomForest()` or `predict.ranger()` call is highly optimized C/C++ code and typically completes in seconds to minutes â€” orders of magnitude faster than the feature engineering above.

**Conclusion:** The bottleneck is the row-level R-loop-based spatial feature construction, not RF inference.

---

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed joins.** Instead of building string keys like `"id_year"` and doing named-vector lookups millions of times, use `data.table` integer-keyed joins.

2. **Vectorize `build_neighbor_lookup`** by expanding the neighbor list into a flat edge table (`row_i`, `neighbor_row_j`) using `data.table`, then computing all neighbor statistics in one vectorized grouped aggregation per variable â€” eliminating all per-row `lapply` calls.

3. **Compute all 5 variables' neighbor stats in a single pass** (or at least via vectorized `data.table` grouped operations) instead of 5 separate `lapply` loops over 6.46M rows.

This reduces complexity from ~32M interpreted R-loop iterations with string operations to a handful of vectorized `data.table` joins and group-by aggregations.

---

## Working R Code

```r
library(data.table)

#' Optimized: build a flat edge table mapping each row index to its neighbor row indices.
#' Replaces build_neighbor_lookup (the primary bottleneck).
build_neighbor_edge_table <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a row index .row_id

  # Step 1: Build a mapping from cell id -> position in id_order (integer)
  id_to_ref <- data.table(
    id  = id_order,
    ref = seq_along(id_order)
  )


  # Step 2: Expand the nb list into a flat edge list of (ref_idx, neighbor_id)
  #   neighbors[[ref]] gives integer indices into id_order for that cell's neighbors
  nb_lengths <- lengths(neighbors)
  from_ref   <- rep(seq_along(neighbors), nb_lengths)
  to_ref     <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  # edge_dt now has ~1.37M rows of directed neighbor pairs (cell-level, year-agnostic)

  # Step 3: Build a row-index lookup keyed on (id, year)
  row_lookup <- data_dt[, .(id, year, .row_id)]
  setkey(row_lookup, id, year)

  # Step 4: For every (from_id, year) combination, find the row index of the
  #         from-cell and every neighbor cell in that same year.
  #         We do this by joining edge_dt with the year dimension.

  # Get unique years
  years <- unique(data_dt$year)

  # Cross-join edges with years: each edge applies to every year
  # This creates ~1.37M * 28 â‰ˆ 38.5M rows â€” large but manageable in RAM
  edge_year <- CJ_edge_year(edge_dt, years)

  # Join to get from_row
  setkey(edge_year, from_id, year)
  setkey(row_lookup, id, year)
  edge_year[row_lookup, from_row := i..row_id, on = .(from_id = id, year = year)]

  # Join to get to_row (neighbor's row in same year)
  edge_year[row_lookup, to_row := i..row_id, on = .(to_id = id, year = year)]

  # Drop edges where either side is missing
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

  return(edge_year[, .(from_row, to_row)])
}

#' Helper: cross join edges Ã— years without full CJ explosion in one shot
CJ_edge_year <- function(edge_dt, years) {
  years_dt <- data.table(year = years)
  # Cross join: every edge paired with every year
  result <- edge_dt[, .(from_id, to_id)][
    , CJ_idx := 1L
  ][
    years_dt[, CJ_idx := 1L],
    on = "CJ_idx",
    allow.cartesian = TRUE
  ]
  result[, CJ_idx := NULL]
  return(result)
}

#' Optimized: compute neighbor max, min, mean for one variable using vectorized data.table ops.
#' Replaces compute_neighbor_stats.
compute_neighbor_stats_fast <- function(data_dt, edge_table, var_name) {
  # edge_table has columns: from_row, to_row
  # Attach the neighbor's value
  edges <- copy(edge_table)
  edges[, neighbor_val := data_dt[[var_name]][to_row]]

  # Drop NAs in neighbor values
  edges <- edges[!is.na(neighbor_val)]

  # Grouped aggregation: one group per from_row
  stats <- edges[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = from_row]

  return(stats)
}

#' Optimized: compute and add neighbor features for all source variables.
compute_and_add_all_neighbor_features <- function(cell_data, neighbor_source_vars,
                                                   id_order, rook_neighbors_unique) {
  # Convert to data.table if needed
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # Add row index

  cell_data[, .row_id := .I]

  # --- Build the edge table ONCE (replaces build_neighbor_lookup) ---
  message("Building vectorized neighbor edge table...")
  edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("Edge table built: %s edges across all cell-years.", format(nrow(edge_table), big.mark = ",")))

  # --- Compute neighbor stats for each variable (replaces per-row lapply loops) ---
  n_rows <- nrow(cell_data)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    stats <- compute_neighbor_stats_fast(cell_data, edge_table, var_name)

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Fill in computed values by row index
    cell_data[stats$from_row, (max_col)  := stats$nb_max]
    cell_data[stats$from_row, (min_col)  := stats$nb_min]
    cell_data[stats$from_row, (mean_col) := stats$nb_mean]
  }

  # Clean up helper column
  cell_data[, .row_id := NULL]

  return(cell_data)
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_and_add_all_neighbor_features(
  cell_data,
  neighbor_source_vars,
  id_order,
  rook_neighbors_unique
)

# --- Random Forest prediction (unchanged â€” not the bottleneck) ---
# library(ranger)  # or randomForest
# predictions <- predict(trained_rf_model, data = cell_data)
# cell_data$predicted <- predictions$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste()` + named-vector string lookups | Single vectorized `data.table` cross-join + integer-keyed join; built **once** |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M `lapply` iterations with per-row subsetting | 5 vectorized `data.table` grouped aggregations (`max`, `min`, `mean` by `from_row`) |
| Estimated runtime | 86+ hours | **Minutes** (vectorized C-level operations on ~38M-row edge table) |
| RF model | Preserved (untouched) | Preserved (untouched) |
| Numerical results | Original estimand | Identical `max`, `min`, `mean` neighbor statistics â†’ identical predictions |

The true bottleneck is the interpreted R-loop spatial feature engineering, not Random Forest inference. The fix is vectorization via `data.table` integer-indexed joins and grouped aggregations.
END_RESPONSE>>>
