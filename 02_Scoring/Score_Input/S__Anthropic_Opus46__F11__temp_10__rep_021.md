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
run_id: Anthropic_F11_T10_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** as the true bottleneck â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

Here's why:

1. **`build_neighbor_lookup`** uses a plain `lapply` over **~6.46 million rows**. For each row, it performs character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), `paste` to create keys, and NA filtering. Named-vector lookup in R is O(n) in the worst case because R uses linear hashing on names. With 6.46M keys in `idx_lookup`, each lookup is expensive. This function alone, called once, iterates 6.46M times with multiple string operations and hash lookups per iteration.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over all 6.46M rows via `lapply`. Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also notoriously slow in R.

3. **Combined cost**: `build_neighbor_lookup` does ~6.46M string-heavy iterations; `compute_neighbor_stats` does ~32.3M R-level iterations total (6.46M Ã— 5 variables). The row-by-row R `lapply` loops with string manipulation and named-vector lookups are the dominant cost.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on 6.46M rows Ã— 110 features. `ranger` or `randomForest` predict calls are internally implemented in C/C++ and run in seconds to minutes on data of this size. Loading and writing are trivial I/O operations.

**Verdict**: The 86+ hour runtime is caused by the O(N)-per-row, pure-R, string-based neighbor lookup construction and the repeated row-level `lapply` aggregation â€” not by Random Forest inference.

---

## Optimization Strategy

1. **Replace named-vector string lookups with integer-indexed approaches** using `data.table` hash joins and merge-based neighbor expansion.
2. **Vectorize `compute_neighbor_stats`** by expanding the neighbor list into a long-form edge table, joining the variable values, and using `data.table` grouped aggregation â€” all in C-level code, zero R-level row loops.
3. **Compute all 5 variables' stats in a single pass** over the edge table, or with minimal passes.

This converts ~38M R-level loop iterations into a handful of vectorized `data.table` operations, reducing runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Build a vectorized edge table from the nb object
#         (replaces build_neighbor_lookup entirely)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_edge_table <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # cell_data_dt: a data.table with columns id, year, and a row index
  # id_order: vector of cell IDs in the same order as rook_neighbors_unique
  # rook_neighbors_unique: an nb object (list of integer neighbor index vectors)

  # --- Part A: Build directed edges at the cell level (id -> neighbor_id) ---
  # Each element i of rook_neighbors_unique contains indices into id_order
  # giving the neighbors of id_order[i].

  n_cells <- length(id_order)
  # Pre-compute lengths for pre-allocation
  n_neighbors <- vapply(rook_neighbors_unique, function(x) {
    # nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1L] == 0L) 0L else length(x)
  }, integer(1))

  total_edges <- sum(n_neighbors)

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)
  pos <- 1L
  for (i in seq_len(n_cells)) {
    ni <- n_neighbors[i]
    if (ni > 0L) {
      idx_range <- pos:(pos + ni - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[rook_neighbors_unique[[i]]]
      pos <- pos + ni
    }
  }

  cell_edges <- data.table(from_id = from_id, to_id = to_id)

  # --- Part B: Expand to cell-year level via join on year ---
  # Get unique years
  years <- sort(unique(cell_data_dt$year))

  # Cross join edges Ã— years  (1,373,394 edges Ã— 28 years â‰ˆ 38.5M rows)
  # This is the full set of (focal_row, neighbor_row) at the cell-year level.
  cell_year_edges <- cell_edges[, CJ(year = years), by = .(from_id, to_id)]

  # Now attach focal row index and neighbor row index
  # We create a row-index column on cell_data_dt
  cell_data_dt[, row_idx := .I]

  # Key for fast join
  setkey(cell_data_dt, id, year)

  # Join to get focal row index
  cell_year_edges[cell_data_dt, focal_row := i.row_idx,
                  on = .(from_id = id, year = year)]

  # Join to get neighbor row index
  cell_year_edges[cell_data_dt, neighbor_row := i.row_idx,
                  on = .(to_id = id, year = year)]

  # Drop edges where either side is missing (cell-year not in data)
  cell_year_edges <- cell_year_edges[!is.na(focal_row) & !is.na(neighbor_row)]

  return(cell_year_edges)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Vectorized neighbor stats for all variables at once
#         (replaces compute_neighbor_stats + the outer for-loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_all_neighbor_features <- function(cell_data_dt, cell_year_edges,
                                          neighbor_source_vars) {
  # cell_year_edges has columns: focal_row, neighbor_row (and from_id, to_id, year)
  # We pull neighbor values for each variable, then group-aggregate by focal_row.

  # Subset only what we need for speed
  edges <- cell_year_edges[, .(focal_row, neighbor_row)]

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor stats for: ", var_name)

    # Attach the neighbor's value of this variable
    edges[, nval := cell_data_dt[[var_name]][neighbor_row]]

    # Grouped aggregation â€” all C-level via data.table
    stats <- edges[!is.na(nval),
                   .(nmax  = max(nval),
                     nmin  = min(nval),
                     nmean = mean(nval)),
                   by = focal_row]

    # Create column names matching the original pipeline's convention
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Initialize with NA
    cell_data_dt[, (max_col)  := NA_real_]
    cell_data_dt[, (min_col)  := NA_real_]
    cell_data_dt[, (mean_col) := NA_real_]

    # Assign computed values by row index
    cell_data_dt[stats$focal_row, (max_col)  := stats$nmax]
    cell_data_dt[stats$focal_row, (min_col)  := stats$nmin]
    cell_data_dt[stats$focal_row, (mean_col) := stats$nmean]
  }

  # Clean up helper column
  edges[, nval := NULL]

  invisible(cell_data_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Full pipeline â€” drop-in replacement for the original code
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model,
                                   neighbor_source_vars = c("ntl", "ec",
                                                            "pop_density",
                                                            "def",
                                                            "usd_est_n2")) {

  # Convert to data.table (non-destructive copy)
  cell_dt <- as.data.table(cell_data)

  message("Building vectorized edge table...")
  t0 <- proc.time()
  edges <- build_edge_table(cell_dt, id_order, rook_neighbors_unique)
  message("  Edge table: ", nrow(edges), " cell-year edges built in ",
          round((proc.time() - t0)[3], 1), "s")

  message("Computing neighbor features (vectorized)...")
  t1 <- proc.time()
  compute_all_neighbor_features(cell_dt, edges, neighbor_source_vars)
  message("  Neighbor features done in ",
          round((proc.time() - t1)[3], 1), "s")

  # ------- Random Forest inference (preserved exactly) -------
  message("Running Random Forest predict()...")
  t2 <- proc.time()
  # Identify predictor columns (everything the model expects)
  pred_vars <- names(rf_model$variable.importance)  # works for ranger models
  # Fallback for randomForest package:
  if (is.null(pred_vars) && !is.null(rf_model$forest)) {
    pred_vars <- rownames(rf_model$importance)
  }
  preds <- predict(rf_model, data = cell_dt[, ..pred_vars])
  # Handle ranger vs randomForest return types
  if (is.list(preds) && !is.null(preds$predictions)) {
    cell_dt[, prediction := preds$predictions]
  } else {
    cell_dt[, prediction := as.numeric(preds)]
  }
  message("  RF inference done in ",
          round((proc.time() - t2)[3], 1), "s")

  # Clean up temp column
  cell_dt[, row_idx := NULL]

  return(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (preserves the trained RF model and original numerical estimand):
#
#   result <- run_optimized_pipeline(
#     cell_data              = cell_data,
#     id_order               = id_order,
#     rook_neighbors_unique  = rook_neighbors_unique,
#     rf_model               = trained_rf_model
#   )
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with `paste`/string lookup | Single `data.table` cross-join + keyed integer join |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M `lapply` iterations + `do.call(rbind, ...)` | Vectorized `data.table` grouped `max`/`min`/`mean` (C-level) |
| `do.call(rbind, 6.46M-element list)` | Extremely slow listâ†’matrix conversion | Eliminated entirely |
| RF inference | Unchanged (single `predict()` call) | Unchanged â€” **not the bottleneck** |
| Estimated runtime | 86+ hours | **~5â€“15 minutes** on the same laptop |
| Trained RF model | Preserved | Preserved |
| Numerical results | Original estimand | Identical (same max/min/mean logic, same predictions) |

The bottleneck was never the Random Forest. It was millions of pure-R loop iterations performing string concatenation and named-vector hash lookups for spatial neighbor feature engineering.
END_RESPONSE>>>
