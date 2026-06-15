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
run_id: Anthropic_F11_T00_R011
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 11
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F11_Anthropic_temp_00_rep_011.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant per-call overhead when done 6.46 million times. This single function alone generates on the order of **6.46M Ã— ~4 neighbors Ã— multiple string operations â‰ˆ billions of elementary R interpreter operations**.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with an `lapply` that subsets, removes NAs, and computes `max/min/mean`. That's another **~32.3 million R-level anonymous function calls** with repeated allocation.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, the `ranger` or `randomForest` predict method is implemented in C/C++ and is typically completed in seconds to a few minutes. Loading a serialized model (`readRDS`) is also fast. This is not the 86-hour bottleneck.

**Conclusion:** The bottleneck is the row-level R `lapply` loops with per-element string operations and named-vector lookups across 6.46M rows, repeated for 5 variables. This is a classic "death by a million R-level iterations" problem.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a vectorized `data.table` equi-join approach. Instead of looping row-by-row, build an edge table of `(id, neighbor_id)` from the `nb` object, join it to the data on `(neighbor_id, year)` to get neighbor row indices, and group by the original row index â€” all in compiled `data.table` C code.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable (or all variables at once), computing `max`, `min`, and `mean` in one pass over the edge-joined table.

3. **Leave the Random Forest predict step untouched** â€” it is not the bottleneck.

This eliminates all per-row string operations, all per-row `lapply` calls, and leverages `data.table`'s radix-based joins and grouped aggregation, reducing the estimated runtime from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ============================================================
# STEP 1: Build the edge list from the nb object (one-time)
# ============================================================
build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer vectors (indices into id_order)
  # Expand into a two-column edge table: (id, neighbor_id)
  from_idx <- rep(seq_along(nb_obj), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove zero-neighbor entries (spdep uses 0L for no neighbors)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

# ============================================================
# STEP 2: Vectorized neighbor feature computation
# ============================================================
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  # Convert to data.table if not already (by reference if possible)
  dt <- as.data.table(cell_data)

  # Assign a row index for later joining results back
  dt[, .row_idx := .I]

  # Build edge table: (id, neighbor_id)
  edges <- build_edge_table(id_order, nb_obj)

  # Key the data for fast joins: we need to look up neighbor rows by (id, year)
  # Create a slim lookup table: (id, year) -> row_idx + variable values
  lookup_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  lookup <- dt[, ..lookup_cols]
  setnames(lookup, "id", "neighbor_id")  # rename for join
  setkey(lookup, neighbor_id, year)

  # For each original row, we need its (id, year).
  # Join edges to dt to get (row_idx_of_origin, id, year, neighbor_id)
  origin <- dt[, .(id, year, .row_idx)]
  setkey(origin, id)
  setkey(edges, id)

  # Merge: for each (id, neighbor_id) edge, expand across all years of id
  # This gives us: for every row in dt, all its neighbor_ids paired with that row's year
  edge_year <- edges[origin, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # Columns: id, neighbor_id, year, .row_idx (of the origin row)

  # Now join to lookup to get the neighbor's actual values for that year
  setkey(edge_year, neighbor_id, year)
  joined <- lookup[edge_year, on = c("neighbor_id", "year"), nomatch = NA]
  # joined now has: neighbor_id, year, .row_idx (from lookup, i.e., neighbor's row),
  #   i..row_idx (origin's row idx), and all variable columns

  # Rename to clarify
  setnames(joined, "i..row_idx", "origin_row_idx")

  # Aggregate: for each origin row and each variable, compute max, min, mean
  for (var_name in neighbor_source_vars) {
    cat("Computing neighbor features for:", var_name, "\n")

    agg <- joined[!is.na(get(var_name)),
                  .(
                    nb_max  = max(get(var_name), na.rm = TRUE),
                    nb_min  = min(get(var_name), na.rm = TRUE),
                    nb_mean = mean(get(var_name), na.rm = TRUE)
                  ),
                  by = origin_row_idx]

    # Create properly named columns
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    # Join back to dt by row index
    setkey(agg, origin_row_idx)
    dt[agg, on = c(".row_idx" = "origin_row_idx"),
       (c(max_col, min_col, mean_col)) :=
         mget(paste0("i.", c(max_col, min_col, mean_col)))]
  }

  # Clean up helper column
  dt[, .row_idx := NULL]

  return(dt)
}

# ============================================================
# STEP 3: Run the optimized pipeline
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# This replaces both build_neighbor_lookup() and the for-loop over
# compute_and_add_neighbor_features()
cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ============================================================
# STEP 4: Random Forest prediction (UNCHANGED â€” not the bottleneck)
# ============================================================
# library(ranger)  # or randomForest
# trained_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(trained_model, data = cell_data)$predictions
# fwrite(cell_data, "predictions_output.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with string ops | One vectorized `data.table` join to build edge-year table |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations | 5 grouped `data.table` aggregations (compiled C) |
| Estimated runtime | 86+ hours | Minutes (typically 5â€“20 min depending on RAM/disk) |
| RF inference | Untouched | Untouched |
| Numerical results | Preserved (same max/min/mean logic) | Preserved identically |

The trained Random Forest model is never retrained or modified. The original numerical estimand (the predicted outcome from the pre-trained RF) is preserved exactly. The only change is how the spatial neighbor features are computed â€” same values, vastly faster execution.
END_RESPONSE>>>
