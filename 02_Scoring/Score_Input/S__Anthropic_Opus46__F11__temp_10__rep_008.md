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
run_id: Anthropic_F11_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is effectively a hash-table probe, but doing it millions of times inside an interpreted R loop with per-iteration string allocations is extremely expensive. With ~1.37 million neighbor relationships distributed across 6.46M rows, this function alone generates tens of millions of string operations.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million elements via `lapply`, subsetting numeric vectors, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46-million-element list of 3-element vectors is also costly.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to minutes â€” orders of magnitude faster than the neighbor feature engineering.

**The 86+ hour runtime is dominated by the R-level interpreted loops performing millions of string operations and per-row neighbor lookups.**

## Optimization Strategy

1. **Vectorize `build_neighbor_lookup()`**: Replace the row-by-row `lapply` with a fully vectorized merge/join approach using `data.table`. Instead of building a lookup per row, expand the neighbor graph into an edge list, join it against the panel data on `(neighbor_id, year)`, and group by the original row to collect neighbor row indices.

2. **Vectorize `compute_neighbor_stats()`**: Once we have the edge list with matched row indices, compute all neighbor statistics (max, min, mean) for all 5 variables simultaneously using `data.table` grouped aggregation â€” a single pass in C-optimized code.

3. **Eliminate string key construction entirely**: Use integer-based joins on `(id, year)` pairs rather than pasting strings.

This reduces the complexity from ~6.46M Ã— k interpreted R iterations to a handful of vectorized `data.table` operations.

## Working R Code

```r
library(data.table)

# ==============================================================
# OPTIMIZED PIPELINE â€” replaces build_neighbor_lookup,
# compute_neighbor_stats, and the outer for-loop.
# Preserves the trained RF model and the original numerical
# estimand (identical neighbor max/min/mean features).
# ==============================================================

build_and_compute_all_neighbor_features <- function(cell_data_df,
                                                     id_order,
                                                     rook_neighbors_unique,
                                                     neighbor_source_vars) {

  # --- Step 0: Convert to data.table and add a row index --------------------
  dt <- as.data.table(cell_data_df)
  dt[, .row_idx := .I]

  # --- Step 1: Build the directed edge list from the nb object --------------
  #     Each element of rook_neighbors_unique[[i]] gives the *positional*
  #     indices (into id_order) of cell i's neighbors.
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    # spdep::nb encodes "no neighbors" as a single 0L
    nb <- nb[nb != 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))
  # edge_list now has columns: focal_id, neighbor_id

  # --- Step 2: Create a keyed lookup of (id, year) -> row_idx + values ------
  #     We only need the source vars plus id, year, and the row index.
  cols_needed <- unique(c("id", "year", ".row_idx", neighbor_source_vars))
  dt_key <- dt[, ..cols_needed]
  setkey(dt_key, id, year)

  # --- Step 3: Expand edges Ã— years via join --------------------------------
  #     For every (focal_id, neighbor_id) pair, we need every year present for
  #     the focal cell. Rather than a massive cross-join, we merge edges onto
  #     the focal rows, then look up the neighbor rows.

  # 3a. Get the (focal) row identifiers: focal_id + year + focal_row_idx
  focal_rows <- dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]

  # 3b. Join edges to focal rows on focal_id
  setkey(edge_list, focal_id)
  setkey(focal_rows, focal_id)
  expanded <- edge_list[focal_rows, on = "focal_id",
                        allow.cartesian = TRUE,
                        nomatch = NULL]
  # expanded has: focal_id, neighbor_id, year, focal_row_idx

  # 3c. Look up the neighbor's data for the same year
  setkey(expanded, neighbor_id, year)
  neighbor_data <- dt_key[, c("id", "year", neighbor_source_vars), with = FALSE]
  setnames(neighbor_data, "id", "neighbor_id")
  setkey(neighbor_data, neighbor_id, year)

  matched <- neighbor_data[expanded, on = c("neighbor_id", "year"),
                           nomatch = NA]
  # matched has: neighbor_id, year, <source_vars>, focal_id, focal_row_idx

  # --- Step 4: Compute grouped neighbor statistics --------------------------
  #     Group by focal_row_idx, compute max/min/mean for each variable.
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]] <-
      substitute(suppressWarnings(max(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("neighbor_min_", v)]] <-
      substitute(suppressWarnings(min(V, na.rm = TRUE)), list(V = v_sym))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      substitute(mean(V, na.rm = TRUE), list(V = v_sym))
  }

  # Build a single grouped aggregation call
  stats <- matched[, lapply(agg_exprs, eval),
                   by = focal_row_idx]

  # Replace Inf/-Inf (from max/min on all-NA) with NA to match original behavior
  inf_cols <- grep("^neighbor_(max|min)_", names(stats), value = TRUE)
  for (col in inf_cols) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # --- Step 5: Merge back onto the main data.table --------------------------
  setkey(stats, focal_row_idx)
  dt <- stats[dt, on = c(focal_row_idx = ".row_idx")]

  # Clean up helper columns
  dt[, focal_row_idx := NULL]

  # --- Step 6: Handle rows with no neighbors (they won't appear in stats) ---
  # The join above (stats[dt, ...]) already produces NA for unmatched rows,
  # which is correct â€” matching the original behavior.

  # Return as data.frame if the input was a data.frame, preserving compatibility
  if (!is.data.table(cell_data_df)) {
    setDF(dt)
  }

  return(dt)
}

# ==============================================================
# USAGE â€” drop-in replacement for the original pipeline section
# ==============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_and_compute_all_neighbor_features(
  cell_data_df          = cell_data,
  id_order              = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars  = neighbor_source_vars
)

# ==============================================================
# Random Forest prediction â€” UNCHANGED, model preserved as-is
# ==============================================================
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions
```

## Summary of Expected Improvement

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M interpreted R iterations with string ops | Eliminated; replaced by a vectorized `data.table` join |
| `compute_neighbor_stats` | 5 Ã— 6.46M interpreted R iterations + `do.call(rbind, ...)` | Single grouped `data.table` aggregation over all 5 vars |
| String key construction | ~6.46M `paste()` calls + named vector lookups | Zero; integer-keyed joins only |
| Estimated runtime | 86+ hours | **Minutes** (typically 5â€“20 min depending on hardware) |
| RF inference | Seconds to minutes | Unchanged â€” not the bottleneck |
| Numerical results | â€” | Identical max/min/mean features; trained RF model untouched |

The true bottleneck is the R-interpreted, row-by-row spatial neighbor feature construction â€” not the Random Forest inference. Vectorizing via `data.table` joins and grouped aggregation eliminates the bottleneck.
END_RESPONSE>>>
