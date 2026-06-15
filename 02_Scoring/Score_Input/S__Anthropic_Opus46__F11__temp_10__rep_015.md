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
run_id: Anthropic_F11_T10_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F11_Anthropic_temp_10_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that **Random Forest inference** is the bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** uses `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string concatenation (`paste(..., sep="_")`), and named vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but has significant per-call overhead when executed 6.46 million times. With ~1.37 million neighbor relationships, each row averages ~4 neighbors, meaning millions of `paste()` and index-matching operations are performed sequentially in interpreted R.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time running `lapply` over 6.46 million rows, subsetting values, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level function invocations just for the stats computation.

3. **`do.call(rbind, result)`** on a 6.46-million-element list of 3-element vectors is itself an expensive operation repeated 5 times.

4. By contrast, Random Forest **inference** (`predict()`) on a pre-trained model is a single vectorized C/C++ call that processes the full matrix in one pass. Even with 6.46M rows Ã— 110 features, a single `predict()` call on a `ranger` or `randomForest` object typically completes in seconds to a few minutesâ€”orders of magnitude less than the neighbor feature pipeline.

**The bottleneck is the row-by-row `lapply`-based neighbor lookup construction and repeated neighbor statistics computation over 6.46 million rows, not the Random Forest inference.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of looping over every row and doing per-row string operations, construct an edge-list data.table of `(source_id, neighbor_id)` from the `nb` object, join it to the data on `(id, year)`, and compute grouped statistics in one pass.

2. **Replace `compute_neighbor_stats()`** with a single `data.table` grouped aggregation per variable (or all variables at once), eliminating all `lapply` overhead.

3. **Eliminate the intermediate list-of-integer-vectors** (`neighbor_lookup`) entirely. The edge-list + join approach never needs it.

This converts ~32+ million interpreted R function calls into a handful of vectorized `data.table` operations that run in compiled C code.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature engineering.
#' Replaces build_neighbor_lookup() + compute_neighbor_stats() loop.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the same order as rook_neighbors_unique
#' @param rook_neighbors  spdep::nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended
compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  dt <- as.data.table(cell_data)

  # --- Step 1: Build edge list from nb object (vectorized) ---
  # Each element rook_neighbors[[i]] is a vector of neighbor indices into id_order.
  # Convert to a two-column data.table: (source_id, neighbor_id)
  n_neighbors <- lengths(rook_neighbors)
  source_idx  <- rep(seq_along(rook_neighbors), times = n_neighbors)
  neighbor_idx <- unlist(rook_neighbors, use.names = FALSE)

  # Remove the spdep "no neighbor" sentinel (0)
  valid <- neighbor_idx > 0L
  source_idx   <- source_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  edge_list <- data.table(
    source_id   = id_order[source_idx],
    neighbor_id = id_order[neighbor_idx]
  )

  # --- Step 2: Assign a row key to every (id, year) in the main data ---
  # We need to join neighbor_id to data rows to get neighbor variable values.
  # Create a lookup keyed on (id, year) containing the variable columns we need.
  neighbor_val_cols <- neighbor_source_vars
  lookup_dt <- dt[, c("id", "year", neighbor_val_cols), with = FALSE]
  setnames(lookup_dt, "id", "neighbor_id")
  setkeyv(lookup_dt, c("neighbor_id", "year"))

  # Also need source-side year for the join (neighbors share the same year).
  # Expand edge_list Ã— years via join with source rows.
  source_years <- dt[, .(id, year)]
  setnames(source_years, "id", "source_id")

  # Add a row index to preserve original row order for final assignment.
  dt[, .row_idx := .I]
  source_years_idx <- dt[, .(source_id = id, year, .row_idx)]

  # --- Step 3: Merge edge_list with source-side (id, year) to get (source_id, year, neighbor_id) ---
  setkeyv(edge_list, "source_id")
  setkeyv(source_years_idx, "source_id")

  # This is the large expansion: each source row Ã— its neighbors
  # ~6.46M rows Ã— ~4 neighbors avg = ~25.8M rows (manageable in 16GB)
  edges_with_year <- edge_list[source_years_idx,
                               on = "source_id",
                               allow.cartesian = TRUE,
                               nomatch = NULL]
  # Result columns: source_id, neighbor_id, year, .row_idx

  # --- Step 4: Join neighbor variable values ---
  setkeyv(edges_with_year, c("neighbor_id", "year"))
  edges_with_vals <- lookup_dt[edges_with_year,
                               on = c("neighbor_id", "year"),
                               nomatch = NA]
  # Result has: neighbor_id, year, <var columns>, source_id, .row_idx

  # --- Step 5: Compute grouped statistics per (source row, variable) ---
  # Group by .row_idx (which maps 1:1 to original rows)
  # Compute max, min, mean for each variable in one pass.
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    v_sym <- as.name(v)
    suffix_max  <- paste0("nb_max_", v)
    suffix_min  <- paste0("nb_min_", v)
    suffix_mean <- paste0("nb_mean_", v)
    agg_exprs[[suffix_max]]  <- bquote(max(.(v_sym),  na.rm = TRUE))
    agg_exprs[[suffix_min]]  <- bquote(min(.(v_sym),  na.rm = TRUE))
    agg_exprs[[suffix_mean]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  stats_dt <- edges_with_vals[, eval(agg_call), by = .row_idx]

  # Replace Inf/-Inf (from max/min on all-NA groups) with NA
  stat_cols <- setdiff(names(stats_dt), ".row_idx")
  for (col in stat_cols) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  # --- Step 6: Handle rows with zero neighbors (they won't appear in stats_dt) ---
  # These rows simply get NA for all neighbor features â€” they are absent from
  # stats_dt, and the join below will fill them with NA automatically.

  # --- Step 7: Join back to original data ---
  setkeyv(stats_dt, ".row_idx")
  dt <- stats_dt[dt, on = ".row_idx"]

  # Clean up helper column
  dt[, .row_idx := NULL]

  return(dt)
}


# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

# Original code replaced:
# -------------------------------------------------------
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }
# -------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the original columns plus:
#   nb_max_ntl, nb_min_ntl, nb_mean_ntl,
#   nb_max_ec, nb_min_ec, nb_mean_ec,
#   ... etc. for all 5 variables (15 new columns total).

# The trained Random Forest model is untouched â€” proceed directly to prediction:
# predictions <- predict(rf_model, data = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with `paste()` + named vector lookup | Vectorized edge-list via `rep`/`unlist` + `data.table` join |
| **Stats computation** | 5 Ã— 6.46M `lapply` iterations (32.3M R function calls) | Single grouped `data.table` aggregation (compiled C) |
| **`do.call(rbind, ...)`** | 5 calls binding 6.46M 3-element vectors | Eliminated entirely |
| **Estimated runtime** | 86+ hours | **Minutes** (edge-list expansion ~26M rows is well within 16GB RAM and `data.table`'s optimized grouping) |
| **RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical estimand** | Preserved âœ“ (same max/min/mean) | Preserved âœ“ |

The true bottleneck is the **interpreted, row-by-row neighbor feature construction**, not Random Forest inference. The fix is to vectorize the entire pipeline using `data.table` joins and grouped aggregation.
END_RESPONSE>>>
