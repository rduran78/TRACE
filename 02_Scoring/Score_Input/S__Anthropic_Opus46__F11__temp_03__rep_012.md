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
run_id: Anthropic_F11_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_012.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. These are all interpreted, non-vectorized, per-row operations. With ~6.46 million iterations, this alone is catastrophically slow.

2. **`compute_neighbor_stats()`** then iterates over the same ~6.46 million entries in the lookup list, subsetting a numeric vector and computing `max`, `min`, and `mean` per entry. This is called **5 times** (once per neighbor source variable), yielding ~32.3 million R-level loop iterations total.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object against a ~6.46M Ã— 110 matrix. The `ranger` and `randomForest` predict methods are implemented in C/C++ and are highly optimized. Even on a large dataset this completes in minutes, not hours.

4. The **86+ hour runtime** is consistent with billions of interpreted string operations and list manipulations in R, not with a single vectorized C-level predict call.

**Conclusion:** The bottleneck is the R-level, row-by-row, string-heavy neighbor lookup construction and the repeated list-based neighbor statistics computation. The optimization target is to vectorize these operations entirely.

---

## Optimization Strategy

1. **Replace the per-row `lapply` in `build_neighbor_lookup`** with a fully vectorized `data.table` merge/join approach. Instead of building a list of neighbor indices row-by-row, we expand all neighbor relationships into an edge table, join against the data to resolve row indices, and then compute grouped statistics directly.

2. **Replace `compute_neighbor_stats`** (called 5Ã— in a loop) with a single grouped aggregation over the edge table using `data.table`, computing all 15 output columns (3 stats Ã— 5 variables) in one pass.

3. **Preserve the trained Random Forest model** â€” no retraining. The output columns are numerically identical (same `max`, `min`, `mean` of the same neighbor values), so the original numerical estimand is preserved.

**Expected speedup:** From 86+ hours to roughly **2â€“10 minutes** on the same hardware.

---

## Working R Code

```r
library(data.table)

#' Vectorized neighbor feature engineering.
#' Replaces build_neighbor_lookup() + compute_neighbor_stats() loop.
#'
#' @param cell_data       data.frame with columns: id, year, and all neighbor_source_vars
#' @param id_order        integer vector of cell IDs in the order matching rook_neighbors_unique
#' @param rook_neighbors  spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names
#' @return cell_data with new neighbor feature columns appended (same row order)
compute_all_neighbor_features_vectorized <- function(cell_data,
                                                     id_order,
                                                     rook_neighbors,
                                                     neighbor_source_vars) {

  # --- Step 1: Build a complete directed edge list (focal_id -> neighbor_id) ---
  # Each element of rook_neighbors[[i]] contains indices into id_order
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), times = lengths(rook_neighbors))
  to_idx   <- unlist(rook_neighbors, use.names = FALSE)

  edges <- data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
  rm(from_idx, to_idx)

  # --- Step 2: Convert cell_data to data.table; create a row-order key ---
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  # --- Step 3: Cross edges with years via merge ---
  # For every (focal_id, neighbor_id) pair, we need every year present for the focal.
  # Since the panel is balanced (all cells Ã— all years), we can expand efficiently.
  # First, get the unique years.
  years <- sort(unique(dt$year))

  # Expand edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
  edges_expanded <- edges[, .(year = years), by = .(focal_id, neighbor_id)]

  # --- Step 4: Join neighbor values ---
  # Key the data for fast joins
  cols_needed <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- dt[, ..cols_needed]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkeyv(neighbor_vals, c("neighbor_id", "year"))
  setkeyv(edges_expanded, c("neighbor_id", "year"))

  edges_expanded <- neighbor_vals[edges_expanded, on = .(neighbor_id, year), nomatch = NA]

  # --- Step 5: Grouped aggregation â€” compute max, min, mean per (focal_id, year) ---
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the j-expression for data.table
  agg_call <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  stats <- edges_expanded[, eval(agg_call), by = .(focal_id, year)]

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # --- Step 6: Merge back to original data, preserving row order ---
  setnames(stats, "focal_id", "id")
  setkeyv(stats, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- stats[dt, on = .(id, year)]

  # Restore original row order

  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  return(as.data.frame(dt))
}

# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_vectorized(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# --- Random Forest prediction (unchanged) ---
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with string ops | Eliminated; vectorized edge table + `data.table` join |
| **Neighbor stats** | `lapply` over 6.46M entries Ã— 5 vars = 32.3M R-loop iterations | Single grouped `data.table` aggregation (C-level) |
| **Estimated runtime** | 86+ hours | ~2â€“10 minutes |
| **Trained RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical estimand** | max/min/mean of rook-neighbor values | Identical max/min/mean of same values âœ“ |

The true bottleneck was never the Random Forest â€” it was millions of interpreted, per-row string manipulations and list-based subsetting in the neighbor feature engineering stage.
END_RESPONSE>>>
