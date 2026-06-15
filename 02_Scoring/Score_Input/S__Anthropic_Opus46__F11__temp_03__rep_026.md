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
run_id: Anthropic_F11_T03_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering stage** â€” specifically `build_neighbor_lookup` and `compute_neighbor_stats` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup`** iterates with `lapply` over **~6.46 million rows**, and for each row it performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is backed by hashing/string matching, and doing this ~6.46 million times with multiple neighbor keys per row is extremely expensive. This single function likely accounts for the vast majority of the 86+ hour runtime.

2. **`compute_neighbor_stats`** then runs another `lapply` over ~6.46 million rows, subsetting a numeric vector by the index lists, computing `max`, `min`, and `mean`, and returning results that are bound with `do.call(rbind, ...)` on a 6.46-million-element list â€” itself a costly operation.

3. **This entire process repeats 5 times** (once per neighbor source variable for `compute_neighbor_stats`, though `build_neighbor_lookup` runs once).

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-trained model. Even with 6.46M rows and 110 predictors, a single `predict()` call on a `ranger` or `randomForest` object typically completes in seconds to minutes â€” orders of magnitude faster than the neighbor feature construction.

**Conclusion:** The bottleneck is the row-level R-loop-based spatial neighbor feature engineering, not Random Forest inference.

---

## Optimization Strategy

The key optimizations are:

1. **Vectorize `build_neighbor_lookup`** â€” Replace the per-row `lapply` with a fully vectorized join. Instead of building a lookup per row, exploit the structure: every cell with the same `id` has the same set of neighbor cell IDs, and every cell-year pair just needs its neighbors in the same year. We can construct this as a **merge/join on (neighbor_id, year)** using `data.table`, which is orders of magnitude faster than 6.46M iterations of string pasting and named-vector lookups.

2. **Vectorize `compute_neighbor_stats`** â€” Instead of per-row `lapply` with subsetting, use `data.table` grouped aggregation (`max`, `min`, `mean` by group) which is implemented in C and parallelized internally.

3. **Process all 5 variables in one pass** â€” Since the neighbor relationships are the same for all variables, we can compute all neighbor stats in a single grouped aggregation rather than repeating the join 5 times.

These changes reduce the complexity from O(N Ã— k) R-level iterations (where N â‰ˆ 6.46M and k â‰ˆ average neighbors) to a single vectorized join + grouped aggregation, bringing runtime from 86+ hours down to **minutes**.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature engineering.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + the outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all columns named in neighbor_source_vars.
#' @param id_order         integer vector of cell IDs in the order matching
#'                         rook_neighbors_unique (i.e., id_order[i] is the cell
#'                         ID for the i-th element of the nb object).
#' @param rook_neighbors   spdep::nb object (list of integer index vectors).
#' @param neighbor_source_vars character vector of variable names to compute
#'                         neighbor stats for.
#'
#' @return data.table with original columns plus, for each var in
#'         neighbor_source_vars, three new columns:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean.
#'         The original numerical estimand and all original columns are preserved.

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors,
                                          neighbor_source_vars) {

  # --- Step 1: Build an edge list (focal_id -> neighbor_id) ----------------
  # This replaces the per-row build_neighbor_lookup entirely.

  # Map positional index in nb object -> cell ID
  # rook_neighbors[[i]] contains positional indices of neighbors of id_order[i]
  n_cells <- length(id_order)

  # Pre-allocate vectors for the edge list
  # Total number of directed neighbor relationships
  n_edges <- sum(lengths(rook_neighbors))

  focal_ids    <- integer(n_edges)
  neighbor_ids <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- rook_neighbors[[i]]
    n_nb   <- length(nb_idx)
    if (n_nb > 0L) {
      focal_ids[pos:(pos + n_nb - 1L)]    <- id_order[i]
      neighbor_ids[pos:(pos + n_nb - 1L)] <- id_order[nb_idx]
      pos <- pos + n_nb
    }
  }

  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)

  # --- Step 2: Convert cell_data to data.table if needed -------------------
  dt <- as.data.table(cell_data)

  # Create a row-order key so we can restore original order at the end
  dt[, .row_order := .I]

  # --- Step 3: Build the neighbor table by joining edges Ã— years -----------
  # For each (focal_id, year), we need the variable values of all
  # (neighbor_id, year) rows.

  # Subset to only the columns we need for the neighbor lookup
  value_cols <- intersect(neighbor_source_vars, names(dt))
  neighbor_values <- dt[, c("id", "year", value_cols), with = FALSE]

  # Key for fast join
  setkey(neighbor_values, id, year)

  # Expand edges by year: join edges with neighbor_values on neighbor_id = id
  # This gives us, for every (focal_id, year), the variable values of each neighbor.
  setnames(edges, "neighbor_id", "id")
  setkey(edges, id)

  # Join: for each edge, pull in all years of the neighbor

  # We do this as a merge: edges Ã— neighbor_values on id (= neighbor_id)
  # Result: focal_id, id (neighbor_id), year, var1, var2, ...
  neighbor_data <- merge(edges, neighbor_values, by = "id", allow.cartesian = TRUE)

  # Rename for clarity
  setnames(neighbor_data, "id", "neighbor_id")

  # --- Step 4: Grouped aggregation -----------------------------------------
  # For each (focal_id, year), compute max/min/mean of each variable
  # across all neighbors.

  agg_exprs <- list()
  for (v in value_cols) {
    agg_exprs[[paste0(v, "_neighbor_max")]]  <-
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE)))
    agg_exprs[[paste0(v, "_neighbor_min")]]  <-
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE)))
    agg_exprs[[paste0(v, "_neighbor_mean")]] <-
      bquote(mean(.(as.name(v)), na.rm = TRUE))
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))

  neighbor_stats <- neighbor_data[, eval(agg_call),
                                  by = .(focal_id, year)]

  # Handle Inf/-Inf from max/min on all-NA groups -> convert to NA
  new_cols <- names(agg_exprs)
  for (col in new_cols) {
    set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
  }

  # --- Step 5: Join aggregated stats back to the original data -------------
  setkey(neighbor_stats, focal_id, year)
  setkey(dt, id, year)

  dt <- merge(dt, neighbor_stats,
              by.x = c("id", "year"),
              by.y = c("focal_id", "year"),
              all.x = TRUE)

  # Restore original row order
  setorder(dt, .row_order)
  dt[, .row_order := NULL]

  return(dt)
}


# ============================================================================
# USAGE â€” drop-in replacement for the original three-stage pipeline
# ============================================================================

# Original code replaced:
#   neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
#   for (var_name in neighbor_source_vars) {
#     cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
#   }

# New code:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# --- Random Forest prediction (UNCHANGED â€” model is preserved as-is) -------
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions$predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **`build_neighbor_lookup`** | 6.46M R-level iterations with string ops | Eliminated; replaced by a single vectorized edge-list construction (~344K iterations over cells, not rows) |
| **`compute_neighbor_stats`** | 6.46M R-level iterations Ã— 5 variables = ~32.3M iterations | Single `data.table` grouped aggregation (C-level, all 5 vars at once) |
| **`do.call(rbind, ...)`** | Binds 6.46M single-row matrices Ã— 5 times | Eliminated entirely |
| **Estimated runtime** | 86+ hours | Minutes (typically 2â€“10 min depending on hardware) |
| **Random Forest model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical estimand** | Preserved âœ“ (same max/min/mean logic) | Preserved âœ“ |

The true bottleneck was the **row-level R loop** over 6.46 million rows performing repeated string operations and named-vector lookups â€” not Random Forest inference. The fix is to vectorize the spatial neighbor feature engineering using `data.table` joins and grouped aggregation.
END_RESPONSE>>>
