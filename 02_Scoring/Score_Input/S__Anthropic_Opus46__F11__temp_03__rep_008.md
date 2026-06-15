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
run_id: Anthropic_F11_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F11
prompt_family_slug: FALSE_RF_DIAGNOSIS
family_label: False RF diagnosis
family_group: Adversarial diagnosis
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F11_Anthropic_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** â€” specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` â€” as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, performing per-row character coercion (`as.character`), string pasting (`paste(..., sep="_")`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries significant overhead when called millions of times in a loop. This single function alone generates on the order of **6.46M Ã— ~4 neighbors â‰ˆ 25+ million** string constructions and hash lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46 million rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's **~32.3 million** R-level function invocations for the stats alone.

3. The `do.call(rbind, result)` at the end of `compute_neighbor_stats()` binds **6.46 million** 3-element vectors into a matrix â€” another expensive operation when done on a list of that length.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with 6.46M rows and 110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in optimized C/C++ and typically completes in seconds to a few minutes. Loading the model from disk is a one-time `readRDS()`. Writing predictions is a single vector write. None of these are loop-heavy R-interpreted operations.

**Conclusion:** The bottleneck is the R-level row-by-row loop over ~6.46 million observations performed in interpreted R, repeated across multiple functions and variables. The 86+ hour runtime is attributable to this feature engineering, not to the RF predict step.

---

## Optimization Strategy

1. **Replace the character-key hash lookup in `build_neighbor_lookup()` with integer arithmetic.** Instead of pasting strings and looking them up in a named vector, compute row indices directly using the formula: `row_index = (year_offset) * n_cells + cell_position`. This eliminates millions of string allocations and hash lookups.

2. **Vectorize `compute_neighbor_stats()` using `data.table` grouping.** Expand the neighbor lookup into a two-column data.table of `(row_id, neighbor_row_id)`, join on the variable values, and compute grouped `max`, `min`, `mean` in a single vectorized pass â€” no R-level `lapply` over millions of rows.

3. **Process all 5 variables in one pass** over the neighbor edge list rather than rebuilding grouped aggregations 5 times.

These changes reduce the complexity from **O(n_rows Ã— k) interpreted R operations** to **vectorized C-level operations** via `data.table`, cutting runtime from 86+ hours to minutes.

---

## Working R Code

```r
library(data.table)

# ==============================================================================
# OPTIMIZED build_neighbor_lookup (integer-arithmetic, no string ops)
# ==============================================================================
# Returns a data.table with columns: row_i, neighbor_row_i
# representing all directed neighbor-row pairs across all cell-years.
#
# Assumptions (matching the original code):
#   - cell_data is ordered (or will be ordered) by (id, year)
#   - id_order is the vector of unique cell IDs in the spatial grid order
#   - rook_neighbors_unique is an nb object (list of integer neighbor indices)
#   - years span a contiguous sequence present in cell_data

build_neighbor_edge_list <- function(cell_data, id_order, neighbors) {
  dt <- as.data.table(cell_data)

  # Ensure consistent ordering: cells within each year block
  # Map each cell id to its spatial index (position in id_order)
  id_to_spatial <- setNames(seq_along(id_order), as.character(id_order))

  # Unique sorted years
  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_offset <- setNames(seq_along(years) - 1L, as.character(years))

  n_cells <- length(id_order)

  # Build a fast row-index map: row_index for cell spatial_idx s and year y is:
  #   row_map[s, year_offset+1]
  # We build this as a matrix for O(1) lookup.
  dt[, spatial_idx := id_to_spatial[as.character(id)]]
  dt[, year_offset := year_to_offset[as.character(year)]]
  dt[, row_pos := .I]

  # Create the matrix: rows = spatial_idx (1..n_cells), cols = year_offset+1 (1..n_years)
  row_map <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_map[cbind(dt$spatial_idx, dt$year_offset + 1L)] <- dt$row_pos

  # Now expand neighbor relationships.
  # For each spatial cell s, neighbors[[s]] gives spatial indices of its neighbors.
  # For each year (column in row_map), we pair row_map[s, t] with row_map[nb, t].

  # Step 1: Build a cell-level edge list (spatial indices)
  from_spatial <- rep(seq_along(neighbors), lengths(neighbors))
  to_spatial   <- unlist(neighbors, use.names = FALSE)

  # Step 2: Expand across all years using vectorized outer operations
  # For efficiency, we iterate over years (only 28) â€” trivial loop.
  edge_list <- rbindlist(lapply(seq_len(n_years), function(t) {
    from_rows <- row_map[from_spatial, t]
    to_rows   <- row_map[to_spatial, t]
    valid <- !is.na(from_rows) & !is.na(to_rows)
    data.table(row_i = from_rows[valid], neighbor_row_i = to_rows[valid])
  }))

  # Clean up temporary columns
  dt[, c("spatial_idx", "year_offset", "row_pos") := NULL]

  return(edge_list)
}

# ==============================================================================
# OPTIMIZED compute_neighbor_stats for ALL variables at once (vectorized)
# ==============================================================================
compute_all_neighbor_features <- function(cell_data, edge_list, neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  n <- nrow(dt)

  # Attach variable values to the neighbor (target) side of each edge
  # We only need the columns for the source vars from the neighbor rows
  neighbor_vals <- dt[edge_list$neighbor_row_i, ..neighbor_source_vars]
  neighbor_vals[, row_i := edge_list$row_i]

  # Compute grouped stats for all variables simultaneously
  # Group by the focal row (row_i), compute max/min/mean per variable
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  # Build the aggregation call
  # For data.table, we use a single grouped operation
  stats <- neighbor_vals[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
      else list(max(vals), min(vals), mean(vals))
    }), neighbor_source_vars),
    by = row_i
  ]

  # The above returns list columns; let's use a cleaner approach:
  stats <- neighbor_vals[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 1L
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k]] <- NA_real_; out[[k+1L]] <- NA_real_; out[[k+2L]] <- NA_real_
      } else {
        out[[k]] <- max(vals); out[[k+1L]] <- min(vals); out[[k+2L]] <- mean(vals)
      }
      k <- k + 3L
    }
    names(out) <- agg_names
    out
  }, by = row_i]

  # Merge back to full data (some rows may have no neighbors)
  result_dt <- data.table(row_i = seq_len(n))
  result_dt <- merge(result_dt, stats, by = "row_i", all.x = TRUE, sort = TRUE)
  result_dt[, row_i := NULL]

  # Bind new columns to original cell_data
  for (col_name in agg_names) {
    dt[, (col_name) := result_dt[[col_name]]]
  }

  return(as.data.frame(dt))
}

# ==============================================================================
# FULL OPTIMIZED PIPELINE (drop-in replacement for the outer loop)
# ==============================================================================
# Usage â€” replaces the original outer loop:
#
#   neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
#   # Step 1: Build integer edge list (replaces build_neighbor_lookup)
#   edge_list <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
#
#   # Step 2: Compute all neighbor features in one vectorized pass
#   cell_data <- compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
#
#   # Step 3: RF inference (unchanged â€” this was never the bottleneck)
#   rf_model <- readRDS("trained_rf_model.rds")
#   cell_data$prediction <- predict(rf_model, newdata = cell_data)
#   write.csv(cell_data, "predictions.csv", row.names = FALSE)

# ==============================================================================
# EXAMPLE: Full invocation
# ==============================================================================
run_optimized_pipeline <- function(cell_data, id_order, rook_neighbors_unique,
                                   rf_model_path = "trained_rf_model.rds",
                                   output_path = "predictions.csv") {

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  message("Building neighbor edge list (integer arithmetic, no string ops)...")
  edge_list <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s edges", format(nrow(edge_list), big.mark = ",")))

  message("Computing neighbor features (vectorized, single pass)...")
  cell_data <- compute_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
  message("  Done.")

  message("Loading trained Random Forest model...")
  rf_model <- readRDS(rf_model_path)

  message("Running RF predict()...")
  cell_data$prediction <- predict(rf_model, newdata = cell_data)

  message("Writing predictions...")
  fwrite(as.data.table(cell_data), output_path)

  message("Pipeline complete.")
  return(cell_data)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Bottleneck** | `build_neighbor_lookup` + `compute_neighbor_stats` (R-level `lapply` over 6.46M rows with string ops) | Eliminated |
| **Lookup method** | `paste()` â†’ named-vector hash lookup per row | Integer matrix indexing `row_map[spatial_idx, year_col]` |
| **Neighbor stats** | `lapply` over 6.46M rows Ã— 5 variables = ~32.3M R-level iterations | Single `data.table` grouped aggregation over an integer edge list |
| **String allocations** | ~25M+ `paste()` calls | Zero |
| **Estimated runtime** | 86+ hours | Minutes (estimated 5â€“20 min depending on hardware) |
| **RF model** | Preserved (not retrained) | Preserved (not retrained) |
| **Numerical output** | Same max/min/mean neighbor stats â†’ same predictions | Identical numerical estimand |

The trained Random Forest model is never retrained â€” only `predict()` is called, exactly as before. The optimization targets the true bottleneck: the interpreted-R, row-by-row spatial feature engineering.
END_RESPONSE>>>
