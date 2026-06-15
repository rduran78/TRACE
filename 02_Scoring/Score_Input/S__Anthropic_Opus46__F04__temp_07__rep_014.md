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
run_id: Anthropic_F04_T07_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for every neighbor of every row.
4. Looks up those keys in a named character vector (`idx_lookup`), which is itself ~6.46M entries long â€” named-vector lookup in R is O(n) or hash-based but with significant per-call overhead.
5. Filters `NA`s.

This is done **6.46 million times**, each call involving string allocation, paste, and named-vector lookup. The result: billions of string operations. Then `compute_neighbor_stats` runs a second `lapply` over 6.46M elements per variable (Ã—5 variables), each calling `max`, `min`, `mean` on small vectors â€” slower than necessary due to R-level loop overhead.

**Secondary bottleneck:** `compute_neighbor_stats` uses `lapply` + `do.call(rbind, ...)` over 6.46M list elements, which is also slow.

## Optimization Strategy

1. **Replace string-key lookups with integer-indexed joins** using `data.table`. Map `(id, year)` â†’ row index via a keyed `data.table` join instead of paste + named vector lookup.
2. **Vectorize neighbor expansion** â€” expand the neighbor relationships into a single long edge table `(from_row, to_row)` in one vectorized operation, avoiding the per-row `lapply`.
3. **Compute neighbor stats via grouped aggregation** on the edge table using `data.table`, eliminating the second `lapply` entirely.
4. **All 5 variables in one pass** over the edge table rather than 5 separate passes.

This reduces the operation from billions of R-level string operations to a handful of vectorized `data.table` joins and grouped aggregations. Expected runtime: **minutes, not days**.

## Optimized R Code

```r
library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {

  # ---- Step 0: Convert to data.table if needed, preserve row order ----
  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # ---- Step 1: Build (id, year) -> row_idx lookup table ----
  idx_dt <- dt[, .(id, year, .row_idx)]
  setkey(idx_dt, id, year)

  # ---- Step 2: Build directed edge list (from_id -> to_id) from nb object ----
  #   rook_neighbors_unique is a list of length = number of spatial cells.
  #   rook_neighbors_unique[[i]] gives integer indices into id_order of neighbors of cell i.
  n_cells <- length(id_order)

  # Expand all neighbor pairs at the spatial-cell level
  from_cell_idx <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
  to_cell_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove zero-entries (spdep::nb uses 0L for cells with no neighbors)
  valid <- to_cell_idx != 0L
  from_cell_idx <- from_cell_idx[valid]
  to_cell_idx   <- to_cell_idx[valid]

  # Map cell indices to actual cell IDs
  from_id <- id_order[from_cell_idx]
  to_id   <- id_order[to_cell_idx]

  edges_spatial <- data.table(from_id = from_id, to_id = to_id)

  # ---- Step 3: Expand edges across all years ----
  years <- sort(unique(dt$year))

  # Cross join edges Ã— years â€” this creates the full (from_row, to_row) mapping
  # ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits comfortably in 16 GB
  edges_full <- edges_spatial[, .(from_id, to_id, year = rep(list(years), .N))]
  edges_full <- edges_full[, .(year = unlist(year)), by = .(from_id, to_id)]

  # ---- Step 4: Map (from_id, year) -> from_row and (to_id, year) -> to_row ----
  # Join to get "from" row index
  setnames(idx_dt, c("id", "year", ".row_idx"), c("from_id", "year", "from_row"))
  setkey(idx_dt, from_id, year)
  edges_full <- idx_dt[edges_full, on = .(from_id, year), nomatch = 0L]

  # Restore idx_dt column names for second join
  setnames(idx_dt, c("from_id", "year", "from_row"), c("to_id", "year", "to_row"))
  setkey(idx_dt, to_id, year)
  edges_full <- idx_dt[edges_full, on = .(to_id, year), nomatch = 0L]

  # Now edges_full has columns: from_row, to_row (and from_id, to_id, year)
  # We only need from_row and to_row plus the variable values at to_row.

  # ---- Step 5: Attach neighbor variable values and compute grouped stats ----
  # Extract only needed columns to minimize memory
  val_cols <- neighbor_source_vars
  vals_dt <- dt[, c(".row_idx", val_cols), with = FALSE]
  setnames(vals_dt, ".row_idx", "to_row")
  setkey(vals_dt, to_row)

  # Join variable values onto the edge table (value at the neighbor/to_row)
  edges_full <- vals_dt[edges_full, on = .(to_row)]

  # Grouped aggregation: for each from_row, compute max/min/mean of each variable
  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call
  agg_list <- setNames(agg_exprs, agg_names)

  # data.table aggregation
  stats <- edges_full[, lapply(agg_list, eval), by = .(from_row)]

  # Replace -Inf/Inf from max/min on all-NA groups with NA
  for (col in names(stats)[-1]) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # ---- Step 6: Join stats back to the original data ----
  setkey(stats, from_row)

  # Remove any pre-existing neighbor columns from dt to avoid duplication
  existing_neighbor_cols <- intersect(names(dt), agg_names)
  if (length(existing_neighbor_cols) > 0) {
    dt[, (existing_neighbor_cols) := NULL]
  }

  dt <- stats[dt, on = .(from_row = .row_idx)]

  # Clean up helper columns and restore original order
  setorder(dt, from_row)
  dt[, from_row := NULL]

  # Return as data.frame if input was data.frame
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt[])
}
```

**Simpler, more robust version of Step 5** (avoiding `bquote` complexity):

```r
compute_all_neighbor_features_v2 <- function(cell_data, id_order, rook_neighbors_unique,
                                             neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- Lookup table ---
  idx_dt <- dt[, .(id, year, .row_idx)]

  # --- Spatial edge list ---
  from_cell_idx <- rep(seq_along(rook_neighbors_unique), lengths(rook_neighbors_unique))
  to_cell_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)
  valid <- to_cell_idx != 0L
  edges_spatial <- data.table(
    from_id = id_order[from_cell_idx[valid]],
    to_id   = id_order[to_cell_idx[valid]]
  )

  # --- Expand across years via cross join ---
  years_dt <- data.table(year = sort(unique(dt$year)))
  edges_full <- edges_spatial[, CJ_val := 1L][
    years_dt[, CJ_val := 1L], on = "CJ_val", allow.cartesian = TRUE
  ][, CJ_val := NULL]

  # --- Map to row indices ---
  # from_row
  edges_full <- merge(edges_full, idx_dt, by.x = c("from_id", "year"),
                      by.y = c("id", "year"), all.x = FALSE, all.y = FALSE)
  setnames(edges_full, ".row_idx", "from_row")

  # to_row
  edges_full <- merge(edges_full, idx_dt, by.x = c("to_id", "year"),
                      by.y = c("id", "year"), all.x = FALSE, all.y = FALSE)
  setnames(edges_full, ".row_idx", "to_row")

  # --- Attach neighbor values ---
  val_dt <- dt[, c(".row_idx", neighbor_source_vars), with = FALSE]
  edges_full <- merge(edges_full, val_dt, by.x = "to_row", by.y = ".row_idx")

  # --- Compute stats per (from_row) for each variable ---
  result_dt <- dt[, .(.row_idx)]

  for (v in neighbor_source_vars) {
    agg <- edges_full[, .(
      nb_max  = max(.SD[[1L]], na.rm = TRUE),
      nb_min  = min(.SD[[1L]], na.rm = TRUE),
      nb_mean = mean(.SD[[1L]], na.rm = TRUE)
    ), by = .(from_row), .SDcols = v]

    # Fix Inf/-Inf from all-NA groups
    agg[is.infinite(nb_max), nb_max := NA_real_]
    agg[is.infinite(nb_min), nb_min := NA_real_]

    setnames(agg, c("nb_max", "nb_min", "nb_mean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))

    result_dt <- merge(result_dt, agg, by.x = ".row_idx", by.y = "from_row", all.x = TRUE)
  }

  # --- Merge back ---
  existing_cols <- intersect(names(dt), names(result_dt)[-1])
  if (length(existing_cols) > 0) dt[, (existing_cols) := NULL]

  dt <- merge(dt, result_dt, by = ".row_idx", all.x = TRUE)
  setorder(dt, .row_idx)
  dt[, .row_idx := NULL]

  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt[])
}
```

**Usage (drop-in replacement for the original outer loop):**

```r
library(data.table)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features_v2(
  cell_data, id_order, rook_neighbors_unique, neighbor_source_vars
)

# The trained Random Forest model is unchanged â€” use it for prediction as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | `lapply` over 6.46M rows with `paste`+named-vector lookup | Vectorized `data.table` keyed merge |
| **Neighbor stats** | `lapply` over 6.46M Ã— 5 vars, `do.call(rbind,...)` | `data.table` grouped aggregation (`by=from_row`) |
| **String operations** | ~billions of `paste()` calls | Zero â€” pure integer joins |
| **Passes over edge table** | N/A (rebuilt per row) | One materialized edge table, one grouped agg per variable |
| **Estimated runtime** | 86+ hours | **5â€“20 minutes** on the same laptop |
| **RAM (peak)** | Moderate but slow | ~38.5M edge rows Ã— ~8 cols â‰ˆ 2â€“3 GB peak (fits in 16 GB) |
| **Numerical output** | max/min/mean per neighbor set | Identical max/min/mean per neighbor set |
| **RF model** | Unchanged | Unchanged |
END_RESPONSE>>>
