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
run_id: Anthropic_F04_T03_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference â€” it is the two spatial neighbor feature construction functions. Here is why:

### `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations

For each of the ~6.46 million rows, the function:

1. Converts an integer ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current row's year to create string keys â€” this is extremely expensive at scale because `paste()` allocates a new string vector per row, and named-vector lookup by character key is O(n) in the worst case (hash collisions in R's internal environment-based names).
4. Indexes into `idx_lookup` (another named character vector) for each neighbor key.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates and looks up roughly **25.8 million string keys** inside an `lapply` loop â€” all in serial, interpreted R. The named-vector lookup `idx_lookup[neighbor_keys]` is particularly punishing because R's named vector access is not a true O(1) hash table at this scale.

### `compute_neighbor_stats` â€” repeated per variable

This function iterates over the 6.46M-element `neighbor_lookup` list **once per source variable** (5 times). Each call extracts neighbor values, removes NAs, and computes max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M small vectors is slow due to repeated memory allocation and the final row-binding of millions of 3-element vectors.

### Quantified cost estimate

| Step | Approximate operations | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M Ã— (string paste + named-vector lookup) | ~70â€“80% of total |
| `compute_neighbor_stats` Ã— 5 vars | 5 Ã— 6.46M Ã— (subset + summary stats) | ~20â€“30% of total |
| RF prediction (`predict()`) | Single vectorized call on ~6.46M Ã— 110 matrix | Fast (secondsâ€“minutes) |

**Estimated current wall time: 86+ hours**, almost entirely in these two functions.

---

## Optimization Strategy

### Principle: Replace string-key lookups and row-wise `lapply` with vectorized integer-indexed `data.table` joins and grouped aggregations.

| Current approach | Optimized approach |
|---|---|
| Named character vector lookup (`idx_lookup[paste(...)]`) | Integer-keyed `data.table` equi-join on `(neighbor_id, year)` |
| Per-row `lapply` building a list of 6.46M elements | Pre-explode all (row, neighbor_row) pairs into a single long edge table, then do grouped `data.table` aggregation |
| `do.call(rbind, lapply(...))` to collect stats | Single `data.table` `[, .(max, min, mean), by = row_idx]` call â€” fully vectorized in C |
| Runs 5 separate passes for 5 variables | Compute all 5 variables' stats in one grouped pass |

**Expected speedup: ~500Ã—â€“1000Ã— (from 86+ hours to ~5â€“15 minutes on the same laptop).**

Memory footprint: The edge table will have ~6.46M Ã— 4 â‰ˆ 25.8M rows Ã— a few integer/double columns â‰ˆ ~1â€“2 GB, well within 16 GB RAM.

The trained Random Forest model is never touched. The numerical output (max, min, mean of neighbor values) is identical â€” we are only changing the computational path, not the estimand.

---

## Working R Code

```r
library(data.table)

#' Build a long edge table mapping every cell-year row to its neighbor cell-year rows.
#' Replaces build_neighbor_lookup entirely â€” no string keys, no per-row lapply.
#'
#' @param cell_data   data.frame/data.table with columns `id` and `year` (and predictor columns).
#' @param id_order    integer vector: the cell IDs in the order matching the nb object.
#' @param neighbors   spdep nb object (list of integer index vectors into id_order).
#' @return A data.table with columns: row_idx (integer index into cell_data),
#'         neighbor_row_idx (integer index into cell_data).
build_edge_table <- function(cell_data, id_order, neighbors) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # --- Step 1: Expand the spatial nb object into a (cell_id, neighbor_id) edge list ---
  #     This is only ~1.37M rows (one per directed rook-neighbor relationship).
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  spatial_edges <- data.table(
    cell_id     = id_order[from_ref],
    neighbor_id = id_order[to_ref]
  )

  # --- Step 2: Build a (cell_id, year) -> row_idx lookup table ---
  row_lookup <- dt[, .(cell_id = id, year, row_idx)]
  setkey(row_lookup, cell_id, year)

  # --- Step 3: Cross spatial edges with all years to get (row_idx, neighbor_row_idx) ---
  #     Join spatial_edges to row_lookup twice: once for the focal cell, once for the neighbor.

  # Get unique years
  years <- sort(unique(dt$year))

  # Expand spatial edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
  # But many (neighbor_id, year) pairs may not exist, so the join will naturally filter.
  edge_years <- CJ_dt(spatial_edges, years)

  # First join: focal cell -> row_idx
  setkey(edge_years, cell_id, year)
  edge_years <- row_lookup[edge_years, nomatch = 0L,
                           on = .(cell_id, year)]
  setnames(edge_years, "row_idx", "focal_row_idx")

  # Second join: neighbor cell -> row_idx
  setkey(edge_years, neighbor_id, year)
  neighbor_lookup_dt <- row_lookup[, .(neighbor_id = cell_id, year,
                                       neighbor_row_idx = row_idx)]
  setkey(neighbor_lookup_dt, neighbor_id, year)
  edge_years <- neighbor_lookup_dt[edge_years, nomatch = 0L,
                                    on = .(neighbor_id, year)]

  edge_years[, .(focal_row_idx, neighbor_row_idx)]
}

#' Helper: cross join a data.table with a vector of years.
CJ_dt <- function(dt_edges, years) {
  dt_edges[, .SD[, .(year = years), by = .I],
           .SDcols = c("cell_id", "neighbor_id")][, I := NULL]
  # More memory-efficient version:
  idx <- rep(seq_len(nrow(dt_edges)), each = length(years))
  out <- dt_edges[idx]
  out[, year := rep(years, nrow(dt_edges))]
  out
}

#' Compute neighbor max, min, mean for ALL source variables in a single vectorized pass.
#'
#' @param cell_data           data.frame with the predictor columns.
#' @param edge_table          data.table from build_edge_table (focal_row_idx, neighbor_row_idx).
#' @param neighbor_source_vars character vector of column names to summarize.
#' @return The original cell_data with new columns appended:
#'         <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean for each var.
compute_all_neighbor_features <- function(cell_data, edge_table, neighbor_source_vars) {

  dt <- as.data.table(cell_data)
  n  <- nrow(dt)

  # Attach neighbor values to the edge table (only the columns we need)
  # This is a simple integer-index extraction â€” very fast.
  neighbor_vals <- dt[edge_table$neighbor_row_idx, ..neighbor_source_vars]
  neighbor_vals[, focal_row_idx := edge_table$focal_row_idx]

  # Grouped aggregation: one pass over the edge table computes all stats.
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  # Build the aggregation call programmatically
  stats <- neighbor_vals[,
    setNames(lapply(neighbor_source_vars, function(v) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) list(NA_real_, NA_real_, NA_real_)
      else list(max(vals), min(vals), mean(vals))
    }), neighbor_source_vars),
    by = focal_row_idx
  ]

  # The above returns list columns; we need to unpack.  Cleaner approach below.

  # ---- Cleaner single-pass aggregation ----
  stats <- neighbor_vals[, {
    out <- vector("list", length(neighbor_source_vars) * 3L)
    k <- 0L
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[k + 1L]] <- NA_real_
        out[[k + 2L]] <- NA_real_
        out[[k + 3L]] <- NA_real_
      } else {
        out[[k + 1L]] <- max(vals)
        out[[k + 2L]] <- min(vals)
        out[[k + 3L]] <- mean(vals)
      }
      k <- k + 3L
    }
    setNames(out, agg_names)
  }, by = focal_row_idx]

  # Left-join stats back to the original row order.
  # Rows with no neighbors (no entry in edge_table) get NA automatically.
  setkey(stats, focal_row_idx)
  for (col in agg_names) {
    dt[stats$focal_row_idx, (col) := stats[[col]]]
  }

  # Rows not present in stats already have NA (data.table default).
  as.data.frame(dt)
}


# =============================================================================
# MAIN EXECUTION â€” drop-in replacement for the original outer loop
# =============================================================================

# 1. Build the integer edge table (runs once, ~30 seconds)
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# 2. Compute and attach all 15 neighbor features in one vectorized pass (~2-5 min)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_features(cell_data, edge_table, neighbor_source_vars)

# 3. Random Forest prediction â€” unchanged, model object is preserved as-is.
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

### Simplified, Maximally Robust Alternative

If the programmatic aggregation above feels complex, here is a leaner version that processes one variable at a time but still achieves the critical speedup (eliminating string keys):

```r
library(data.table)

build_edge_table <- function(cell_data, id_order, neighbors) {
  dt <- data.table(id = cell_data$id, year = cell_data$year,
                   row_idx = seq_len(nrow(cell_data)))

  # Spatial edges (~1.37M rows)
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)
  sp_edges <- data.table(cell_id = id_order[from_ref],
                         neighbor_id = id_order[to_ref])

  # Expand by year via cross join
  yr <- data.table(year = sort(unique(dt$year)))
  sp_year <- sp_edges[, CJ(year = yr$year), by = .(cell_id, neighbor_id)]

  # Join focal row index
  setkey(dt, id, year)
  sp_year <- dt[sp_year, on = .(id = cell_id, year = year), nomatch = 0L]
  setnames(sp_year, "row_idx", "focal_row_idx")

  # Join neighbor row index
  sp_year <- dt[sp_year, on = .(id = neighbor_id, year = year), nomatch = 0L]
  setnames(sp_year, "row_idx", "neighbor_row_idx")

  sp_year[, .(focal_row_idx, neighbor_row_idx)]
}

compute_and_add_one_var <- function(cell_dt, edge_table, var_name) {
  vals <- cell_dt[[var_name]]
  et <- copy(edge_table)
  et[, v := vals[neighbor_row_idx]]
  et <- et[!is.na(v)]
  agg <- et[, .(vmax = max(v), vmin = min(v), vmean = mean(v)),
            by = focal_row_idx]
  new_cols <- paste0(var_name, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  cell_dt[agg$focal_row_idx, (new_cols[1]) := agg$vmax]
  cell_dt[agg$focal_row_idx, (new_cols[2]) := agg$vmin]
  cell_dt[agg$focal_row_idx, (new_cols[3]) := agg$vmean]
  invisible(cell_dt)
}

# --- Main ---
cell_data <- as.data.table(cell_data)
edge_table <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
for (v in neighbor_source_vars) {
  compute_and_add_one_var(cell_data, edge_table, v)
}

cell_data <- as.data.frame(cell_data)
# predict(trained_rf_model, newdata = cell_data)  # unchanged
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Key data structure | Named character vectors, per-row `lapply` | Integer-indexed `data.table` join + grouped aggregation |
| `build_neighbor_lookup` | ~6.46M `paste` + named-vector lookups | Single `data.table` equi-join (~25M rows, seconds) |
| `compute_neighbor_stats` | 5 Ã— `lapply` over 6.46M-element list + `do.call(rbind)` | 5 Ã— `data.table` `[, .(max,min,mean), by=]` (vectorized C) |
| Estimated wall time | **86+ hours** | **5â€“15 minutes** |
| Peak RAM | Moderate (list of 6.46M vectors) | ~2â€“3 GB (edge table + value column) â€” fits in 16 GB |
| RF model | Untouched | Untouched |
| Numerical output | max, min, mean of neighbor values | Identical max, min, mean of neighbor values |
END_RESPONSE>>>
