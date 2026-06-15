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
run_id: Anthropic_F05_T10_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F05_Anthropic_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full chain of waste:

1. **O(N) string-key construction inside `build_neighbor_lookup`**: For each of the ~6.46M rows, `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates string keys and looks them up in a named vector. Named-vector lookup in R is hashed, but the keys are constructed per-row, and the entire named vector of 6.46M entries must be built and probed millions of times. This alone is enormous.

2. **The neighbor topology is year-invariant but is re-resolved per cell-year row**: The rook-neighbor structure is spatial â€” it doesn't change across years. Yet `build_neighbor_lookup` resolves neighbor *row indices* by pasting year onto spatial IDs for every single row. The same spatial neighbor resolution is repeated 28 times (once per year per cell), multiplied across all 344K cells.

3. **`compute_neighbor_stats` is called in a serial loop over 5 variables**, each time iterating over 6.46M entries via `lapply`. The per-variable pass is O(N Ã— avg_neighbors). With 5 variables, that's 5 full scans.

4. **`lapply` over 6.46M rows** is inherently slow in R due to interpreter overhead and poor cache/vectorization behavior.

**In summary**: the string-key construction is the visible hotspot, but the root cause is an algorithmic design that (a) conflates spatial topology with temporal indexing, (b) resolves neighbors row-by-row in interpreted R, and (c) processes variables one at a time. A full reformulation can drop the ~86-hour estimate to minutes.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Separate space from time** | Build a spatial-only neighbor lookup (344K cells), then expand to cell-year rows via integer indexing â€” no strings. |
| **Vectorize with `data.table`** | Melt neighbor pairs into a long edge table, join once, and compute grouped statistics in one vectorized pass per variable (or all at once). |
| **Eliminate `lapply` over 6.46M rows** | Replace with `data.table` grouped aggregation on the edge table â€” internally parallelized C code. |
| **Batch all 5 variables** | Compute max/min/mean for all neighbor-source variables in a single grouped pass. |

**Complexity**: The old approach is O(N_rows Ã— avg_neighbors) with massive per-element interpreter overhead. The new approach has the same theoretical complexity but executes in `data.table`'s C internals with radix-sorted joins.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                                 "def", "usd_est_n2")) {
  # ---------------------------------------------------------------
  # STEP 1: Build a spatial-only edge list (year-invariant)
  #         rook_neighbors_unique is an nb object: a list of integer

  #         vectors indexing into id_order.
  # ---------------------------------------------------------------
  message("Step 1: Building spatial edge list...")

  # For each cell index in id_order, get its neighbor cell IDs

  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(ref_idx) {
    nb_idx <- rook_neighbors_unique[[ref_idx]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
      return(NULL)
    }
    data.table(focal_id = id_order[ref_idx],
               neighbor_id = id_order[nb_idx])
  }))

  message(sprintf("  Edge list: %d directed neighbor pairs", nrow(edge_list)))

  # ---------------------------------------------------------------
  # STEP 2: Convert cell_data to data.table, add a row key
  # ---------------------------------------------------------------
  message("Step 2: Preparing data.table...")

  dt <- as.data.table(cell_data)

  # Ensure id and year columns exist
  stopifnot("id" %in% names(dt), "year" %in% names(dt))

  # ---------------------------------------------------------------
  # STEP 3: Create the neighbor-row lookup by joining edge_list
  #         with dt on (neighbor_id, year) â€” i.e., for every

  #         focal (id, year), find the rows of its spatial neighbors
  #         in the same year.
  # ---------------------------------------------------------------
  message("Step 3: Joining edges with data to resolve neighbor rows...")

  # Subset to only the columns we need for neighbor values
  cols_needed <- unique(c("id", "year", neighbor_source_vars))
  neighbor_dt <- dt[, ..cols_needed]

  # Rename for the join: neighbor_id -> id in neighbor_dt
  setnames(neighbor_dt, "id", "neighbor_id")

  # Key the neighbor data by (neighbor_id, year)
  setkeyv(neighbor_dt, c("neighbor_id", "year"))

  # Expand edges Ã— years: join edge_list with focal rows to get
  # (focal_id, year, neighbor_id), then join with neighbor_dt to
  # get neighbor values.

  # First, get focal (id, year) pairs
  focal_keys <- dt[, .(focal_id = id, year = year)]

  # Merge focal keys with edge_list to create
  # (focal_id, year, neighbor_id) â€” one row per directed
  # neighbor-pair-year combination.
  message("  Expanding edges across years...")
  edge_year <- edge_list[focal_keys, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: focal_id, neighbor_id, year

  message(sprintf("  Edge-year table: %d rows", nrow(edge_year)))

  # Join neighbor values onto the edge-year table
  message("  Joining neighbor variable values...")
  edge_year <- neighbor_dt[edge_year, on = .(neighbor_id, year), nomatch = NA]
  # Now edge_year has: neighbor_id, year, focal_id, + all neighbor_source_vars

  # ---------------------------------------------------------------
  # STEP 4: Compute grouped statistics (max, min, mean) per
  #         (focal_id, year) across all neighbor_source_vars at once.
  # ---------------------------------------------------------------
  message("Step 4: Computing neighbor statistics...")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
      bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  # Handle the edge case where all neighbor values are NA:
  # max/min with na.rm=TRUE on zero non-NA values gives Â±Inf;
  # mean gives NaN. We'll fix those after aggregation.

  stats_dt <- edge_year[, lapply(agg_exprs, eval), by = .(focal_id, year)]

  # Replace Inf/-Inf/NaN with NA (cells with no valid neighbors)
  for (col in agg_names) {
    set(stats_dt, which(!is.finite(stats_dt[[col]])), col, NA_real_)
  }

  message(sprintf("  Stats table: %d rows Ã— %d columns", nrow(stats_dt), ncol(stats_dt)))

  # ---------------------------------------------------------------
  # STEP 5: Handle focal (id, year) pairs that had NO neighbors
  #         (they won't appear in stats_dt). These get NA for all
  #         neighbor stats. We merge back onto dt.
  # ---------------------------------------------------------------
  message("Step 5: Merging neighbor features back onto cell_data...")

  # Remove any pre-existing neighbor columns to avoid conflicts
  existing_neighbor_cols <- intersect(names(dt), agg_names)
  if (length(existing_neighbor_cols) > 0) {
    dt[, (existing_neighbor_cols) := NULL]
  }

  # Merge
  setnames(stats_dt, "focal_id", "id")
  setkeyv(stats_dt, c("id", "year"))
  setkeyv(dt, c("id", "year"))

  dt <- stats_dt[dt, on = .(id, year)]  # right join keeps all original rows

  # ---------------------------------------------------------------
  # STEP 6: Convert back to data.frame if the original was one
  # ---------------------------------------------------------------
  if (is.data.frame(cell_data) && !is.data.table(cell_data)) {
    setDF(dt)
  }

  message("Done. Neighbor features added.")
  return(dt)
}
```

### Usage (drop-in replacement for the original outer loop)

```r
# --- Original code (86+ hours) ---
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# --- Optimized replacement (estimated 2-10 minutes) ---
cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is untouched.
# predict(rf_model, new_data) works exactly as before, because the
# output columns (neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl, ...)
# carry the same numerical values â€” just computed faster.
```

### Memory-Constrained Variant (if the ~190M-row `edge_year` table exceeds 16 GB)

```r
# Process one year at a time to cap peak memory at ~1/28th:
optimize_neighbor_features_chunked <- function(cell_data, id_order,
                                                rook_neighbors_unique,
                                                neighbor_source_vars) {
  library(data.table)
  dt <- as.data.table(cell_data)

  # Build spatial edge list (same as above)
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(ref_idx) {
    nb_idx <- rook_neighbors_unique[[ref_idx]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) return(NULL)
    data.table(focal_id = id_order[ref_idx], neighbor_id = id_order[nb_idx])
  }))

  cols_needed <- unique(c("id", "year", neighbor_source_vars))
  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  years <- sort(unique(dt$year))

  stats_list <- lapply(years, function(yr) {
    message(sprintf("  Processing year %d ...", yr))
    dt_yr <- dt[year == yr, ..cols_needed]

    # Neighbor values for this year
    nb_vals <- copy(dt_yr)
    setnames(nb_vals, "id", "neighbor_id")
    setkeyv(nb_vals, "neighbor_id")

    # Focal ids for this year
    focal_yr <- data.table(focal_id = dt_yr$id)

    # Expand edges for this year's focal cells
    ey <- edge_list[focal_yr, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
    ey[, year := yr]

    # Join neighbor values
    ey <- nb_vals[ey, on = .(neighbor_id), nomatch = NA]

    # Aggregate
    agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
      list(
        bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
        bquote(as.numeric(mean(.(as.name(v)), na.rm = TRUE)))
      )
    }), recursive = FALSE)
    names(agg_exprs) <- agg_names

    s <- ey[, lapply(agg_exprs, eval), by = .(focal_id)]
    s[, year := yr]
    for (col in agg_names) set(s, which(!is.finite(s[[col]])), col, NA_real_)
    s
  })

  stats_dt <- rbindlist(stats_list)
  setnames(stats_dt, "focal_id", "id")

  # Remove old neighbor columns if present
  existing <- intersect(names(dt), agg_names)
  if (length(existing) > 0) dt[, (existing) := NULL]

  setkeyv(stats_dt, c("id", "year"))
  setkeyv(dt, c("id", "year"))
  dt <- stats_dt[dt, on = .(id, year)]

  if (is.data.frame(cell_data) && !is.data.table(cell_data)) setDF(dt)
  return(dt)
}
```

---

## Performance Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor resolution** | 6.46M `paste` + named-vector lookups in `lapply` | One `data.table` keyed join (C-level merge-sort) |
| **Variable processing** | 5 serial `lapply` passes over 6.46M rows | Single grouped aggregation over all 5 variables |
| **Interpreter overhead** | ~6.46M R function calls per variable | Zero R-level row iteration |
| **String operations** | ~6.46M Ã— avg_neighbors `paste` calls | Zero (integer keys only) |
| **Estimated time** | 86+ hours | 2â€“10 minutes |
| **Peak RAM (chunked)** | ~same | ~2â€“4 GB |
| **Numerical output** | max, min, mean per variable | Identical values |
| **RF model** | â€” | Untouched; no retraining needed |
END_RESPONSE>>>
