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
run_id: Anthropic_F02_T07_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F02_Anthropic_temp_07_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is twofold:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over ~6.46 million rows
Each iteration performs character coercion, `paste` key construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is hash-based but still carries overhead per call. Doing this 6.46 million times, each time creating intermediate character vectors, is extremely slow and memory-hungry (millions of small character allocations trigger GC pressure).

### 2. `compute_neighbor_stats` â€” repeated per variable, pure R loop
`lapply` over 6.46 million elements, each time subsetting a numeric vector and computing `max/min/mean`, is called 5 times (once per neighbor source variable). That is ~32.3 million R-level function calls with per-element allocation.

**Combined**: the design is O(N Ã— k) in pure interpreted R with heavy allocation, where N â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4 for rook). The 86+ hour estimate is consistent with this.

---

## Optimization Strategy

| Principle | Action |
|---|---|
| **Eliminate per-row string operations** | Replace `paste(id, year)` key lookups with integer arithmetic: encode each row as `id * 100 + (year - 1992)` or use `data.table` keyed joins. |
| **Vectorize neighbor lookup** | Pre-expand the neighbor list into a flat edge table (`from_row`, `to_row`) using `data.table` keyed merge â€” one join instead of 6.46M `lapply` iterations. |
| **Vectorize stats computation** | Group-by aggregation on the edge table: `edge_dt[, .(max, min, mean), by = from_row]` â€” fully vectorized in `data.table` C code. |
| **Compute all 5 variables in one pass** | Melt or loop over columns *inside* the edge table rather than re-running the full lookup per variable. |
| **Memory management** | The flat edge table is ~6.46M Ã— 4 neighbors Ã— 2 integer columns â‰ˆ 200 MB â€” fits comfortably in 16 GB. Intermediate results are small. |

**Expected speedup**: from 86+ hours to roughly 5â€“15 minutes.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 1. Build a flat edge table (from_row -> to_row) ONCE
#    This replaces build_neighbor_lookup entirely.
# ---------------------------------------------------------------
build_edge_table <- function(cell_data, id_order, neighbors) {
  # cell_data must have columns: id, year
  # id_order: vector of cell IDs in the same order as the nb object

# neighbors: spdep nb object (list of integer index vectors into id_order)

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # Map each cell id to its position in id_order (reference index)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Expand neighbor list into a flat data.table: (ref_idx, neighbor_ref_idx)
  # This is ~1.37 M directed edges (not per year â€” per cell)
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_cells <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # Get all unique years
  years <- sort(unique(dt$year))

  # Cross-join edges Ã— years, then map to row indices via keyed join
  # To avoid a huge cross join in memory, we do two keyed joins.

  # Keyed lookup: (id, year) -> row_idx
  setkey(dt, id, year)

  # Expand edges to all years using CJ inside a join
  # More memory-efficient: replicate edge_cells for each year
  edge_year <- edge_cells[, .(from_id, to_id, year = rep(list(years), .N))]
  edge_year <- edge_year[, .(year = unlist(year)), by = .(from_id, to_id)]

  # Join to get from_row
  edge_year[dt, from_row := i.row_idx, on = .(from_id = id, year = year)]

  # Join to get to_row
  edge_year[dt, to_row := i.row_idx, on = .(to_id = id, year = year)]

  # Drop edges where either side is missing
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

  # Return only the integer index columns we need
  edge_year[, .(from_row, to_row)]
}

# ---------------------------------------------------------------
# 2. Compute neighbor stats for ALL variables at once
# ---------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edge_dt, neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]
  n <- nrow(dt)

  # Attach neighbor values to edge table for all vars at once
  # Only select needed columns to save memory
  val_cols <- intersect(neighbor_source_vars, names(dt))
  to_vals  <- dt[edge_dt$to_row, ..val_cols]

  # Bind from_row identifier
  to_vals[, from_row := edge_dt$from_row]

  # Aggregate: max, min, mean per from_row for each variable
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

  names(agg_exprs) <- agg_names

  stats <- to_vals[, lapply(agg_exprs, eval, envir = .SD), by = from_row]

  # --- cleaner aggregation (equivalent, more robust) ---
  # We do it variable by variable to keep code readable:
  stats_list <- vector("list", length(val_cols))

  for (i in seq_along(val_cols)) {
    v <- val_cols[i]
    s <- to_vals[, .(
      nb_max  = max(.SD[[1L]], na.rm = TRUE),
      nb_min  = min(.SD[[1L]], na.rm = TRUE),
      nb_mean = mean(.SD[[1L]], na.rm = TRUE)
    ), by = from_row, .SDcols = v]

    setnames(s, c("nb_max", "nb_min", "nb_mean"),
             paste0("neighbor_", c("max_", "min_", "mean_"), v))
    stats_list[[i]] <- s
  }

  # Merge all stats together on from_row
  merged <- stats_list[[1]]
  for (i in seq_along(stats_list)[-1]) {
    merged <- merge(merged, stats_list[[i]], by = "from_row", all = TRUE)
  }

  # Replace Inf/-Inf (from max/min on empty groups) with NA
  for (col in names(merged)[-1]) {
    set(merged, which(is.infinite(merged[[col]])), col, NA_real_)
  }

  # Join back to full data (rows with no neighbors get NA)
  dt <- merge(dt, merged, by.x = "row_idx", by.y = "from_row", all.x = TRUE)
  setorder(dt, row_idx)
  dt[, row_idx := NULL]

  # Return as data.frame to preserve downstream compatibility
  as.data.frame(dt)
}

# ---------------------------------------------------------------
# 3. Full pipeline (drop-in replacement for the outer loop)
# ---------------------------------------------------------------

# Build edge table once (~1â€“2 min, ~200â€“400 MB)
message("Building edge table...")
edge_dt <- build_edge_table(cell_data, id_order, rook_neighbors_unique)

# Compute and attach all neighbor features (~2â€“5 min)
message("Computing neighbor features...")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Clean up
rm(edge_dt); gc()

message("Done. Neighbor features added to cell_data.")

# ---------------------------------------------------------------
# 4. Predict with the existing trained Random Forest (unchanged)
# ---------------------------------------------------------------
# cell_data$predicted_gdp <- predict(trained_rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets â€” the edge table encodes the identical rook-neighbor relationships, matched on the same `(id, year)` pairs. |
| **Trained RF model** | The model object is never touched; only the input feature columns are constructed. Column names follow the same `neighbor_{max,min,mean}_{var}` convention. Adjust `setnames` if your trained model expects different names. |
| **Missing data handling** | Rows with no neighbors (e.g., boundary cells missing from a year) receive `NA`, identical to the original logic. |

---

## Summary of Gains

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup | ~6.46M `lapply` iterations with string ops | One `data.table` keyed join (~1â€“2 min) |
| Stats per variable | ~6.46M `lapply` iterations Ã— 5 vars | Vectorized `data.table` group-by Ã— 5 vars (~2â€“5 min) |
| Peak memory | Millions of small character vectors + list | ~400 MB flat integer edge table |
| **Total wall time** | **86+ hours** | **~5â€“15 minutes** |
END_RESPONSE>>>
