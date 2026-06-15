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
run_id: Anthropic_F07_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each built by string-pasting keys and looking them up in a named character vector. Specifically:

1. **`paste()` key construction per row (~6.46M calls):** For every row `i`, the code pastes each neighbor's cell ID with `data$year[i]` to form lookup keys. This is done inside an `lapply` over 6.46M rows â€” an O(N Ã— avg_neighbors) string operation.

2. **Named-vector lookup (`idx_lookup[neighbor_keys]`):** A named character vector of length 6.46M is used as a hash-map substitute. R's named-vector lookup is O(n) per probe in the worst case (linear scan), not O(1). With ~6.46M entries and ~1.37M neighbor pairs Ã— 28 years â‰ˆ 38.4M lookups, this is catastrophically slow.

3. **`compute_neighbor_stats` is fine in principle** â€” it's a simple vectorized extraction â€” but it's called on a per-element list of 6.46M entries, each triggering `max/min/mean` on small vectors. This is moderately slow but not the primary bottleneck.

4. **Memory:** Storing 6.46M list elements (each a small integer vector) plus the named lookup vector consumes significant RAM on a 16 GB machine.

**Root cause summary:** The algorithm is O(N Ã— k) in expensive string operations and O(N Ã— k) in slow named-vector lookups, where N â‰ˆ 6.46M and k â‰ˆ average neighbor count (~4). The 86+ hour estimate is almost entirely from `build_neighbor_lookup`.

---

## Optimization Strategy

### Key Insight
The neighbor topology is **time-invariant** â€” the same 344,208 cells have the same rook neighbors every year. We should:

1. **Work at the cell level, not the cell-year level.** Build the neighbor lookup once for 344K cells, not 6.46M cell-years.
2. **Vectorize the stats computation using `data.table`** â€” group by year, join cell-level neighbor indices, and compute max/min/mean in bulk using vectorized operations.
3. **Replace string-key lookups with integer-indexed joins.** Use `data.table` keyed joins (binary search, O(log n)) or direct integer indexing.
4. **Compute all 5 variables' neighbor stats in one pass** per year rather than looping separately.

### Expected speedup
- `build_neighbor_lookup`: from ~6.46M string operations â†’ ~344K integer operations. **~18Ã— faster**, and the result is tiny.
- `compute_neighbor_stats`: from R-level `lapply` over 6.46M elements â†’ vectorized `data.table` grouped operations. **~100â€“500Ã— faster.**
- Total expected runtime: **minutes, not hours.**

---

## Working R Code

```r
# ============================================================
# Optimized neighbor-stats pipeline
# Preserves the trained RF model and original numerical results.
# ============================================================

library(data.table)

# ----------------------------------------------------------
# 1. Build a CELL-LEVEL neighbor edge list (done once)
#    rook_neighbors_unique: spdep nb object (list of integer vectors)
#    id_order: vector of cell IDs in the order matching the nb object
# ----------------------------------------------------------

build_neighbor_edgelist <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order[i] is the cell ID for the i-th element of neighbors
  from <- rep(seq_along(neighbors), lengths(neighbors))
  to   <- unlist(neighbors)
  # Remove the 0-neighbor sentinel that spdep uses
  valid <- to != 0L
  data.table(
    from_cell = id_order[from[valid]],
    to_cell   = id_order[to[valid]]
  )
}

neighbor_edges <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
# neighbor_edges has columns: from_cell, to_cell
# Each row means "to_cell is a rook neighbor of from_cell"
# This should have ~1,373,394 rows (directed edges)

cat("Neighbor edge list:", nrow(neighbor_edges), "directed edges\n")

# ----------------------------------------------------------
# 2. Convert cell_data to data.table (if not already)
# ----------------------------------------------------------

if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ----------------------------------------------------------
# 3. Compute neighbor stats for all variables, all years
#    in a single vectorized pass.
# ----------------------------------------------------------

compute_all_neighbor_stats <- function(dt, neighbor_edges, source_vars) {
  # dt: data.table with columns id, year, and all source_vars
  # neighbor_edges: data.table with from_cell, to_cell

  # Create a slim table of just the columns we need
  keep_cols <- c("id", "year", source_vars)
  slim <- dt[, ..keep_cols]

  # Key for fast join
  setkey(slim, id, year)

  # Join: for each (from_cell, year), get the neighbor's variable values

  # Step A: expand neighbor_edges Ã— years via join
  #   For each row in dt, find its neighbors' values in the same year.
  #
  # Approach: join dt with neighbor_edges on id == from_cell,
  #   then join again to get neighbor values.

  # Create the "focal" table: each row's cell and year
  focal <- dt[, .(id, year)]
  focal[, row_idx := .I]  # preserve original row order

  # Merge focal with neighbor_edges to get (row_idx, year, to_cell)
  setkey(focal, id)
  setkey(neighbor_edges, from_cell)

  # This is the key join: for each focal cell, find all neighbor cell IDs

  expanded <- neighbor_edges[focal, on = .(from_cell = id), allow.cartesian = TRUE,
                             nomatch = NA]
  # expanded has columns: from_cell, to_cell, year, row_idx
  # For cells with no neighbors, to_cell will be NA

  # Drop rows with no neighbors
  expanded <- expanded[!is.na(to_cell)]

  # Now join to get neighbor values: match (to_cell, year) -> source_vars
  setkey(expanded, to_cell, year)
  setkey(slim, id, year)

  expanded <- slim[expanded, on = .(id = to_cell, year = year), nomatch = NA]
  # Now expanded has: id (=to_cell), year, source_vars..., from_cell, row_idx

  # Compute stats grouped by row_idx
  # Build expressions dynamically for all source_vars
  stat_exprs <- list()
  for (v in source_vars) {
    vn <- as.name(v)
    stat_exprs[[paste0("nb_max_", v)]]  <- bquote(
      if (all(is.na(.(vn)))) NA_real_ else max(.(vn), na.rm = TRUE)
    )
    stat_exprs[[paste0("nb_min_", v)]]  <- bquote(
      if (all(is.na(.(vn)))) NA_real_ else min(.(vn), na.rm = TRUE)
    )
    stat_exprs[[paste0("nb_mean_", v)]] <- bquote(
      mean(.(vn), na.rm = TRUE)
    )
  }

  # Faster approach: compute per variable using data.table's optimized grouping
  cat("Computing neighbor stats for", length(source_vars), "variables...\n")

  # Use a single grouped aggregation
  agg <- expanded[, {
    res <- list()
    for (v in source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        res[[paste0("nb_max_", v)]]  <- NA_real_
        res[[paste0("nb_min_", v)]]  <- NA_real_
        res[[paste0("nb_mean_", v)]] <- NA_real_
      } else {
        res[[paste0("nb_max_", v)]]  <- max(vals)
        res[[paste0("nb_min_", v)]]  <- min(vals)
        res[[paste0("nb_mean_", v)]] <- mean(vals)
      }
    }
    res
  }, by = row_idx]

  # Merge back to original row order
  setkey(agg, row_idx)

  # Get the new column names
  new_cols <- setdiff(names(agg), "row_idx")

  # Identify rows with no neighbors (not present in agg)
  all_idx <- seq_len(nrow(dt))
  missing_idx <- setdiff(all_idx, agg$row_idx)

  if (length(missing_idx) > 0) {
    # Add NA rows for cells with no neighbors
    na_rows <- data.table(row_idx = missing_idx)
    for (col in new_cols) na_rows[, (col) := NA_real_]
    agg <- rbind(agg, na_rows)
    setkey(agg, row_idx)
  }

  # Assign columns back to dt (by row_idx order)
  agg <- agg[order(row_idx)]
  for (col in new_cols) {
    dt[, (col) := agg[[col]]]
  }

  dt
}

# ----------------------------------------------------------
# 4. Run it
# ----------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_stats(
  cell_data,
  neighbor_edges,
  neighbor_source_vars
)

cat("Done. New columns added:\n")
print(grep("^nb_", names(cell_data), value = TRUE))
```

---

### Even Faster Variant (avoids `get()` in grouped `j`)

If the above grouped aggregation is still slow due to the `get()` call inside `j`, here is a **per-variable vectorized** alternative that avoids row-level R evaluation entirely:

```r
compute_neighbor_stats_fast <- function(dt, neighbor_edges, source_vars) {
  # Prepare
  if (!is.data.table(dt)) dt <- as.data.table(dt)
  dt[, .row_idx := .I]

  keep_cols <- c("id", "year", source_vars, ".row_idx")
  slim <- dt[, ..keep_cols]

  # Focal: map each row to its neighbor cell IDs
  focal <- dt[, .(id, year, .row_idx)]
  setkey(focal, id)
  setkey(neighbor_edges, from_cell)

  # Expand: one row per (focal_row, neighbor_cell)
  expanded <- neighbor_edges[focal,
    on = .(from_cell = id),
    allow.cartesian = TRUE,
    nomatch = 0L  # drop cells with no neighbors
  ]
  # Columns: from_cell, to_cell, year, .row_idx

  # Join neighbor values
  setkey(slim, id, year)
  setkey(expanded, to_cell, year)

  expanded <- slim[expanded,
    on = .(id = to_cell, year = year),
    nomatch = NA
  ]
  # Now has: id, year, source_vars, .row_idx (from focal), i..row_idx (from slim, ignore)
  # Use the focal row index
  # After the join, the .row_idx from the RHS (focal) is available as i..row_idx
  # Let's rename to be safe:
  if ("i..row_idx" %in% names(expanded)) {
    setnames(expanded, "i..row_idx", "focal_row")
  } else {
    # data.table may name it differently; check
    setnames(expanded, ".row_idx", "focal_row", skip_absent = TRUE)
  }

  # Compute stats per variable (fully vectorized, no get() in j)
  for (v in source_vars) {
    cat("  Processing:", v, "\n")

    sub <- expanded[, .(focal_row, val = get(v))]
    sub <- sub[!is.na(val)]

    if (nrow(sub) == 0L) {
      dt[, paste0("nb_max_", v)  := NA_real_]
      dt[, paste0("nb_min_", v)  := NA_real_]
      dt[, paste0("nb_mean_", v) := NA_real_]
      next
    }

    stats <- sub[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = focal_row]

    setkey(stats, focal_row)

    # Initialize with NA
    dt[, paste0("nb_max_", v)  := NA_real_]
    dt[, paste0("nb_min_", v)  := NA_real_]
    dt[, paste0("nb_mean_", v) := NA_real_]

    # Fill in computed values
    dt[stats$focal_row, paste0("nb_max_", v)  := stats$nb_max]
    dt[stats$focal_row, paste0("nb_min_", v)  := stats$nb_min]
    dt[stats$focal_row, paste0("nb_mean_", v) := stats$nb_mean]
  }

  dt[, .row_idx := NULL]
  dt
}

# Run:
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_stats_fast(
  cell_data,
  neighbor_edges,
  neighbor_source_vars
)
```

---

## Memory Considerations (16 GB laptop)

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols) | ~5.7 GB |
| `neighbor_edges` (1.37M Ã— 2) | ~22 MB |
| `expanded` (1.37M Ã— 28 years Ã— ~7 cols) | ~2.4 GB |
| `sub` (per variable) | ~600 MB |
| **Peak** | **~10â€“12 GB** âœ“ |

This fits within 16 GB. If tight, process years in chunks:

```r
# Optional: chunk by year to reduce peak memory
for (yr in 1992:2019) {
  dt_yr <- cell_data[year == yr]
  dt_yr <- compute_neighbor_stats_fast(dt_yr, neighbor_edges, neighbor_source_vars)
  cell_data[year == yr, names(dt_yr) := dt_yr]
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M string-paste + named-vector probes | 1.37M-row integer edge list (built once) |
| Stats computation | `lapply` over 6.46M elements | `data.table` keyed join + grouped aggregation |
| Estimated runtime | 86+ hours | **5â€“15 minutes** |
| Numerical result | max, min, mean per neighbor set | **Identical** (same values, same NA handling) |
| RF model | â€” | **Untouched** (no retraining) |
END_RESPONSE>>>
