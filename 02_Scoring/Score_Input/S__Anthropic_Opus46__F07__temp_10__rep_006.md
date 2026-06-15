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
run_id: Anthropic_F07_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F07_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which creates a list of ~6.46 million elements, each constructed via per-row string pasting, named-vector lookups, and NA filtering inside an `lapply`. This is O(N) in interpreted R with heavy per-iteration overhead:

1. **String-keyed lookups are slow.** `paste(id, year)` and named-vector indexing (`idx_lookup[neighbor_keys]`) for every row is extremely expensive at 6.46M rows.
2. **Per-row `lapply` over 6.46M rows** in base R is inherently slowâ€”each iteration has R function-call overhead.
3. **`compute_neighbor_stats` is also list-based**, iterating 6.46M elements and subsetting a vector by index each time, then calling `max/min/mean`.
4. **Memory pressure.** A 6.46M-element list of integer vectors, plus intermediate character vectors, can consume many GB on a 16 GB machine.

The 86+ hour estimate is almost entirely attributable to these two functions repeated for 5 variables.

---

## Optimization Strategy

**Replace the per-row list-based approach with a vectorized sparse-matrix multiplication / grouped aggregation approach:**

1. **Build a sparse adjacency matrix** (cell Ã— cell, ~344K Ã— ~344K, ~1.37M non-zero entries) from `rook_neighbors_unique` once. This is instant with `Matrix::sparseMatrix`.

2. **For each year, extract the variable column as a dense vector aligned to cells, then multiply by the sparse adjacency matrix** to get neighbor sums. Simultaneously compute neighbor counts (multiply a vector of ones), neighbor max, and neighbor min using efficient grouped operations.

3. **For max and min**, use `data.table` grouped joins: expand directed neighbor pairs, join the variable values, and compute `max/min/mean` grouped by `(id, year)`. With `data.table` this runs in seconds, not hours.

4. **Avoid any per-row `lapply` or string-key lookups entirely.**

This reduces runtime from 86+ hours to **minutes**. The numerical results are identical (same neighbor sets, same `max/min/mean`), preserving the original estimand. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)
library(Matrix)

# ============================================================
# 1. Build directed edge list from spdep nb object (once)
# ============================================================
build_edge_dt <- function(id_order, rook_neighbors_unique) {
  # rook_neighbors_unique is a list of integer index vectors (spdep nb object)
  # id_order is the vector of cell IDs in the order matching the nb object
  from_ref <- rep(seq_along(rook_neighbors_unique),
                  lengths(rook_neighbors_unique))
  to_ref   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-neighbor placeholders (spdep uses 0L for "no neighbors")
  valid <- to_ref != 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
}

# ============================================================
# 2. Compute neighbor stats for one variable (vectorized)
# ============================================================
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {
  # cell_dt must be a data.table with columns: id, year, <var_name>
  # edge_dt has columns: from_id, to_id  (directed: from -> to means
  #   "to is a neighbor of from")

  # We need, for each (from_id, year), the max/min/mean of var_name
  # across all neighbors (to_id) in that same year.

  # Step 1: Build lookup of (to_id, year) -> value
  val_col <- var_name
  lookup <- cell_dt[, .(id, year, val = get(val_col))]
  setkey(lookup, id, year)

  # Step 2: Expand edges Ã— years via join
  #   For each edge (from_id -> to_id), we need every year present for from_id.
  #   But since every cell has the same 28 years, we can cross-join edges with years.

  years <- sort(unique(cell_dt$year))

  # Cross join edges with years
  # To save memory, do the join directly:
  # For each (from_id, to_id, year), get val of to_id in that year.
  edge_year <- CJ_edge_year <- edge_dt[, .(from_id, to_id)]

  # Replicate for all years â€” but this would be 1.37M * 28 = 38.4M rows.
  # That's fine for data.table on 16 GB.
  edge_year <- edge_dt[, .(year = years), by = .(from_id, to_id)]

  # Join neighbor values
  setkey(edge_year, to_id, year)
  setkey(lookup, id, year)
  edge_year[lookup, neighbor_val := i.val, on = .(to_id = id, year = year)]

  # Step 3: Aggregate by (from_id, year)
  stats <- edge_year[!is.na(neighbor_val),
    .(nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)),
    by = .(from_id, year)
  ]

  # Return with standardized names
  setnames(stats, "from_id", "id")
  stats
}

# ============================================================
# 3. Master pipeline
# ============================================================
run_neighbor_feature_pipeline <- function(cell_data, id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {
  # Convert to data.table (non-destructive copy)
  cell_dt <- as.data.table(cell_data)

  # Build edge list once
  message("Building edge list...")
  edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
  message(sprintf("  %d directed edges", nrow(edge_dt)))

  # For each variable, compute and attach neighbor stats
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)

    # Name the new columns to match the original pipeline's convention
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")
    setnames(stats, c("nb_max", "nb_min", "nb_mean"),
             c(max_col, min_col, mean_col))

    # Join back to main table
    setkey(stats, id, year)
    setkey(cell_dt, id, year)

    # Remove old columns if they exist (idempotency)
    for (col in c(max_col, min_col, mean_col)) {
      if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
    }

    cell_dt <- stats[cell_dt, on = .(id, year)]

    # Rows with no valid neighbors get NA (automatically from the left join)
    message(sprintf("  Done: %s", var_name))
  }

  # Return as data.frame if the original was one
  if (!is.data.table(cell_data)) {
    setDF(cell_dt)
  }

  cell_dt
}

# ============================================================
# 4. Call it  (drop-in replacement for the original outer loop)
# ============================================================
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- run_neighbor_feature_pipeline(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now has the same 15 new columns (5 vars Ã— 3 stats)
# with numerically identical values to the original implementation.
# The trained Random Forest model is unchanged and can be used directly.
```

---

## Memory-Optimized Variant (if 38.4M rows is tight on 16 GB)

If memory is a concern, process years in chunks:

```r
compute_neighbor_stats_chunked <- function(cell_dt, edge_dt, var_name,
                                           chunk_size = 7) {
  years <- sort(unique(cell_dt$year))
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  val_col <- var_name
  lookup <- cell_dt[, .(id, year, val = get(val_col))]
  setkey(lookup, id, year)

  results <- rbindlist(lapply(year_chunks, function(yr_chunk) {
    # Subset lookup to this chunk of years
    lk <- lookup[year %in% yr_chunk]
    setkey(lk, id, year)

    # Expand edges for this year chunk only
    ey <- edge_dt[, .(year = yr_chunk), by = .(from_id, to_id)]
    setkey(ey, to_id, year)
    ey[lk, neighbor_val := i.val, on = .(to_id = id, year = year)]

    ey[!is.na(neighbor_val),
      .(nb_max  = max(neighbor_val),
        nb_min  = min(neighbor_val),
        nb_mean = mean(neighbor_val)),
      by = .(from_id, year)
    ]
  }))

  setnames(results, "from_id", "id")
  results
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M string pastes + named vector lookups | Single vectorized edge `data.table` build |
| **Per-variable computation** | 6.46M `lapply` iterations | One keyed `data.table` join + grouped aggregation |
| **Estimated total runtime** | 86+ hours | **~5â€“15 minutes** |
| **Peak memory** | Unpredictable (large lists) | ~4â€“8 GB (controlled) |
| **Numerical identity** | Baseline | Identical (same neighbor sets, same arithmetic) |
| **RF model** | Preserved | Preserved (untouched) |
END_RESPONSE>>>
