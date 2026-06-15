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
run_id: Anthropic_F07_T07_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows in a pure-R `lapply`, performing per-row character coercion, `paste`, and named-vector lookups. This creates ~6.46 million small character vectors, each hitting an O(n) hash-table lookup against a 6.46-million-entry named vector (`idx_lookup`). The result is **O(NÂ²)-like wall-clock behavior** due to repeated named-vector searches and massive memory churn from millions of tiny allocations.

`compute_neighbor_stats` then loops over 6.46 million entries again, subsetting a numeric vector each timeâ€”less severe but still slow in pure R.

**Root causes:**

1. **Per-row string construction and named-vector lookup in `build_neighbor_lookup`**: `paste()` and `idx_lookup[neighbor_keys]` inside a 6.46M-iteration `lapply` is catastrophically slow. Named vector lookup in R is O(n) in the worst case for large vectors.
2. **List-of-vectors representation**: Storing 6.46M small integer vectors in a list causes massive memory overhead and GC pressure.
3. **Sequential per-variable recomputation**: `compute_neighbor_stats` is called 5 times, each time looping over 6.46M rows.

## Optimization Strategy

**Replace the entire row-level lookup with vectorized operations using `data.table` joins.**

The key insight: the neighbor graph is **static across years**. We can express the full set of (cell, neighbor, year) relationships as a single join table and compute grouped statistics in one vectorized pass per variableâ€”or all variables at once.

**Steps:**

1. **Expand the `nb` object into an edge list** of (cell_id, neighbor_id) pairs â€” ~1.37M rows.
2. **Cross-join with years** to get ~1.37M Ã— 28 â‰ˆ 38.5M (cell_id, neighbor_id, year) rows (but since each edge is per-cell, it's already directed; we just join on year).
3. **Join** this edge table to the data to retrieve neighbor values.
4. **Group by (cell_id, year)** and compute max, min, mean in one pass.
5. **Join results back** to the main data.

This eliminates all per-row R loops and replaces them with `data.table` indexed joins and grouped aggregations. Expected runtime: **minutes, not days**.

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert cell_data to data.table (if not already) â”€â”€
setDT(cell_data)

# â”€â”€ Step 1: Build directed edge list from the nb object â”€â”€
# rook_neighbors_unique is an nb object (list of integer index vectors)
# id_order is the vector of cell IDs corresponding to each nb index

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(id = id_order[i], neighbor_id = id_order[nb_idx])
}))

cat("Edge list rows:", nrow(edges), "\n")
# Should be ~1,373,394

# â”€â”€ Step 2: Define source variables â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# â”€â”€ Step 3: Build a slim lookup table: (id, year, var1, var2, ...) â”€â”€
# This is what we join neighbor values FROM
value_cols <- intersect(neighbor_source_vars, names(cell_data))
lookup_dt <- cell_data[, c("id", "year", value_cols), with = FALSE]

# Key for fast join
setnames(lookup_dt, "id", "neighbor_id")
setkeyv(lookup_dt, c("neighbor_id", "year"))

# â”€â”€ Step 4: Join edges Ã— years to get neighbor values â”€â”€
# Start from edges, cross with the year dimension via join to lookup_dt
# We need (id, year) -> list of neighbor values
# Build: edges joined to lookup_dt on (neighbor_id, year)
# But we need to know which years exist. We get that from the focal cell.

# Approach: for each (id, year) in cell_data, find neighbors via edges,
# then look up their values.

# First, create the (id, year, neighbor_id) table:
# Join cell_data's (id, year) to edges on id
focal_keys <- unique(cell_data[, .(id, year)])
setkeyv(focal_keys, "id")
setkeyv(edges, "id")

# This gives us (id, year, neighbor_id) â€” one row per neighbor per cell-year
edge_year <- edges[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = 0L]
# edge_year has columns: id, neighbor_id, year

cat("Edge-year rows:", nrow(edge_year), "\n")
# ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (fits in 16 GB easily)

# â”€â”€ Step 5: Join to get neighbor variable values â”€â”€
setkeyv(edge_year, c("neighbor_id", "year"))
edge_year <- lookup_dt[edge_year, on = c("neighbor_id", "year"), nomatch = NA]

# edge_year now has: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2, id

# â”€â”€ Step 6: Compute grouped statistics per (id, year) â”€â”€
stat_exprs <- unlist(lapply(value_cols, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

stat_names <- unlist(lapply(value_cols, function(v) {
  paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
}))

# Build the aggregation call dynamically
agg_list <- setNames(stat_exprs, stat_names)

neighbor_stats <- edge_year[,
  lapply(value_cols, function(v) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      list(NA_real_, NA_real_, NA_real_)
    } else {
      list(max(vals), min(vals), mean(vals))
    }
  }) |> unlist() |> as.list() |> setNames(stat_names),
  by = .(id, year)
]

# â”€â”€ Step 7: Merge back into cell_data â”€â”€
# Remove old neighbor columns if they exist (to avoid duplication)
old_cols <- intersect(stat_names, names(cell_data))
if (length(old_cols) > 0) {
  cell_data[, (old_cols) := NULL]
}

setkeyv(cell_data, c("id", "year"))
setkeyv(neighbor_stats, c("id", "year"))
cell_data <- neighbor_stats[cell_data, on = c("id", "year")]

# Cells with no neighbors will have NA for neighbor stats (correct behavior,
# matching the original code).

cat("Done. cell_data now has", ncol(cell_data), "columns and", nrow(cell_data), "rows.\n")
```

### Faster alternative for Step 6 (simpler, more robust)

If the dynamic aggregation above feels fragile, here is a cleaner per-variable loop that is still fully vectorized:

```r
# â”€â”€ Step 6 (alternative): per-variable vectorized aggregation â”€â”€
setkeyv(edge_year, c("id", "year"))

all_stats <- list()
for (v in value_cols) {
  max_name  <- paste0(v, "_neighbor_max")
  min_name  <- paste0(v, "_neighbor_min")
  mean_name <- paste0(v, "_neighbor_mean")

  # Suppress -Inf/Inf warnings from max/min on empty groups
  stats_v <- edge_year[!is.na(get(v)),
    .(
      V_max  = max(get(v)),
      V_min  = min(get(v)),
      V_mean = mean(get(v))
    ),
    by = .(id, year)
  ]
  setnames(stats_v, c("V_max", "V_min", "V_mean"),
                     c(max_name, min_name, mean_name))
  all_stats[[v]] <- stats_v
}

# Merge all stat tables together
neighbor_stats <- Reduce(function(a, b) merge(a, b, by = c("id", "year"), all = TRUE), all_stats)

# â”€â”€ Step 7 (same as above) â”€â”€
old_cols <- intersect(names(neighbor_stats), names(cell_data))
old_cols <- setdiff(old_cols, c("id", "year"))
if (length(old_cols) > 0) cell_data[, (old_cols) := NULL]

cell_data <- merge(cell_data, neighbor_stats, by = c("id", "year"), all.x = TRUE)
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-loop iterations with string ops | Eliminated; replaced by `data.table` edge-list join |
| `compute_neighbor_stats` | 6.46M R-loop iterations Ã— 5 vars | Vectorized `data.table` grouped aggregation |
| Estimated runtime | 86+ hours | **5â€“15 minutes** |
| Peak memory | Excessive (millions of small list elements) | ~2â€“4 GB for edge_year table |
| Numerical results | âœ… | âœ… Identical (same max/min/mean, same NA handling) |
| Trained RF model | âœ… Untouched | âœ… Untouched |

The trained Random Forest model is never touchedâ€”only the feature-engineering pipeline is replaced. The computed neighbor max, min, and mean values are numerically identical to the originals because the same arithmetic operations (R's `max`, `min`, `mean` with `na.rm = TRUE`) are applied to exactly the same sets of neighbor values, just via vectorized joins instead of per-row loops.
END_RESPONSE>>>
