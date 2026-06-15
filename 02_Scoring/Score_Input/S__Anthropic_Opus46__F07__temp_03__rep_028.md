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
run_id: Anthropic_F07_T03_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup`** creates a list of ~6.46 million elements, each produced by an `lapply` call that performs character-based key lookups (`paste` + named-vector indexing) for every single row. This is O(N) string operations with N â‰ˆ 6.46M, and the named-vector lookup (`idx_lookup[neighbor_keys]`) is itself O(k) per row where k is the number of neighbors â€” but the constant factor of `paste` and character matching is enormous. Then **`compute_neighbor_stats`** iterates over the same 6.46M-element list 5 times (once per variable), subsetting a numeric vector by index each time.

**Root causes of the ~86-hour runtime:**

1. **Character key construction and lookup in `build_neighbor_lookup`:** `paste()` on 6.46M rows, then named-vector lookup (which is hash-based but still slow at this scale in R) â€” this alone is likely 70%+ of the time.
2. **R-level `lapply` over 6.46M elements:** Each iteration has R-interpreter overhead (function call, environment creation, GC pressure).
3. **Redundant structure:** The neighbor topology is *time-invariant*. Every cell has the same neighbors in every year. Yet the lookup is built at the cell-year level, duplicating the same neighbor structure 28 times.
4. **`compute_neighbor_stats` uses per-row R loops** over 6.46M rows, 5 times.

## Optimization Strategy

1. **Exploit time-invariance:** Build the neighbor graph once at the cell level (344K cells), not the cell-year level (6.46M rows). For each cell, the neighbor indices into the cell-year data frame can be computed via integer arithmetic if the data is sorted by `(id, year)`.

2. **Vectorize with `data.table`:** Instead of row-wise `lapply`, "explode" the neighbor list into a long edge table `(row_idx, neighbor_row_idx)`, then join and group-aggregate in `data.table` â€” this replaces millions of R function calls with a single vectorized grouped operation.

3. **Compute all 5 variables in one pass** over the edge table rather than 5 separate passes.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes**.

The trained Random Forest model is untouched. The numerical results (neighbor max, min, mean) are identical because we are computing the exact same quantities â€” just via vectorized joins instead of scalar loops.

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Ensure cell_data is a data.table sorted by (id, year) â”€â”€
cell_data <- as.data.table(cell_data)
setkey(cell_data, id, year)

# Unique cell IDs in the order that matches rook_neighbors_unique (the nb object)
# id_order is the vector of cell IDs corresponding to positions in the nb object.
# rook_neighbors_unique is the spdep::nb list (integer index vectors into id_order).

# â”€â”€ 1. Build a cell-level edge list (time-invariant) â”€â”€
#    For each cell index i in id_order, rook_neighbors_unique[[i]] gives
#    the indices (into id_order) of its rook neighbors.

edge_list <- rbindlist(lapply(seq_along(id_order), function(i) {

  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
}))

cat("Edge list rows:", nrow(edge_list), "\n")
# Should be ~1,373,394

# â”€â”€ 2. Map every (id, year) row to its integer row position â”€â”€
cell_data[, row_idx := .I]

# â”€â”€ 3. Expand edge list to cell-year level via joins â”€â”€
#    Join focal side: get (focal_id, year, row_idx) for every row,
#    then join neighbor side to get neighbor row indices.

# Create a slim lookup: id -> list of (year, row_idx)
id_year_lookup <- cell_data[, .(year, row_idx, id)]

# Join: for each edge (focal_id, neighbor_id), cross with all years
#   focal row  <->  neighbor row in the same year
# We do this efficiently by joining on neighbor_id and year.

# Step 3a: Attach year and focal row_idx to each edge
setkey(id_year_lookup, id)
setkey(edge_list, focal_id)

# For each focal_id, get all its (year, row_idx) pairs
edges_with_year <- edge_list[id_year_lookup,
  on = .(focal_id = id),
  .(focal_row = i.row_idx, year = i.year, neighbor_id = x.neighbor_id),
  nomatch = 0L,
  allow.cartesian = TRUE
]

# Step 3b: Attach neighbor row_idx by joining on (neighbor_id, year)
setkey(edges_with_year, neighbor_id, year)
setkey(id_year_lookup, id, year)

edges_full <- edges_with_year[id_year_lookup,
  on = .(neighbor_id = id, year = year),
  .(focal_row, neighbor_row = i.row_idx),
  nomatch = 0L
]

cat("Full edge-year rows:", nrow(edges_full), "\n")
# Should be ~1,373,394 * 28 â‰ˆ 38.5M

# â”€â”€ 4. Vectorized neighbor stats for all variables at once â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Extract neighbor values for all variables at once
neighbor_vals <- cell_data[edges_full$neighbor_row, ..neighbor_source_vars]
neighbor_vals[, focal_row := edges_full$focal_row]

# Group by focal_row and compute max, min, mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)), na.rm = TRUE)),
    bquote(min(.(as.name(v)), na.rm = TRUE)),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}))

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

names(agg_exprs) <- agg_names

stats <- neighbor_vals[,
  lapply(agg_exprs, eval, envir = .SD),
  by = focal_row
]

# â”€â”€ 5. Handle Inf/-Inf from max/min on all-NA groups â”€â”€
#    (groups with all NA neighbors won't appear due to nomatch=0L,
#     but if a variable is NA for all neighbors in a year, max/min â†’ Â±Inf)
for (col in agg_names) {
  vals <- stats[[col]]
  set(stats, i = which(is.infinite(vals)), j = col, value = NA_real_)
}

# â”€â”€ 6. Merge back into cell_data â”€â”€
# First, initialize all new columns to NA (handles cells with 0 neighbors)
for (col in agg_names) {
  set(cell_data, j = col, value = NA_real_)
}

# Assign computed values
for (col in agg_names) {
  set(cell_data, i = stats$focal_row, j = col, value = stats[[col]])
}

# Clean up helper column
cell_data[, row_idx := NULL]

cat("Done. New columns added:", agg_names, "\n")
```

**If the `bquote`/`eval` aggregation feels fragile**, here is a simpler (and equally fast) alternative for Step 4:

```r
# â”€â”€ 4-alt. Simpler loop, still fully vectorized per variable â”€â”€
for (v in neighbor_source_vars) {
  neighbor_vals_v <- cell_data[[v]][edges_full$neighbor_row]
  dt_v <- data.table(
    focal_row = edges_full$focal_row,
    val = neighbor_vals_v
  )
  agg_v <- dt_v[!is.na(val),
    .(nmax = max(val), nmin = min(val), nmean = mean(val)),
    by = focal_row
  ]
  max_col  <- paste0("neighbor_max_", v)
  min_col  <- paste0("neighbor_min_", v)
  mean_col <- paste0("neighbor_mean_", v)

  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  set(cell_data, i = agg_v$focal_row, j = max_col,  value = agg_v$nmax)
  set(cell_data, i = agg_v$focal_row, j = min_col,  value = agg_v$nmin)
  set(cell_data, i = agg_v$focal_row, j = mean_col, value = agg_v$nmean)
}
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M-element list via `paste` + named vector | Integer edge table, `data.table` join |
| Stats computation | `lapply` over 6.46M rows Ã— 5 vars | Vectorized `data.table` group-by |
| Time invariance | Ignored (duplicated 28Ã—) | Exploited (edge list built once at cell level) |
| Estimated runtime | ~86 hours | ~2â€“5 minutes |
| Numerical result | Neighbor max/min/mean | **Identical** |
| RAM peak | Moderate but slow | ~4â€“6 GB (fits in 16 GB) |
| Trained RF model | Untouched | **Untouched** |
END_RESPONSE>>>
