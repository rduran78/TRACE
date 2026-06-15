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
run_id: Anthropic_F02_T10_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F02
prompt_family_slug: CODE_ONLY_NO_HINT
family_label: Code only, no hint
family_group: Code-visible neutral
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F02_Anthropic_temp_10_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates ~6.46 million R list elements, each produced by an anonymous function inside `lapply` that performs character coercion, string pasting, and named-vector lookups per row.** This is catastrophically slow for several reasons:

1. **Per-row string operations.** `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times. String hashing over a 6.46M-entry named vector is O(1) amortized per lookup but the constant factor is large, and the sheer number of calls dominates.

2. **Redundant work across years.** The neighbor *structure* is purely spatial (rook contiguity between cells), yet the lookup is rebuilt by iterating over every cell-year row. For 344,208 cells Ã— 28 years, the same neighbor set is re-resolved 28 times per cell.

3. **`compute_neighbor_stats` uses an R-level `lapply` over 6.46M elements**, calling `max`, `min`, `mean` individually. This prevents any vectorised or compiled-code speedup.

4. **Memory:** Storing 6.46M list elements, each a variable-length integer vector, plus intermediate character vectors, easily consumes multiple gigabytes and triggers repeated garbage collection.

---

## Optimization Strategy

### Core Idea: Flatten to a vectorised, integer-indexed join; exploit the year-invariance of the neighbor graph.

| Problem | Solution |
|---|---|
| Per-row string pasting & named-vector lookup | Replace with integer join via `data.table` keyed merge |
| Neighbor structure recomputed for every year | Build a spatial-only edge list once (cell â†’ neighbor_cell), then join on `(neighbor_cell, year)` |
| R-level `lapply` for summary stats | Use `data.table` grouped aggregation (`[, .(max, min, mean), by = ...]`) â€” internally C-compiled |
| Memory pressure from list-of-vectors | Edge list is a simple two-column (or three-column after year expansion) integer table â€” far more compact |

### Steps

1. **Convert `spdep::nb` to an edge-list `data.table`** with columns `(id, neighbor_id)` â€” done once, ~1.37M rows.
2. **Cross-join with years** â†’ ~1.37M Ã— 28 â‰ˆ 38.5M rows `(id, year, neighbor_id)`.
3. **Keyed merge** the neighbor values in: join on `(neighbor_id, year)` to pick up `ntl`, `ec`, etc.
4. **Grouped aggregation** `[, .(max_v, min_v, mean_v), by = .(id, year)]` â€” all in compiled `data.table` C code.
5. **Left-join** the aggregated features back onto the main data.

This replaces the ~86-hour R loop with operations that should complete in **minutes** on 16 GB RAM.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Ensure main data is a data.table, keyed for fast joins
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)          # copy once
setkey(cell_dt, id, year)

# ---------------------------------------------------------------
# 1.  Convert spdep::nb â†’ spatial edge list  (done ONCE)
#     rook_neighbors_unique is a list of integer index vectors;
#     id_order maps those indices to actual cell ids.
# ---------------------------------------------------------------
edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(data.table(id = integer(0), neighbor_id = integer(0)))
  }
  data.table(
    id          = id_order[i],
    neighbor_id = id_order[nb_idx]
  )
}))
# ~ 1.37 M rows, two integer columns â€” very compact

# ---------------------------------------------------------------
# 2.  Expand edge list across all years
#     Instead of a full cross join (which would be large),
#     we merge via the main data's existing (id, year) pairs.
# ---------------------------------------------------------------
# Get the unique years
all_years <- sort(unique(cell_dt$year))

# Cross-join edge list with years â†’ ~38.5 M rows
edges_by_year <- edge_list[, .(year = all_years), by = .(id, neighbor_id)]
# columns: id, neighbor_id, year

# ---------------------------------------------------------------
# 3.  For each neighbor source variable, compute stats and merge
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Key the edges for the upcoming join on (neighbor_id, year)
setkey(edges_by_year, neighbor_id, year)

# Build a slim lookup table with only the columns we need
lookup_cols <- c("id", "year", neighbor_source_vars)
value_dt    <- cell_dt[, ..lookup_cols]
setnames(value_dt, "id", "neighbor_id")
setkey(value_dt, neighbor_id, year)

# Join neighbor values onto the edge table (all vars at once)
edges_vals <- value_dt[edges_by_year, on = .(neighbor_id, year), nomatch = NA]
# columns: neighbor_id, year, ntl, ec, â€¦, id  (the focal cell)

# Now compute grouped stats per (id, year) for every variable
for (var in neighbor_source_vars) {
  
  cat("Computing neighbor stats for:", var, "\n")
  
  # Aggregate â€” data.table executes this in compiled C
  agg <- edges_vals[
    !is.na(get(var)),
    .(
      nbr_max  = max(get(var)),
      nbr_min  = min(get(var)),
      nbr_mean = mean(get(var))
    ),
    by = .(id, year)
  ]
  
  # Rename columns to match the original pipeline's naming convention
  new_names <- paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  setnames(agg, c("nbr_max", "nbr_min", "nbr_mean"), new_names)
  
  # Keyed left-join back onto the main table
  setkey(agg, id, year)
  cell_dt <- agg[cell_dt, on = .(id, year)]
  setkey(cell_dt, id, year)
}

# ---------------------------------------------------------------
# 4.  (Optional) Convert back to data.frame if downstream code expects it
# ---------------------------------------------------------------
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------
# 5.  The trained Random Forest model is untouched.
#     Proceed with predict() as before:
# ---------------------------------------------------------------
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Memory Notes for a 16 GB Laptop

| Object | Approximate Size |
|---|---|
| `cell_dt` (6.46M Ã— 110 cols, mostly numeric) | ~5.4 GB |
| `edge_list` (1.37M Ã— 2 int cols) | ~11 MB |
| `edges_by_year` (38.5M Ã— 3 cols) | ~460 MB |
| `edges_vals` (38.5M Ã— 7+ cols) | ~2.1 GB |
| Each `agg` intermediate | ~150 MB |
| **Peak total** | **~9â€“10 GB** âœ” fits in 16 GB |

If memory is still tight, process one variable at a time and drop intermediate columns:

```r
# Memory-saving variant: join only one variable at a time
for (var in neighbor_source_vars) {
  
  slim <- cell_dt[, .(neighbor_id = id, year, val = get(var))]
  setkey(slim, neighbor_id, year)
  
  tmp <- slim[edges_by_year, on = .(neighbor_id, year), nomatch = NA]
  
  agg <- tmp[!is.na(val), .(
    nbr_max  = max(val),
    nbr_min  = min(val),
    nbr_mean = mean(val)
  ), by = .(id, year)]
  
  new_names <- paste0(var, c("_nbr_max", "_nbr_min", "_nbr_mean"))
  setnames(agg, c("nbr_max", "nbr_min", "nbr_mean"), new_names)
  setkey(agg, id, year)
  
  cell_dt <- agg[cell_dt, on = .(id, year)]
  setkey(cell_dt, id, year)
  
  rm(slim, tmp, agg); gc()
}
```

---

## Summary

| Aspect | Original | Optimised |
|---|---|---|
| Lookup construction | 6.46M `lapply` calls with string ops | One `rbindlist` + keyed `data.table` join |
| Stat computation | 6.46M R-level `lapply` per variable | Compiled `data.table` grouped aggregation |
| Estimated runtime | 86+ hours | **~5â€“15 minutes** |
| Peak RAM | Unbounded (GC thrashing) | ~9â€“10 GB (fits 16 GB) |
| RF model | Unchanged âœ” | Unchanged âœ” |
| Numerical results | Identical `max`, `min`, `mean` âœ” | Identical âœ” |
END_RESPONSE>>>
