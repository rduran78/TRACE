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
run_id: Anthropic_F04_T00_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for each neighbor.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookup (which is O(n) hash probing on character keys) repeated ~6.46M times, each with a variable number of neighbors, is extremely expensive in interpreted R. The secondary bottleneck is `compute_neighbor_stats`, which runs an `lapply` over 6.46M list elements calling `max`/`min`/`mean` on small vectors â€” slow due to R function-call overhead, but less severe than the lookup construction.

**Root causes, ranked:**

| Rank | Cause | Impact |
|------|-------|--------|
| 1 | Per-row `paste()` + character-key lookup in `build_neighbor_lookup` over 6.46M rows | Dominant (~80%+ of runtime) |
| 2 | `lapply` + per-element `max/min/mean` in `compute_neighbor_stats` over 6.46M list elements | Significant (~15%) |
| 3 | Repeated extraction for 5 variables (Ã—3 stats = 15 new columns) | Multiplicative on #2 |

## Optimization Strategy

**Core idea:** Replace all per-row string operations and list-based lookups with a single vectorized merge/join using `data.table`, and replace the per-row `lapply` stats computation with grouped `data.table` aggregation.

**Steps:**

1. **Expand the neighbor graph into an edge table** (one-time, ~1.37M directed edges Ã— 28 years â‰ˆ 38.5M rows). This is large but fits in 16 GB RAM.
2. **Join** the edge table to the panel data to pull neighbor variable values â€” fully vectorized.
3. **Group-by aggregate** (`max`, `min`, `mean`) by `(id, year)` â€” fully vectorized via `data.table`.
4. **Join** the aggregated stats back to the main data.

This eliminates all per-row `paste`, all named-vector lookups, and all `lapply` loops. Expected runtime: **minutes, not hours**.

## Working R Code

```r
library(data.table)

# â”€â”€ Step 0: Convert panel data to data.table (non-destructive) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure id and year are keyed for fast joins
setkey(cell_dt, id, year)

# â”€â”€ Step 1: Build a vectorized edge table from the nb object â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#   rook_neighbors_unique is an nb object: a list of length N_cells,
#   where element i contains integer indices of neighbors of cell i
#   in the ordering given by id_order.

# Expand to edge list: (focal_index, neighbor_index)
n_cells <- length(id_order)
focal_idx   <- rep(seq_len(n_cells), lengths(rook_neighbors_unique))
neighbor_idx <- unlist(rook_neighbors_unique)

# Map positional indices to actual cell IDs
edges <- data.table(
  focal_id    = id_order[focal_idx],
  neighbor_id = id_order[neighbor_idx]
)
rm(focal_idx, neighbor_idx)  # free memory

# â”€â”€ Step 2: Cross with years to get (focal_id, year, neighbor_id) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
years <- sort(unique(cell_dt$year))
edges_by_year <- CJ_dt_edges(edges, years)
# CJ_dt_edges: simple cross join helper
# We do this efficiently:
edges_by_year <- edges[, .(year = years), by = .(focal_id, neighbor_id)]
setkey(edges_by_year, neighbor_id, year)

# â”€â”€ Step 3: For each source variable, join, aggregate, merge back â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var in neighbor_source_vars) {

  # Subset only the columns we need from the panel for the join
  # neighbor_id + year -> value
  val_dt <- cell_dt[, .(id, year, val = get(var))]
  setkey(val_dt, id, year)

  # Join: attach neighbor's value to each edge-year row
  # edges_by_year has (focal_id, neighbor_id, year)
  # We join on neighbor_id == id, year == year
  merged <- val_dt[edges_by_year, on = .(id = neighbor_id, year = year), nomatch = NA]
  # Result columns: id (=neighbor_id), year, val, focal_id
  # Drop rows where val is NA (matches original !is.na filter)
  merged <- merged[!is.na(val)]

  # Aggregate by (focal_id, year)
  stats <- merged[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(focal_id, year)]

  # Rename columns to match original naming convention
  # Original code produces columns like: ntl_nb_max, ntl_nb_min, ntl_nb_mean
  max_col  <- paste0(var, "_nb_max")
  min_col  <- paste0(var, "_nb_min")
  mean_col <- paste0(var, "_nb_mean")
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # Merge back to main table
  setkey(stats, focal_id, year)
  cell_dt <- stats[cell_dt, on = .(focal_id = id, year = year)]

  # The join above replaces 'id' with 'focal_id'; fix:
  if ("focal_id" %in% names(cell_dt)) {
    setnames(cell_dt, "focal_id", "id")
  }

  setkey(cell_dt, id, year)

  rm(val_dt, merged, stats)
  gc()
}

# â”€â”€ Step 4: Convert back to data.frame if downstream code expects it â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)
```

**Note on the `edges_by_year` construction:** The line `edges[, .(year = years), by = .(focal_id, neighbor_id)]` creates one row per (edge Ã— year). With ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows, at ~24 bytes/row this is under 1 GB â€” well within 16 GB RAM. If memory is tight, process years in batches:

```r
# â”€â”€ Memory-conservative variant: process one year at a time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for (var in neighbor_source_vars) {

  max_col  <- paste0(var, "_nb_max")
  min_col  <- paste0(var, "_nb_min")
  mean_col <- paste0(var, "_nb_mean")

  stats_list <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]

    # Slice this year's values
    val_yr <- cell_dt[year == yr, .(id, val = get(var))]
    setkey(val_yr, id)

    # Join neighbor values onto edges
    merged <- val_yr[edges, on = .(id = neighbor_id), nomatch = NA, allow.cartesian = FALSE]
    # merged has columns: id (neighbor_id), val, focal_id
    merged <- merged[!is.na(val)]

    stats_list[[yi]] <- merged[, .(
      year    = yr,
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(focal_id)]
  }

  stats <- rbindlist(stats_list)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))
  setkey(stats, focal_id, year)

  cell_dt <- stats[cell_dt, on = .(focal_id = id, year = year)]
  if ("focal_id" %in% names(cell_dt)) setnames(cell_dt, "focal_id", "id")
  setkey(cell_dt, id, year)

  rm(stats_list, stats)
  gc()
}

cell_data <- as.data.frame(cell_dt)
```

## Summary

| Aspect | Original | Optimized |
|--------|----------|-----------|
| Lookup construction | 6.46M `paste` + character hash lookups | One-time vectorized edge table |
| Stats computation | 6.46M `lapply` calls Ã— 5 vars | `data.table` grouped aggregation Ã— 5 vars |
| Estimated runtime | 86+ hours | ~5â€“15 minutes |
| RAM peak | Moderate (large list of 6.46M elements) | ~1â€“2 GB for edge table |
| Numerical results | â€” | Identical (`max`, `min`, `mean` preserved) |
| Trained RF model | â€” | Untouched |
END_RESPONSE>>>
