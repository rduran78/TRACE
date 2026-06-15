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
run_id: Anthropic_F04_T00_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The suspicion is correct: **the computational bottleneck is the spatial neighbor feature construction, not Random Forest inference.**

Here is a precise breakdown of why:

### `build_neighbor_lookup` â€” O(N) `lapply` with expensive string operations

For each of the ~6.46 million rows, the function:

1. Converts an integer ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Calls `paste()` to construct `"id_year"` key strings for every neighbor of that row.
4. Looks up each key string in `idx_lookup` (a named character vector â€” i.e., linear-hashed lookup per key).
5. Filters `NA`s and coerces to integer.

With ~6.46M rows and an average of ~4 rook neighbors per cell, this produces roughly **25.8 million `paste` + named-vector lookups**, all inside an interpreted R `lapply` loop. Named vector lookup in R is hash-based but carries significant per-call overhead when done millions of times inside `lapply`. The `paste` calls alone generate tens of millions of temporary string allocations.

**Estimated cost:** The `build_neighbor_lookup` step alone likely accounts for **60â€“70%** of the 86+ hour runtime.

### `compute_neighbor_stats` â€” O(N) `lapply` with per-row subsetting

For each of 5 variables Ã— 6.46M rows, the function:

1. Subsets `vals[idx]` (a numeric vector index â€” fast in isolation).
2. Removes `NA`s.
3. Computes `max`, `min`, `mean`.

This is called 5 times, so ~32.3 million iterations total. The `lapply` + anonymous function overhead and the `do.call(rbind, ...)` on a 6.46M-element list (each a length-3 vector) are both costly. `do.call(rbind, ...)` on a long list is notoriously slow because it incrementally allocates.

**Estimated cost:** ~25â€“35% of total runtime.

### Summary of root causes

| Cause | Location | Severity |
|---|---|---|
| Millions of `paste()` string constructions | `build_neighbor_lookup` | **Critical** |
| Named-vector lookups inside `lapply` | `build_neighbor_lookup` | **Critical** |
| Per-row `lapply` over 6.46M rows (Ã—5 vars) | `compute_neighbor_stats` | **High** |
| `do.call(rbind, list_of_6.46M_vectors)` | `compute_neighbor_stats` | **High** |
| Redundant per-variable passes over same neighbor structure | Outer loop | **Moderate** |

---

## Optimization Strategy

The key insight is: **replace row-level R loops and string-key lookups with vectorized joins and grouped aggregations using `data.table`.**

### Step 1: Replace `build_neighbor_lookup` entirely

Instead of building a list of 6.46M integer vectors (one per row), construct a **long-format edge table** that maps every `(cell-year row) â†’ (neighbor cell-year row)` using integer joins. This eliminates all `paste` and named-vector lookups.

- Create a `data.table` of the panel with a row index column.
- Expand the `nb` object into a two-column edge data.table: `(id, neighbor_id)`.
- Join on `(neighbor_id, year)` to get the row index of each neighbor in each year.

This is a single equi-join â€” `data.table` does this in seconds on 25M rows.

### Step 2: Replace `compute_neighbor_stats` with grouped `data.table` aggregation

Once we have the long-format edge table `(focal_row, neighbor_row)`, we attach the neighbor's variable values and compute `max`, `min`, `mean` as a single grouped aggregation per variable â€” fully vectorized, no `lapply`.

### Step 3: Process all 5 variables in one pass (or 5 fast passes)

Since the edge table is the same for all variables, we can either compute all 5 variables' stats in a single grouped aggregation, or loop over 5 variables with the same edge table. Both are fast.

### Expected speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup construction | ~50â€“60 hours | ~10â€“30 seconds | **~6000Ã—** |
| Neighbor stats (5 vars) | ~25â€“30 hours | ~30â€“90 seconds | **~1500Ã—** |
| **Total neighbor features** | **~80+ hours** | **~1â€“3 minutes** | **~2000Ã—** |

### What is preserved

- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of non-NA neighbor values per cell-year per variable) is identical.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED SPATIAL NEIGHBOR FEATURE CONSTRUCTION
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact numerical output (max, min, mean of non-NA neighbor values)
# Requires: data.table
# =============================================================================

library(data.table)

#' Convert an spdep nb object into a two-column data.table of directed edges.
#'
#' @param neighbors  An nb object (list of integer vectors of neighbor indices).
#' @param id_order   The vector of cell IDs corresponding to each nb index.
#' @return A data.table with columns: id (focal cell), neighbor_id (neighbor cell).
nb_to_edge_dt <- function(neighbors, id_order) {
  # Determine the number of neighbors per cell (handles 0-neighbor cells)
  n_neighbors <- vapply(neighbors, function(x) {
    # spdep nb objects use 0L to indicate no neighbors
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1))

  focal_idx <- rep(seq_along(neighbors), times = n_neighbors)

  neighbor_idx <- unlist(lapply(neighbors, function(x) {
    if (length(x) == 1L && x[1] == 0L) integer(0) else x
  }), use.names = FALSE)

  data.table(
    id          = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

#' Build all neighbor features for the specified variables and attach them
#' to the panel data.table.
#'
#' @param cell_data              A data.frame or data.table with columns: id, year,
#'                               and all columns named in neighbor_source_vars.
#' @param id_order               Integer vector of cell IDs matching the nb object.
#' @param rook_neighbors_unique  An spdep nb object (precomputed).
#' @param neighbor_source_vars   Character vector of variable names to aggregate.
#' @return cell_data with new columns: <var>_max, <var>_min, <var>_mean for each var.
build_all_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {

  # --- Convert to data.table if needed (by reference if already one) ----------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Step 1: Add a row-index column ----------------------------------------
  cell_data[, .row_idx := .I]

  # --- Step 2: Build the edge table from the nb object -----------------------
  edges <- nb_to_edge_dt(rook_neighbors_unique, id_order)
  # edges now has columns: id (focal), neighbor_id

  # --- Step 3: Create a keyed lookup from (id, year) -> row index ------------
  #     This replaces the entire build_neighbor_lookup function.
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # --- Step 4: Expand edges across all years ---------------------------------
  #     For every (focal_id, neighbor_id) pair, we need every year in the panel.
  years <- sort(unique(cell_data$year))

  # Cross join edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
  # This is the full set of (focal_id, neighbor_id, year) triples.
  edge_year <- CJ_dt(edges, years)

  # --- Step 5: Attach focal row index ----------------------------------------
  setkey(edge_year, id, year)
  edge_year[row_lookup, focal_row := i..row_idx, on = .(id, year)]

  # --- Step 6: Attach neighbor row index -------------------------------------
  setnames(row_lookup, c("id", "year", ".row_idx"),
           c("neighbor_id", "year", "neighbor_row"))
  setkey(row_lookup, neighbor_id, year)
  edge_year[row_lookup, neighbor_row := i.neighbor_row,
            on = .(neighbor_id, year)]

  # Restore row_lookup names for safety
  setnames(row_lookup, c("neighbor_id", "year", "neighbor_row"),
           c("id", "year", ".row_idx"))

  # --- Step 7: Drop edges where either focal or neighbor is missing ----------
  edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

  # --- Step 8: Compute grouped stats for each variable -----------------------
  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)

    # Attach the neighbor's value for this variable
    edge_year[, nval := cell_data[[var_name]][neighbor_row]]

    # Grouped aggregation â€” only over non-NA neighbor values
    stats <- edge_year[!is.na(nval),
                       .(var_max  = max(nval),
                         var_min  = min(nval),
                         var_mean = mean(nval)),
                       by = focal_row]

    # Initialize new columns to NA
    max_col  <- paste0(var_name, "_max")
    min_col  <- paste0(var_name, "_min")
    mean_col <- paste0(var_name, "_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Write results back by row index
    cell_data[stats$focal_row, (max_col)  := stats$var_max]
    cell_data[stats$focal_row, (min_col)  := stats$var_min]
    cell_data[stats$focal_row, (mean_col) := stats$var_mean]

    # Clean up the temporary column
    edge_year[, nval := NULL]
  }

  # --- Clean up ---------------------------------------------------------------
  cell_data[, .row_idx := NULL]

  return(cell_data)
}

#' Helper: Cross join an edge data.table with a vector of years.
#' More memory-efficient than a full CJ on three columns.
CJ_dt <- function(edges, years) {
  # edges: data.table with columns id, neighbor_id
  # years: integer/numeric vector
  n_edges <- nrow(edges)
  n_years <- length(years)

  data.table(
    id          = rep(edges$id,          times = n_years),
    neighbor_id = rep(edges$neighbor_id, times = n_years),
    year        = rep(years, each = n_edges)
  )
}

# =============================================================================
# USAGE â€” drop-in replacement for the original outer loop
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- build_all_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars
)

# cell_data now contains the same 15 new columns (5 vars Ã— 3 stats) as before.
# The trained Random Forest model can be applied directly â€” no retraining needed.
```

---

## Memory Considerations (16 GB laptop)

The largest intermediate object is `edge_year` after the cross join:

- ~1.37M edges Ã— 28 years â‰ˆ **38.5M rows** Ã— 5 columns (3 integer/numeric IDs + 2 row indices) â‰ˆ **1.5 GB**.
- `cell_data` itself at 6.46M rows Ã— ~125 columns â‰ˆ **6â€“7 GB**.
- Total peak â‰ˆ **~10â€“11 GB**, which fits within 16 GB.

If memory is tight, the cross join can be chunked by year (process 4â€“7 years at a time) with negligible speed penalty. Here is a chunked variant of Step 4 that can be substituted:

```r
# Memory-safe chunked variant (replaces Steps 4â€“8 above)
build_all_neighbor_features_chunked <- function(cell_data,
                                                 id_order,
                                                 rook_neighbors_unique,
                                                 neighbor_source_vars,
                                                 chunk_size = 7L) {
  if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)
  cell_data[, .row_idx := .I]

  edges <- nb_to_edge_dt(rook_neighbors_unique, id_order)

  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  years <- sort(unique(cell_data$year))

  # Pre-allocate result columns
  for (var_name in neighbor_source_vars) {
    cell_data[, paste0(var_name, "_max")  := NA_real_]
    cell_data[, paste0(var_name, "_min")  := NA_real_]
    cell_data[, paste0(var_name, "_mean") := NA_real_]
  }

  # Process in year chunks
  year_chunks <- split(years, ceiling(seq_along(years) / chunk_size))

  for (yc in year_chunks) {
    message("Processing years: ", paste(range(yc), collapse = "â€“"))

    edge_year <- CJ_dt(edges, yc)

    # Focal row
    setkey(edge_year, id, year)
    edge_year[row_lookup, focal_row := i..row_idx, on = .(id, year)]

    # Neighbor row
    rl2 <- copy(row_lookup)
    setnames(rl2, c("id", "year", ".row_idx"),
             c("neighbor_id", "year", "neighbor_row"))
    setkey(rl2, neighbor_id, year)
    edge_year[rl2, neighbor_row := i.neighbor_row,
              on = .(neighbor_id, year)]

    edge_year <- edge_year[!is.na(focal_row) & !is.na(neighbor_row)]

    for (var_name in neighbor_source_vars) {
      edge_year[, nval := cell_data[[var_name]][neighbor_row]]

      stats <- edge_year[!is.na(nval),
                         .(var_max  = max(nval),
                           var_min  = min(nval),
                           var_mean = mean(nval)),
                         by = focal_row]

      max_col  <- paste0(var_name, "_max")
      min_col  <- paste0(var_name, "_min")
      mean_col <- paste0(var_name, "_mean")

      set(cell_data, i = stats$focal_row, j = max_col,  value = stats$var_max)
      set(cell_data, i = stats$focal_row, j = min_col,  value = stats$var_min)
      set(cell_data, i = stats$focal_row, j = mean_col, value = stats$var_mean)

      edge_year[, nval := NULL]
    }

    rm(edge_year, rl2)
    gc()
  }

  cell_data[, .row_idx := NULL]
  return(cell_data)
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Paradigm** | Row-level `lapply` + string keys | Vectorized `data.table` joins + grouped aggregation |
| **`build_neighbor_lookup`** | ~6.46M `paste` + named-vector lookups | Eliminated; replaced by a single keyed join |
| **`compute_neighbor_stats`** | `lapply` over 6.46M rows Ã— 5 vars + `do.call(rbind, ...)` | `data.table` grouped `max`/`min`/`mean` by `focal_row` |
| **Estimated runtime** | **86+ hours** | **1â€“3 minutes** |
| **Peak memory** | Lower but irrelevant (time-dominated) | ~10â€“11 GB (fits 16 GB; chunked variant ~7â€“8 GB) |
| **Numerical output** | max, min, mean of non-NA neighbor values | **Identical** |
| **Trained RF model** | â€” | **Untouched** |
END_RESPONSE>>>
