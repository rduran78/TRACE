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
run_id: Anthropic_F04_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character coercion and named-vector lookup.
2. Retrieves neighbor cell IDs.
3. Constructs paste-based keys (`"id_year"`) for each neighbor.
4. Matches those keys against a named vector (`idx_lookup`) of length ~6.46M.

Named-vector lookup in R is O(n) per query in the worst case (hash collisions aside, the overhead of repeated `as.character`, `paste`, and named-vector subsetting across 6.46M rows Ã— ~4 neighbors each â‰ˆ 25+ million string operations). The `compute_neighbor_stats` function is lighter but still uses an R-level `lapply` over 6.46M elements, calling `max`/`min`/`mean` individually.

**Root causes (ranked):**
1. **String key construction and lookup** in `build_neighbor_lookup`: millions of `paste()` and named-vector lookups.
2. **Row-level R `lapply`** loops over 6.46M elements (interpreter overhead).
3. **`compute_neighbor_stats`** repeats an R-level loop per variable (Ã—5 variables).

## Optimization Strategy

**Core idea:** Replace all string-key operations with integer-arithmetic indexing, and replace row-level `lapply` with vectorized/`data.table` operations.

**Key observations:**
- The data is a balanced panel (344,208 cells Ã— 28 years). If sorted by `(id, year)`, every cell's row for year `y` is at a deterministic offset. A neighbor in the same year is simply at a fixed row offset â€” no string lookup needed.
- `compute_neighbor_stats` can be fully vectorized using `data.table` with a long-form neighbor-edge table and grouped aggregation.

**Steps:**
1. Sort data by `(id, year)` and assign integer cell indices and year indices.
2. Build a flat edge table (integer pairs: `from_row â†’ to_row`) using arithmetic, not string keys.
3. Compute all neighbor stats via vectorized `data.table` grouped aggregation on the edge table.

This eliminates all `paste`, all named-vector lookups, and all R-level row loops.

## Optimized R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                       neighbor_source_vars) {

  # --- Step 0: Convert to data.table, ensure sorted by (id, year) ---
  dt <- as.data.table(cell_data)
  dt[, orig_row_order := .I]  # preserve original row order for later

  # Create integer cell index matching id_order
  id_map <- data.table(id = id_order, cell_idx = seq_along(id_order))
  dt <- merge(dt, id_map, by = "id", all.x = TRUE)

  # Sort by cell_idx, year for deterministic row positioning
  setorder(dt, cell_idx, year)
  dt[, sorted_row := .I]

  # --- Step 1: Build year-to-year_idx mapping ---
  years <- sort(unique(dt$year))
  n_years <- length(years)
  n_cells <- length(id_order)
  year_map <- data.table(year = years, year_idx = seq_along(years))
  dt <- merge(dt, year_map, by = "year", all.x = TRUE)
  setorder(dt, cell_idx, year_idx)
  # Now row for (cell_idx=c, year_idx=y) is at position: (c - 1) * n_years + y

  # Verify balanced panel assumption
  stopifnot(nrow(dt) == n_cells * n_years)

  # --- Step 2: Build flat edge table using integer arithmetic ---
  # rook_neighbors_unique is an nb object: list of length n_cells,
  # each element is integer vector of neighbor cell indices into id_order.

  # Build edges: from_cell_idx -> to_cell_idx
  from_cell <- rep(seq_len(n_cells), times = lengths(rook_neighbors_unique))
  to_cell   <- unlist(rook_neighbors_unique)

  # Expand across all years: for each year_idx, compute from_row and to_row
  # Row formula: (cell_idx - 1) * n_years + year_idx
  n_edges_per_year <- length(from_cell)

  edges <- data.table(
    from_cell = rep(from_cell, times = n_years),
    to_cell   = rep(to_cell,   times = n_years),
    year_idx  = rep(seq_len(n_years), each = n_edges_per_year)
  )

  edges[, from_row := (from_cell - 1L) * n_years + year_idx]
  edges[, to_row   := (to_cell   - 1L) * n_years + year_idx]

  # Drop the cell/year columns to save memory
  edges[, c("from_cell", "to_cell", "year_idx") := NULL]

  # --- Step 3: Vectorized neighbor stats per variable ---
  for (var_name in neighbor_source_vars) {
    vals <- dt[[var_name]]  # aligned to sorted_row order

    # Fetch neighbor values via integer indexing (fully vectorized)
    edges[, nval := vals[to_row]]

    # Remove NAs before aggregation
    valid_edges <- edges[!is.na(nval)]

    # Grouped aggregation: max, min, mean per from_row
    agg <- valid_edges[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = from_row]

    # Initialize columns with NA
    col_max  <- paste0("neighbor_max_",  var_name)
    col_min  <- paste0("neighbor_min_",  var_name)
    col_mean <- paste0("neighbor_mean_", var_name)

    dt[, (col_max)  := NA_real_]
    dt[, (col_min)  := NA_real_]
    dt[, (col_mean) := NA_real_]

    # Assign aggregated values back
    dt[agg$from_row, (col_max)  := agg$nb_max]
    dt[agg$from_row, (col_min)  := agg$nb_min]
    dt[agg$from_row, (col_mean) := agg$nb_mean]
  }

  # --- Step 4: Restore original row order, clean up helper columns ---
  setorder(dt, orig_row_order)
  dt[, c("orig_row_order", "sorted_row", "cell_idx", "year_idx") := NULL]

  # Return as data.frame if the original was one (for RF predict compatibility)
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}

# --- Usage (drop-in replacement for the original outer loop) ---
cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# The trained Random Forest model is unchanged; proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Same max/min/mean of same non-NA neighbor values per cell-year. Identical estimand. |
| **Trained RF model** | Untouched. Only the feature-construction preprocessing is changed. Column names match originals. |
| **Memory (16 GB laptop)** | The edge table is ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows of two integers (~590 MB). The data table is ~6.46M rows. Fits comfortably in 16 GB. |
| **Speed** | Eliminates all `paste`/string operations and R-level row loops. `data.table` grouped aggregation on 38.5M rows typically completes in seconds-to-low-minutes per variable. **Expected total: ~5â€“15 minutes** vs. 86+ hours. |

### Memory optimization note

If 16 GB is tight, process years in batches (e.g., 7 years at a time) by subsetting the edge expansion in Step 2. This quarters peak memory at the cost of a trivial outer loop.
END_RESPONSE>>>
