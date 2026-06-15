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
run_id: Anthropic_F04_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The computational bottleneck is **not** Random Forest inference â€” it is the two spatial neighbor feature construction functions. Here is why:

### `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations

For each of the **~6.46 million rows**, the function:

1. Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. **Pastes** each neighbor ID with the current row's year to create string keys (`paste(..., sep="_")`).
4. Looks up each key in a **named character vector** (`idx_lookup`), which in R is an **O(n) linear scan** per lookup in the worst case (R named vectors are not hash tables).

With ~6.46M rows and an average of ~4 rook neighbors per cell, this creates **~25.8 million string keys** and performs **~25.8 million named-vector lookups**. Named vector lookup in R is slow â€” it uses linear hashing that degrades badly at millions of entries. The `idx_lookup` vector itself has 6.46M entries, making each lookup expensive. This alone can take tens of hours.

### `compute_neighbor_stats` â€” Repeated per variable

For each of the 5 source variables, `compute_neighbor_stats` iterates over all 6.46M rows again via `lapply`, subsetting and computing `max`, `min`, `mean`. The `lapply` over 6.46M elements with anonymous R functions is slow due to R's interpreter overhead. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also very expensive (repeated memory allocation).

### Summary of bottleneck sources

| Source | Problem | Severity |
|---|---|---|
| `paste()` key construction | ~25.8M string allocations inside `lapply` | High |
| Named vector lookup (`idx_lookup[...]`) | Pseudo-hash on 6.46M-entry vector, called 25.8M times | **Critical** |
| `lapply` over 6.46M rows (twice: build + compute) | R interpreter loop overhead | High |
| `do.call(rbind, list_of_6.46M)` | Repeated reallocation | Moderate |
| Ã—5 variables | Multiplies `compute_neighbor_stats` cost | Moderate |

**Estimated breakdown of the ~86 hours:**
- `build_neighbor_lookup`: ~50â€“60 hours (string key creation + named vector lookup)
- `compute_neighbor_stats` (Ã—5 vars): ~20â€“30 hours
- Random Forest prediction: likely < 1 hour

---

## Optimization Strategy

### Principle: Replace string-key lookups with integer-arithmetic indexing; vectorize everything.

**Key insight:** The data has a regular panel structure (344,208 cells Ã— 28 years). If we sort the data by `(id, year)` â€” or equivalently by `(year, id)` â€” we can compute the row index of any `(cell_id, year)` pair with **pure integer arithmetic**, completely eliminating string construction and hash lookups.

### Specific changes:

1. **Sort data by `id` then `year`** and create a direct integer mapping from `cell_id â†’ position` (1-based rank). Then row index = `(position - 1) * 28 + (year - 1991)`. This is O(1) per lookup with zero string allocation.

2. **Replace `build_neighbor_lookup`** with a vectorized construction that expands the `nb` object into an edge list (cell-index pairs), then uses integer arithmetic to compute all neighbor row indices for all years simultaneously â€” fully vectorized, no `lapply` over 6.46M rows.

3. **Replace `compute_neighbor_stats`** with a vectorized grouped aggregation using `data.table`, computing `max`, `min`, `mean` of neighbor values in one pass per variable using the edge list.

4. **Process all 5 variables in one pass** over the edge list rather than 5 separate passes.

5. **Eliminate `do.call(rbind, ...)`** entirely by pre-allocating result matrices or using `data.table` aggregation which returns a data.table directly.

**Expected speedup:** From ~86 hours to **~2â€“5 minutes**.

---

## Working R Code

```r
library(data.table)

optimize_neighbor_features <- function(cell_data,
                                       id_order,
                                       rook_neighbors_unique,
                                       neighbor_source_vars,
                                       year_range = 1992:2019) {
  # ------------------------------------------------------------------
  # 0. Convert to data.table if needed; preserve original row order
  # ------------------------------------------------------------------
  dt <- as.data.table(cell_data)
  dt[, .original_row_order := .I]

  n_cells <- length(id_order)
  n_years <- length(year_range)
  min_year <- min(year_range)

  # ------------------------------------------------------------------
  # 1. Create integer mapping: cell id -> position index (1-based)
  #    id_order is assumed to define the canonical ordering matching
  #    the nb object (i.e., rook_neighbors_unique[[k]] gives neighbor
  #    positions for the k-th element of id_order).
  # ------------------------------------------------------------------
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))

  # Assign each row its cell position index
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # ------------------------------------------------------------------
  # 2. Sort by (cell_pos, year) so that row index is deterministic:
  #    row_index = (cell_pos - 1) * n_years + (year - min_year + 1)
  #    This is the KEY insight that eliminates all string lookups.
  # ------------------------------------------------------------------
  setorder(dt, cell_pos, year)
  dt[, sorted_row_idx := .I]

  # Verify the deterministic mapping holds
  dt[, expected_idx := (cell_pos - 1L) * n_years + (year - min_year + 1L)]
  stopifnot(all(dt$sorted_row_idx == dt$expected_idx))

  # ------------------------------------------------------------------
  # 3. Build edge list from nb object (vectorized)
  #    Each entry: (source_cell_pos, neighbor_cell_pos)
  # ------------------------------------------------------------------
  # Expand nb list into an edge list
  n_neighbors <- lengths(rook_neighbors_unique)  # integer vector, length = n_cells
  edge_source <- rep.int(seq_len(n_cells), n_neighbors)
  edge_target <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-entries (spdep::nb uses 0 to indicate no neighbors)
  valid <- edge_target > 0L
  edge_source <- edge_source[valid]
  edge_target <- edge_target[valid]

  n_edges <- length(edge_source)
  cat(sprintf("Edge list: %d directed neighbor relationships\n", n_edges))

  # ------------------------------------------------------------------
  # 4. Expand edges across all years (vectorized integer arithmetic)
  #    For each edge (s, t) and each year y:
  #      source_row = (s - 1) * n_years + (y - min_year + 1)
  #      target_row = (t - 1) * n_years + (y - min_year + 1)
  # ------------------------------------------------------------------
  # year offsets: 1, 2, ..., n_years
  year_offsets <- seq_len(n_years)

  # Repeat each edge n_years times; repeat year_offsets n_edges times
  src_base <- rep.int((edge_source - 1L) * n_years, n_years)
  tgt_base <- rep.int((edge_target - 1L) * n_years, n_years)
  yr_off   <- rep(year_offsets, times = n_edges)
  # Reorder so that edges cycle within years
  # Actually: rep.int repeats each element n_years times sequentially,
  # but rep repeats the whole vector n_edges times. We need alignment.
  # Let's use the outer-product approach:

  # More efficient: use rep with each vs times
  src_base <- rep(((edge_source - 1L) * n_years), each = n_years)
  tgt_base <- rep(((edge_target - 1L) * n_years), each = n_years)
  yr_off   <- rep.int(year_offsets, times = n_edges)

  all_src_rows <- src_base + yr_off   # source row indices in sorted dt
  all_tgt_rows <- tgt_base + yr_off   # neighbor row indices in sorted dt

  cat(sprintf("Expanded edge-year pairs: %s\n", format(length(all_src_rows), big.mark = ",")))

  # ------------------------------------------------------------------
  # 5. Validate: ensure all indices are within bounds
  # ------------------------------------------------------------------
  max_row <- nrow(dt)
  valid_mask <- all_src_rows >= 1L & all_src_rows <= max_row &
                all_tgt_rows >= 1L & all_tgt_rows <= max_row
  if (!all(valid_mask)) {
    cat(sprintf("Dropping %d out-of-bounds edge-year pairs (cells with missing years)\n",
                sum(!valid_mask)))
    all_src_rows <- all_src_rows[valid_mask]
    all_tgt_rows <- all_tgt_rows[valid_mask]
  }

  # ------------------------------------------------------------------
  # 6. For each variable, compute neighbor max/min/mean via data.table
  #    grouped aggregation on the edge list.
  # ------------------------------------------------------------------
  # Build a data.table of (source_row, neighbor_value) per variable
  # and aggregate by source_row.

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Extract neighbor values using integer indexing (vectorized)
    neighbor_vals <- dt[[var_name]][all_tgt_rows]

    # Build edge table
    edge_dt <- data.table(
      src_row = all_src_rows,
      nval    = neighbor_vals
    )

    # Remove NAs in neighbor values
    edge_dt <- edge_dt[!is.na(nval)]

    # Grouped aggregation â€” this is extremely fast in data.table
    agg <- edge_dt[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = src_row]

    # Initialize result columns with NA
    max_col  <- paste0("max_neighbor_",  var_name)
    min_col  <- paste0("min_neighbor_",  var_name)
    mean_col <- paste0("mean_neighbor_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign aggregated values back by integer index
    dt[agg$src_row, (max_col)  := agg$nb_max]
    dt[agg$src_row, (min_col)  := agg$nb_min]
    dt[agg$src_row, (mean_col) := agg$nb_mean]

    # Free memory
    rm(edge_dt, agg, neighbor_vals)
  }

  # ------------------------------------------------------------------
  # 7. Restore original row order and return as data.frame
  # ------------------------------------------------------------------
  setorder(dt, .original_row_order)
  dt[, c(".original_row_order", "sorted_row_idx", "expected_idx", "cell_pos") := NULL]

  cat("Done. Neighbor features computed.\n")
  return(as.data.frame(dt))
}
```

### Drop-in replacement for the outer loop:

```r
# ---- BEFORE (original, ~86 hours) ----
# neighbor_lookup <- build_neighbor_lookup(cell_data, id_order, rook_neighbors_unique)
# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# for (var_name in neighbor_source_vars) {
#   cell_data <- compute_and_add_neighbor_features(cell_data, var_name, neighbor_lookup)
# }

# ---- AFTER (optimized, ~2-5 minutes) ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data              = cell_data,
  id_order               = id_order,
  rook_neighbors_unique  = rook_neighbors_unique,
  neighbor_source_vars   = neighbor_source_vars,
  year_range             = 1992:2019
)

# The trained Random Forest model is unchanged â€” proceed directly to prediction.
# predictions <- predict(rf_model, newdata = cell_data[, predictor_vars])
```

### Handling incomplete panels (if some cells lack certain years):

If the panel is **unbalanced** (not every cell has all 28 years), the deterministic index formula breaks. Here is a robust fallback that still avoids string keys â€” it uses `data.table` integer joins instead:

```r
optimize_neighbor_features_unbalanced <- function(cell_data,
                                                   id_order,
                                                   rook_neighbors_unique,
                                                   neighbor_source_vars) {
  dt <- as.data.table(cell_data)
  dt[, .row_id := .I]

  n_cells <- length(id_order)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_pos := id_to_pos[as.character(id)]]

  # Build row lookup: keyed data.table for (cell_pos, year) -> row_id
  row_lookup <- dt[, .(cell_pos, year, .row_id)]
  setkey(row_lookup, cell_pos, year)

  # Build edge list from nb object
  n_neighbors <- lengths(rook_neighbors_unique)
  edge_source_pos <- rep.int(seq_len(n_cells), n_neighbors)
  edge_target_pos <- unlist(rook_neighbors_unique, use.names = FALSE)
  valid <- edge_target_pos > 0L
  edge_source_pos <- edge_source_pos[valid]
  edge_target_pos <- edge_target_pos[valid]

  # Get unique years per source cell position, then expand
  cell_years <- dt[, .(year = unique(year)), by = cell_pos]
  setkey(cell_years, cell_pos)

  # For each edge, get the years of the source cell and expand
  edge_dt <- data.table(src_pos = edge_source_pos, tgt_pos = edge_target_pos)
  edge_dt <- merge(edge_dt, cell_years, by.x = "src_pos", by.y = "cell_pos",
                   allow.cartesian = TRUE)

  # Join to get source row id
  edge_dt[row_lookup, src_row := i..row_id, on = .(src_pos = cell_pos, year)]
  # Join to get target row id
  edge_dt[row_lookup, tgt_row := i..row_id, on = .(tgt_pos = cell_pos, year)]

  # Drop edges where either side is missing
  edge_dt <- edge_dt[!is.na(src_row) & !is.na(tgt_row)]

  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    edge_dt[, nval := dt[[var_name]][tgt_row]]
    agg <- edge_dt[!is.na(nval), .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = src_row]

    max_col  <- paste0("max_neighbor_",  var_name)
    min_col  <- paste0("min_neighbor_",  var_name)
    mean_col <- paste0("mean_neighbor_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]
    dt[agg$src_row, (max_col)  := agg$nb_max]
    dt[agg$src_row, (min_col)  := agg$nb_min]
    dt[agg$src_row, (mean_col) := agg$nb_mean]

    rm(agg)
  }

  setorder(dt, .row_id)
  dt[, c(".row_id", "cell_pos") := NULL]
  return(as.data.frame(dt))
}
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Key construction | `paste()` â†’ 25.8M strings | Integer arithmetic, zero strings |
| Row lookup | Named vector (pseudo-hash, 6.46M entries) | Direct integer index or `data.table` keyed join |
| Iteration | `lapply` over 6.46M rows (R interpreter) | Fully vectorized `data.table` grouped aggregation |
| Result assembly | `do.call(rbind, 6.46M-element list)` | Pre-allocated columns, indexed assignment |
| Per-variable pass | Separate `lapply` Ã—5 | Reuses edge list; only value extraction + `data.table` group-by Ã—5 |
| Estimated runtime | **~86 hours** | **~2â€“5 minutes** |
| RF model | Unchanged | Unchanged |
| Numerical output | max/min/mean per variable | Identical max/min/mean per variable |
END_RESPONSE>>>
