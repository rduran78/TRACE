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
run_id: Anthropic_F04_T03_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference â€” it is the two spatial neighbor feature construction steps: `build_neighbor_lookup` and `compute_neighbor_stats`. Here is why:

### `build_neighbor_lookup` â€” O(N Ã— k) with massive overhead

For each of the ~6.46 million rows, the function:

1. Converts `data$id[i]` to character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs `paste(neighbor_cell_ids, data$year[i], sep = "_")` â€” allocating a character vector per row.
4. Looks up each key in the named character vector `idx_lookup` (which itself has 6.46 million entries â€” named vector lookup in R is **O(n)** per query via hashing, but the repeated `paste` and character allocation across 6.46M iterations inside `lapply` is extremely expensive).

The result is ~6.46 million iterations of character allocation, pasting, and named-vector lookup. With an average of ~4 rook neighbors per cell, that is ~25.8 million `paste` operations and named-vector lookups, all inside an interpreted R loop. **This alone likely accounts for the majority of the 86+ hour runtime.**

### `compute_neighbor_stats` â€” O(N Ã— k) but lighter

This function iterates over 6.46M entries, pulling numeric subsets and computing `max`, `min`, `mean`. It is called 5 times (once per source variable). This is comparatively cheaper but still slow because of the per-row `lapply` overhead and repeated subsetting.

### Root cause summary

| Issue | Location | Impact |
|---|---|---|
| Per-row `paste()` + character key construction | `build_neighbor_lookup` | ~6.46M Ã— 4 string allocations |
| Named character vector lookup on 6.46M-entry vector | `build_neighbor_lookup` | Slow hash lookups in a loop |
| `lapply` over 6.46M rows (interpreted R loop) | Both functions | No vectorization benefit |
| Redundant: neighbor topology is **year-invariant** but rebuilt per cell-year | `build_neighbor_lookup` | 28Ã— more work than necessary |

The single most important insight: **the neighbor graph is defined over the 344,208 spatial cells and does not change across years.** The current code redundantly resolves neighbor relationships for every cell-year row (344,208 Ã— 28 = 9.6M), when it only needs to resolve them once for 344,208 cells and then apply them within each year via integer arithmetic.

---

## Optimization Strategy

1. **Eliminate all character key construction.** Replace `paste`-based lookups with pure integer-index arithmetic. Since every cell appears once per year in a panel, we can map `(cell_index, year_index)` â†’ row number with a simple integer matrix or arithmetic formula, completely avoiding `paste` and named vector lookups.

2. **Exploit year-invariance of the neighbor graph.** Build the neighbor index mapping once over the 344,208 cells, then replicate across years with integer offsets.

3. **Vectorize `compute_neighbor_stats`.** Replace the per-row `lapply` with a single vectorized operation using `data.table` grouping or direct C-level vectorized indexing. We can "unroll" the neighbor lookup into a long-form edge table and use `data.table` grouped aggregation (`max`, `min`, `mean`) in one pass.

4. **Preserve the trained Random Forest model and the original numerical estimand.** The output columns (neighbor max, min, mean for each of the 5 variables) are numerically identical â€” we are only changing how they are computed, not what is computed.

**Expected speedup:** From 86+ hours to **minutes** (roughly 2â€“10 minutes depending on disk I/O).

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#'
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#' Preserves the exact same output columns and numerical values.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and
#'                         all neighbor_source_vars columns.
#' @param id_order         integer vector of cell IDs in the order matching the
#'                         nb object (i.e., id_order[i] is the cell ID for the
#'                         i-th element of rook_neighbors_unique).
#' @param neighbors        spdep nb object (list of integer vectors of neighbor
#'                         indices into id_order).
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return data.table with original columns plus neighbor feature columns.
add_neighbor_features_optimized <- function(cell_data,
                                            id_order,
                                            neighbors,
                                            neighbor_source_vars) {

  # --- Step 0: Convert to data.table and preserve original row order ----------
  dt <- as.data.table(cell_data)
  dt[, .row_order := .I]

  # --- Step 1: Build a cell-index column (integer, no characters) -------------
  # Map each cell ID to its position in id_order (1-based).
  id_to_cellidx <- setNames(seq_along(id_order), as.character(id_order))
  dt[, cell_idx := id_to_cellidx[as.character(id)]]

  # --- Step 2: Build the directed edge table (cell-level, year-invariant) -----
  # Each row: (from_cell_idx, to_cell_idx)
  # This is done once for 344,208 cells, not 6.46M cell-years.
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells),
                  times = lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  edges <- data.table(from_cell_idx = from_idx,
                      to_cell_idx   = to_idx)

  # --- Step 3: Get unique years and create a year index -----------------------
  years_unique <- sort(unique(dt$year))
  n_years      <- length(years_unique)
  year_to_yidx <- setNames(seq_along(years_unique), as.character(years_unique))
  dt[, year_idx := year_to_yidx[as.character(year)]]

  # --- Step 4: Create a fast (cell_idx, year_idx) -> row_number lookup --------
  # We use a matrix: lookup_mat[cell_idx, year_idx] = row number in dt.
  # This requires that each (cell, year) pair appears at most once.
  lookup_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  lookup_mat[cbind(dt$cell_idx, dt$year_idx)] <- dt$.row_order

  # --- Step 5: Expand edges across all years (vectorized integer arithmetic) --
  # For each year, every edge (from, to) maps to row indices via lookup_mat.
  # We build a long table: (focal_row, neighbor_row) across all years.

  # Pre-allocate: n_edges * n_years rows (upper bound; some may be NA if cells

  # are missing in certain years, but the panel is balanced so this is exact).
  n_edges <- nrow(edges)

  # Replicate edges for each year
  year_idx_rep   <- rep(seq_len(n_years), each = n_edges)
  from_cell_rep  <- rep(edges$from_cell_idx, times = n_years)
  to_cell_rep    <- rep(edges$to_cell_idx,   times = n_years)

  focal_row    <- lookup_mat[cbind(from_cell_rep, year_idx_rep)]
  neighbor_row <- lookup_mat[cbind(to_cell_rep,   year_idx_rep)]

  # Remove pairs where either focal or neighbor is missing
  valid <- !is.na(focal_row) & !is.na(neighbor_row)
  edge_long <- data.table(focal_row    = focal_row[valid],
                          neighbor_row = neighbor_row[valid])

  # Free large temporaries
  rm(year_idx_rep, from_cell_rep, to_cell_rep, focal_row, neighbor_row, valid)
  gc()

  # --- Step 6: For each source variable, compute grouped stats in one pass ----
  for (var_name in neighbor_source_vars) {

    # Attach the neighbor's value to each edge
    edge_long[, nval := dt[[var_name]][neighbor_row]]

    # Remove NAs in the variable before aggregation
    agg <- edge_long[!is.na(nval),
                     .(nb_max  = max(nval),
                       nb_min  = min(nval),
                       nb_mean = mean(nval)),
                     by = focal_row]

    # Initialize new columns with NA
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Fill in computed values
    set(dt, i = agg$focal_row, j = max_col,  value = agg$nb_max)
    set(dt, i = agg$focal_row, j = min_col,  value = agg$nb_min)
    set(dt, i = agg$focal_row, j = mean_col, value = agg$nb_mean)
  }

  # --- Step 7: Clean up helper columns and restore order ----------------------
  dt[, c("cell_idx", "year_idx", ".row_order") := NULL]
  edge_long[, nval := NULL]

  return(dt[])
}


# =============================================================================
# Usage (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_optimized(
  cell_data            = cell_data,
  id_order             = id_order,
  neighbors            = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predict as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves Correctness

| Original operation | Optimized equivalent | Numerically identical? |
|---|---|---|
| `paste(id, year)` key â†’ named vector lookup | `matrix[cell_idx, year_idx]` integer lookup | âœ… Yes â€” same row is found |
| `vals[idx]` then `max/min/mean` per row | `data.table` grouped `max/min/mean` by `focal_row` | âœ… Yes â€” same values, same aggregation |
| NA handling: drop NA neighbors and NA values | Same: `!is.na` filter on both join and value | âœ… Yes |
| Output columns: `nb_max_ntl`, `nb_min_ntl`, `nb_mean_ntl`, etc. | Same column names, same positions | âœ… Yes |

The trained Random Forest model expects these exact column names and is never retrained â€” only used for `predict()`. Nothing changes on that side.

---

## Performance Analysis

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M `paste` + named-vector lookups in `lapply` | One integer matrix fill (`cbind` indexing), ~38.4M integer lookups vectorized in C |
| Neighbor stats (per variable) | ~6.46M `lapply` iterations Ã— 5 vars | One `data.table` grouped aggregation Ã— 5 vars |
| Total character allocations | ~25.8M `paste` calls | **Zero** |
| Estimated wall time | 86+ hours | **2â€“10 minutes** |
| Peak RAM (additional) | Moderate (6.46M-element list of integer vectors) | ~1.2 GB for `edge_long` (~38.4M rows Ã— 3 int columns) + lookup matrix (~38 MB). Fits in 16 GB. |

The key wins are: (1) eliminating all character/string operations, (2) exploiting year-invariance of the spatial topology, and (3) replacing interpreted R loops with vectorized `data.table` grouped aggregation backed by C.
END_RESPONSE>>>
