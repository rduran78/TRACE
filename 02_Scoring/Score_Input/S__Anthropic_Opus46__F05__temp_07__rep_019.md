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
run_id: Anthropic_F05_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F05
prompt_family_slug: STRING_KEY_REDUNDANCY
family_label: String-key probe
family_group: Line-level probe
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F05_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimized Solution

## Diagnosis

The repeated string-key work is **a symptom of a larger repeated-lookup pattern**, not merely a local inefficiency. Here's the full picture:

### Local inefficiency
- `paste(data$id, data$year, sep = "_")` constructs ~6.46 million strings to build `idx_lookup`. This happens once, but the named-vector lookup `idx_lookup[neighbor_keys]` is O(n) per probe in R's named-vector hashing â€” and it's called **6.46 million times** inside the `lapply`, each time constructing new paste keys for that row's neighbors.

### Broader structural problem
The real issue is that `build_neighbor_lookup` solves a problem **row-by-row with string manipulation** that is actually a **pure join/index problem** solvable in vectorized bulk. The algorithm does this:

1. For each of ~6.46M rows, find the row's cell ID â†’ find its spatial neighbors â†’ paste neighbor IDs with the current year â†’ look up the row indices of those neighbor-year combinations.

But the spatial neighbor graph is **year-invariant**. Every year has the same neighbor structure. So the entire lookup can be reformulated as:

> For each cell-year row `i`, the neighbor rows are simply: "the rows for cell `i`'s spatial neighbors **in the same year**."

This means we can:
- Build a single integer-indexed map from `(id, year)` â†’ row number (using `data.table` or a matrix, not string keys).
- Expand the neighbor list in bulk using vectorized operations, avoiding 6.46M `lapply` iterations entirely.

Then `compute_neighbor_stats` applies 5 variables Ã— 6.46M rows, each time subsetting by the neighbor indices. This too can be vectorized using `data.table` grouped operations or sparse-matrix multiplication.

**Estimated complexity reduction**: from O(N Ã— avg_neighbors Ã— string_ops) â‰ˆ billions of string operations â†’ O(N Ã— avg_neighbors) integer lookups done in bulk, plus vectorized grouped aggregation. Expected runtime: **minutes, not days**.

---

## Optimization Strategy

1. **Eliminate all string-key construction.** Use integer-indexed lookups via `data.table`.
2. **Vectorize the neighbor expansion.** Expand the year-invariant neighbor list into a full edge list `(from_row, to_row)` in one vectorized pass per year, or better, in one bulk operation across all years.
3. **Vectorize the statistics computation.** Use the edge list with `data.table` grouped aggregation to compute `max`, `min`, `mean` for all rows simultaneously.
4. **Process all 5 variables in a single grouped pass** over the edge list to minimize repeated grouping overhead.

---

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optimized neighbor-feature construction
# Replaces: build_neighbor_lookup + compute_neighbor_stats + outer loop
# Preserves: exact same numerical output (max, min, mean of each
#            neighbor source variable per cell-year row)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

library(data.table)

optimized_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {

  # ------------------------------------------------------------------
  # 0. Convert to data.table (by reference if already one; copy if not)
  # ------------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    dt <- as.data.table(cell_data)
  } else {
    dt <- copy(cell_data)
  }

  # Preserve original row order for final output
  dt[, .rowid := .I]

  # ------------------------------------------------------------------
  # 1. Build year-invariant spatial edge list (vectorized)
  #    rook_neighbors_unique is an nb object: a list of length

  #    length(id_order), where element k is an integer vector of
  #    neighbor indices into id_order (0 means no neighbors in spdep).
  # ------------------------------------------------------------------
  n_cells <- length(id_order)

  # Expand neighbor list into an edge list of (from_cell_id, to_cell_id)
  # Each element of rook_neighbors_unique[[k]] indexes into id_order
  from_idx <- rep(seq_len(n_cells),
                  times = lengths(rook_neighbors_unique))
  to_idx   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove spdep's 0-entries (no-neighbor sentinel)
  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  # Map positional indices to actual cell IDs
  spatial_edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, valid)

  # ------------------------------------------------------------------
  # 2. Build integer row-index map: (id, year) -> row number
  #    Using data.table keyed join (no strings, no hashing overhead)
  # ------------------------------------------------------------------
  row_map <- dt[, .(id, year, .rowid)]
  setkey(row_map, id, year)

  # ------------------------------------------------------------------
  # 3. Expand spatial edges across years into full (from_row, to_row)
  #    edge list. This is the key vectorized step.
  #
  #    For every year, every spatial edge (A->B) becomes a row-level

  #    edge (row_of_A_in_year_t -> row_of_B_in_year_t).
  # ------------------------------------------------------------------
  years <- sort(unique(dt$year))

  # Cross join spatial edges with years
  # For 1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in 16 GB easily
  edge_year <- CJ_dt_edges(spatial_edges, years)

  # Now join to get from_row and to_row
  setkey(edge_year, from_id, year)
  edge_year[row_map, from_row := i..rowid, on = .(from_id = id, year)]

  setkey(edge_year, to_id, year)
  edge_year[row_map, to_row := i..rowid, on = .(to_id = id, year)]

  # Drop edges where either endpoint is missing (boundary / incomplete panel)
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row)]

  # Keep only the row-index columns we need
  edge_list <- edge_year[, .(from_row, to_row)]
  rm(edge_year, row_map, spatial_edges)
  gc()

  # ------------------------------------------------------------------
  # 4. Compute neighbor stats: vectorized grouped aggregation
  #    For each from_row, gather to_row values, compute max/min/mean.
  # ------------------------------------------------------------------
  for (var_name in neighbor_source_vars) {

    message("Computing neighbor features for: ", var_name)

    # Attach the neighbor (to_row) values to the edge list
    edge_list[, val := dt[[var_name]][to_row]]

    # Grouped aggregation â€” one pass over the edge list
    stats <- edge_list[!is.na(val),
                       .(nmax  = max(val),
                         nmin  = min(val),
                         nmean = mean(val)),
                       by = from_row]

    # Initialize columns with NA
    max_col  <- paste0("n_max_",  var_name)
    min_col  <- paste0("n_min_",  var_name)
    mean_col <- paste0("n_mean_", var_name)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign results back by row index
    set(dt, i = stats$from_row, j = max_col,  value = stats$nmax)
    set(dt, i = stats$from_row, j = min_col,  value = stats$nmin)
    set(dt, i = stats$from_row, j = mean_col, value = stats$nmean)

    # Clean up the temporary column
    edge_list[, val := NULL]
  }

  # ------------------------------------------------------------------
  # 5. Restore original order and return
  # ------------------------------------------------------------------
  setorder(dt, .rowid)
  dt[, .rowid := NULL]

  # Return as data.frame if the input was data.frame
  if (!is.data.table(cell_data)) {
    return(as.data.frame(dt))
  }
  return(dt)
}


# Helper: cross-join spatial edges with year vector, memory-efficiently
CJ_dt_edges <- function(spatial_edges, years) {
  # Approach: replicate edges for each year
  n_edges <- nrow(spatial_edges)
  n_years <- length(years)

  data.table(
    from_id = rep(spatial_edges$from_id, times = n_years),
    to_id   = rep(spatial_edges$to_id,   times = n_years),
    year    = rep(years, each = n_edges)
  )
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage â€” drop-in replacement for the original outer loop
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimized_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors_unique = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Column names produced: n_max_ntl, n_min_ntl, n_mean_ntl, ... etc.
# These match the original output exactly.
```

---

## Memory Budget Check

| Object | Rows | Approx. Size |
|---|---|---|
| `dt` (main data) | 6.46M Ã— ~115 cols | ~5.5 GB |
| `edge_list` (row-index pairs) | ~38.5M Ã— 2 int cols | ~0.6 GB |
| `val` column on edge_list | ~38.5M Ã— 1 double | ~0.3 GB |
| `stats` (grouped result) | â‰¤6.46M Ã— 4 cols | ~0.2 GB |
| **Total peak** | | **~6.6 GB** âœ“ (fits 16 GB) |

---

## Why This Preserves Correctness

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[ref_idx]]` mapped through `id_order` | Same `id_order` mapping, same `rook_neighbors_unique` |
| Year matching | `paste(neighbor_id, year)` string key | Equi-join on `(id, year)` â€” identical semantics |
| Missing handling | `!is.na(result)` then `!is.na(neighbor_vals)` | `!is.na(from_row) & !is.na(to_row)` then `!is.na(val)` |
| Aggregates | `max`, `min`, `mean` on non-NA neighbor values | Identical functions in `data.table` grouped expr |
| Column names | `n_max_{var}`, `n_min_{var}`, `n_mean_{var}` | Same |
| RF model | Untouched | Untouched |

**Expected runtime: 5â€“15 minutes** (vs. 86+ hours), dominated by the `data.table` grouped aggregation over ~38.5M edges Ã— 5 variables.
END_RESPONSE>>>
