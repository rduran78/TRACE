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
run_id: Anthropic_F07_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is clear: **`build_neighbor_lookup` creates a list of 6.46 million elements, each produced by an `lapply` call that performs character pasting, hash-table lookups, and NA filtering row-by-row.** This is an O(n) loop in pure R over ~6.46M rows, with expensive string operations (`paste`, named-vector indexing via character keys) inside the loop. The `compute_neighbor_stats` function then loops over the same 6.46M-element list again for each of the 5 variables, yielding ~32.3M R-level iterations total.

**Specific costs:**

1. **`build_neighbor_lookup`**: ~6.46M iterations, each doing `paste()`, character vector subsetting on `idx_lookup` (a named vector of length 6.46M â€” linear scan or hash lookup per call), and NA removal. This alone can take tens of hours.
2. **`compute_neighbor_stats`**: 5 variables Ã— 6.46M iterations = 32.3M R-level function calls, each extracting a sub-vector, removing NAs, and computing max/min/mean.
3. **Memory**: The `neighbor_lookup` list of 6.46M integer vectors has enormous R object overhead (each list element is a separate SEXP).

## Optimization Strategy

**Replace the row-level R loops with vectorized sparse-matrix operations.**

The key insight: the neighbor relationships can be encoded as a **sparse adjacency matrix W** of dimension N_cells Ã— N_cells (344,208 Ã— 344,208). For any year, the neighbor statistics (max, min, mean) over a variable can be computed by operating on the sparse matrix. But since the neighbor structure is **time-invariant** (same spatial grid every year), we can:

1. Build the sparse adjacency matrix **once** from `rook_neighbors_unique` (the `nb` object).
2. For each year, extract the variable vector for that year's cells, then use sparse matrix operations to compute neighbor sums and counts (for mean), and use `dgCMatrix` column/row operations for max/min.
3. Alternatively, build a **single large sparse matrix** of dimension N_rows Ã— N_rows (6.46M Ã— 6.46M) that encodes "row i's neighbors in the panel" â€” but this is memory-prohibitive.

**Best approach: year-by-year vectorized computation using the 344K Ã— 344K spatial weights matrix.**

- Neighbor **mean** = (W %*% x) / (W %*% 1_{non-NA}) â€” a sparse matrix-vector multiply, extremely fast.
- Neighbor **max** and **min** require a grouped operation. We use the sparse matrix structure to do this efficiently via `data.table` grouping on a pre-built edge list.

**The edge-list approach is the most general and fastest:**

1. Convert the `nb` object to an edge list (from, to) â€” ~1.37M directed edges.
2. Expand to panel edges: for each year, the edge (from, to) maps to (from_row, to_row) â€” ~1.37M Ã— 28 = ~38.5M panel-edge rows. This is done via a merge/join, not a loop.
3. Group by `from_row` and compute max, min, mean of the target variable at `to_row`. This is a single `data.table` grouped aggregation â€” extremely fast.

**Expected speedup: from 86+ hours to ~2â€“5 minutes.**

## Working R Code

```r
library(data.table)
library(spdep)

# ---------------------------------------------------------------
# 0.  Ensure cell_data is a data.table keyed for fast joins
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)

# ---------------------------------------------------------------
# 1.  Build the spatial edge list ONCE from the nb object
#     rook_neighbors_unique is the spdep::nb object
#     id_order is the vector of cell IDs in nb-index order
# ---------------------------------------------------------------
build_edge_list <- function(nb_obj, id_order) {
  from_idx <- rep(seq_along(nb_obj),
                  lengths(nb_obj))
  to_idx   <- unlist(nb_obj)


  # Remove the 0-neighbor sentinel that spdep uses

  valid    <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(rook_neighbors_unique, id_order)
# edge_dt has ~1,373,394 rows: (from_id, to_id)

# ---------------------------------------------------------------
# 2.  Build a row-index table for fast joining
# ---------------------------------------------------------------
cell_dt[, row_idx := .I]
setkey(cell_dt, id, year)

# Create a join key table:  (id, year) -> row_idx
key_table <- cell_dt[, .(id, year, row_idx)]

# ---------------------------------------------------------------
# 3.  Expand edge list across all years (vectorized cross-join)
#     Result: for every (from_id, to_id, year) we know both row indices
# ---------------------------------------------------------------
years <- sort(unique(cell_dt$year))  # 1992:2019, 28 values

# Cross join edges Ã— years
panel_edges <- CJ_dt <- edge_dt[, .(from_id, to_id)]
panel_edges <- panel_edges[, .(year = years), by = .(from_id, to_id)]
# This is ~1.37M Ã— 28 â‰ˆ 38.5M rows

# Attach the "from" row index
setnames(key_table, c("id", "year", "row_idx"),
         c("from_id", "year", "from_row"))
panel_edges <- key_table[panel_edges, on = .(from_id, year), nomatch = 0L]

# Attach the "to" (neighbor) row index
setnames(key_table, c("from_id", "year", "from_row"),
         c("to_id",   "year", "to_row"))
panel_edges <- key_table[panel_edges, on = .(to_id, year), nomatch = 0L]

# Restore key_table names for later use
setnames(key_table, c("to_id", "year", "to_row"),
         c("id",    "year", "row_idx"))

# panel_edges now has columns: from_row, to_row, from_id, to_id, year
# Keep only what we need to save memory
panel_edges <- panel_edges[, .(from_row, to_row)]
setkey(panel_edges, from_row)

# ---------------------------------------------------------------
# 4.  Compute neighbor stats for each variable (vectorized)
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  # Pull the variable values indexed by row position
  panel_edges[, nbr_val := cell_dt[[var_name]][to_row]]

  # Remove NA neighbor values before aggregation
  work <- panel_edges[!is.na(nbr_val)]

  # Grouped aggregation â€” single pass, data.table optimized

  agg <- work[, .(
    nbr_max  = max(nbr_val),
    nbr_min  = min(nbr_val),
    nbr_mean = mean(nbr_val)
  ), keyby = .(from_row)]

  # Initialize result columns with NA
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  # Assign results back (only rows that had valid neighbors)
  set(cell_dt, i = agg$from_row, j = max_col,  value = agg$nbr_max)
  set(cell_dt, i = agg$from_row, j = min_col,  value = agg$nbr_min)
  set(cell_dt, i = agg$from_row, j = mean_col, value = agg$nbr_mean)

  # Clean up the temporary column
  panel_edges[, nbr_val := NULL]

  cat("Done:", var_name, "\n")
}

# ---------------------------------------------------------------
# 5.  Convert back to data.frame if downstream code expects it
# ---------------------------------------------------------------
cell_dt[, row_idx := NULL]
cell_data <- as.data.frame(cell_dt)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed over exactly the same neighbor sets as the original code â€” the edge list is derived from the identical `rook_neighbors_unique` nb object and matched by `(id, year)`. |
| **Irregular topology / masked cells** | The `nomatch = 0L` join drops edges where a neighbor cell-year doesn't exist in the panel, exactly replicating the `!is.na(result)` filter in the original `build_neighbor_lookup`. |
| **NA handling** | `work <- panel_edges[!is.na(nbr_val)]` replicates the `neighbor_vals[!is.na(neighbor_vals)]` filter. Cells with zero valid neighbors get `NA` for all three stats. |
| **Trained RF model** | No model retraining â€” we only recompute the same feature columns that already exist. |

## Performance Estimate

| Step | Cost |
|---|---|
| Edge list construction | < 1 second (vectorized `rep`/`unlist`) |
| Panel-edge expansion (CJ) | ~38.5M rows, ~10â€“20 seconds, ~1.5 GB |
| Joins to attach row indices | ~30 seconds each (keyed `data.table` join) |
| Per-variable aggregation | ~38.5M rows grouped by ~6.46M keys, ~20â€“40 seconds each |
| **Total** | **~3â€“5 minutes** on a 16 GB laptop |

This replaces the 86+ hour runtime with a fully vectorized pipeline that produces bit-identical results.
END_RESPONSE>>>
