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
run_id: Anthropic_F04_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is clearly in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Here's why:

**`build_neighbor_lookup`** iterates over ~6.46 million rows via `lapply`, and **for each row** it:
1. Performs character coercion and named-vector lookup (`id_to_ref[as.character(...)]`).
2. Constructs neighbor keys by pasting strings (`paste(..., sep="_")`).
3. Looks up those keys in a ~6.46-million-entry named character vector (`idx_lookup[neighbor_keys]`).

String operations (`paste`, `as.character`, named vector matching on character keys) on 6.46M iterations, each touching ~4 neighbors on average (~1.37M relationships / 344K cells â‰ˆ 4 neighbors per cell, replicated across 28 years), result in billions of character allocations and linear scans (named vector lookup in base R is O(n) per query unless hashed). This alone explains the 86+ hour estimate.

**`compute_neighbor_stats`** is comparatively lightweightâ€”it just indexes a numeric vector and computes `max`/`min`/`mean`â€”but the `lapply` over 6.46M rows followed by `do.call(rbind, ...)` on a 6.46M-element list is also unnecessarily slow.

## Optimization Strategy

1. **Replace string-key lookups with integer-arithmetic joins.** Since every cell appears exactly once per year and years are contiguous (1992â€“2019), we can compute the row index of any (cell, year) pair arithmetically: `row = (year_offset) * n_cells + cell_position`. No strings needed.

2. **Vectorize neighbor lookup construction** using `data.table` or pre-sorted integer math, eliminating the per-row `lapply`.

3. **Vectorize `compute_neighbor_stats`** by building an edge list (source_row â†’ neighbor_row), then using `data.table` grouped aggregation to compute max/min/mean in one pass per variable, fully vectorized in C.

4. **Preserve the trained RF model and numerical estimand exactly**â€”the output columns are identical (same neighbor max/min/mean values), just computed faster.

Expected speedup: from ~86 hours to **minutes**.

## Optimized Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a fully vectorized edge list (source_row -> neighbor_row)
#         This replaces build_neighbor_lookup entirely.
# ==============================================================

build_neighbor_edgelist <- function(cell_data_dt, id_order, neighbors) {
  # cell_data_dt: a data.table with columns 'id' and 'year', 
  #               plus a row index column '.row_id'
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer vectors of neighbor positions)
  
  n_cells <- length(id_order)
  years   <- sort(unique(cell_data_dt$year))
  n_years <- length(years)
  
  # Map each cell ID to its position in id_order (1-based)
  id_to_pos <- setNames(seq_along(id_order), as.character(id_order))
  
  # Build a lookup: for each (cell_pos, year) -> row index in cell_data_dt
  # Sort data by (id position, year) so we can do arithmetic lookup
  cell_data_dt[, cell_pos := id_to_pos[as.character(id)]]
  setorder(cell_data_dt, cell_pos, year)
  cell_data_dt[, .row_id := .I]
  
  # Create a matrix-style lookup: row_index_matrix[cell_pos, year_offset+1] = .row_id
  # But with 344K cells Ã— 28 years this is ~9.6M entries, easily fits in memory
  year_to_offset <- setNames(seq_along(years) - 1L, as.character(years))
  
  # Fast lookup table: key = (cell_pos, year) -> row_id
  lookup_dt <- cell_data_dt[, .(cell_pos, year, .row_id)]
  setkey(lookup_dt, cell_pos, year)
  
  # Build the cell-level edge list (no year dimension yet)
  # From the nb object: for cell i, neighbors[[i]] gives positions of neighbors
  # We expand this into a two-column data.table: (source_cell_pos, neighbor_cell_pos)
  
  src_lengths <- lengths(neighbors)
  cell_edges <- data.table(
    src_pos = rep(seq_len(n_cells), times = src_lengths),
    nbr_pos = unlist(neighbors, use.names = FALSE)
  )
  # Remove zero-neighbor entries (spdep uses integer(0) for islands, 

  # but rep/unlist handles that naturallyâ€”zero-length entries contribute nothing)
  
  # Now cross with years to get row-level edge list
  year_dt <- data.table(year = years)
  
  # Cross join: every cell-edge Ã— every year
  # This gives ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits easily in RAM
  cell_edges_by_year <- cell_edges[, CJ_year := 1L]  # dummy
  year_dt[, CJ_year := 1L]
  
  # Efficient cross join
  edge_year <- cell_edges[, .(src_pos, nbr_pos, year = list(years)), by = .I]
  edge_year <- cell_edges[rep(seq_len(.N), each = n_years)]
  edge_year[, year := rep(years, times = nrow(cell_edges))]
  
  # Now join to get source row_id and neighbor row_id
  # Source row
  setkey(edge_year, src_pos, year)
  edge_year[lookup_dt, src_row := i..row_id, on = .(src_pos = cell_pos, year)]
  
  # Neighbor row
  setkey(edge_year, nbr_pos, year)
  edge_year[lookup_dt, nbr_row := i..row_id, on = .(nbr_pos = cell_pos, year)]
  
  # Drop edges where either source or neighbor row is missing (incomplete panel)
  edge_year <- edge_year[!is.na(src_row) & !is.na(nbr_row)]
  
  list(
    edge_year  = edge_year[, .(src_row, nbr_row)],
    cell_data_dt = cell_data_dt
  )
}

# ==============================================================
# STEP 2: Vectorized neighbor stats using data.table grouping
#         This replaces compute_neighbor_stats entirely.
# ==============================================================

compute_neighbor_stats_vec <- function(cell_data_dt, edges, var_name) {
  # edges: data.table with columns src_row, nbr_row
  # Returns the same 3 columns as original: neighbor_max, neighbor_min, neighbor_mean
  
  n_rows <- nrow(cell_data_dt)
  
  # Extract neighbor values
  work <- edges[, .(src_row, val = cell_data_dt[[var_name]][nbr_row])]
  
  # Drop NA values
  work <- work[!is.na(val)]
  
  # Grouped aggregation â€” fully vectorized in C via data.table
  stats <- work[, .(
    nmax  = max(val),
    nmin  = min(val),
    nmean = mean(val)
  ), keyby = src_row]
  
  # Allocate full result (NA for rows with no valid neighbors)
  result <- data.table(
    nmax  = rep(NA_real_, n_rows),
    nmin  = rep(NA_real_, n_rows),
    nmean = rep(NA_real_, n_rows)
  )
  result[stats$src_row, `:=`(
    nmax  = stats$nmax,
    nmin  = stats$nmin,
    nmean = stats$nmean
  )]
  
  # Name columns to match original pipeline expectations
  prefix <- var_name
  setnames(result, c(
    paste0(prefix, "_neighbor_max"),
    paste0(prefix, "_neighbor_min"),
    paste0(prefix, "_neighbor_mean")
  ))
  
  result
}

# ==============================================================
# STEP 3: Main execution â€” drop-in replacement for outer loop
# ==============================================================

# Convert to data.table (non-destructive)
cell_data_dt <- as.data.table(cell_data)

# Build the edge list once (replaces build_neighbor_lookup)
message("Building vectorized neighbor edge list...")
timing <- system.time({
  result <- build_neighbor_edgelist(cell_data_dt, id_order, rook_neighbors_unique)
  edges        <- result$edge_year
  cell_data_dt <- result$cell_data_dt
})
message("Edge list built in ", round(timing["elapsed"], 1), " seconds")

# Compute neighbor features for all 5 variables (replaces the for loop)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor stats for: ", var_name)
  timing <- system.time({
    stats_cols <- compute_neighbor_stats_vec(cell_data_dt, edges, var_name)
  })
  message("  Done in ", round(timing["elapsed"], 1), " seconds")
  
  # Bind columns to cell_data_dt
  cell_data_dt <- cbind(cell_data_dt, stats_cols)
}

# Restore original row order if needed (we sorted by cell_pos, year)
setorder(cell_data_dt, .row_id)

# Remove helper columns before prediction
cell_data_dt[, c("cell_pos", ".row_id") := NULL]

# Convert back to data.frame if the RF predict() method requires it
cell_data <- as.data.frame(cell_data_dt)

# ---- RF prediction (unchanged) ----
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This Is Correct and Equivalent

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` are computed over exactly the same neighbor index sets. The edge list encodes the identical relationships as the original `neighbor_lookup` list. |
| **RF model untouched** | No retraining. The output data.frame has the same column names, types, and values. `predict()` receives identical input. |
| **Memory** | The edge list is ~38.5M rows Ã— 2 integer columns â‰ˆ 308 MB. The working `val` column adds ~308 MB temporarily. Total peak â‰ˆ ~4 GB, well within 16 GB. |
| **Speed** | `data.table` grouped aggregation on 38.5M rows runs in seconds per variable. Total wall time: estimated **2â€“10 minutes** versus 86+ hours. |

## Summary of Root Causes and Fixes

| Original bottleneck | Complexity | Fix | New complexity |
|---|---|---|---|
| `paste(id, year)` string keys, 6.46M times | O(N) string allocs | Integer arithmetic + `data.table` keyed join | O(N) integer ops, vectorized in C |
| Named vector lookup `idx_lookup[keys]` (linear scan) | O(N Ã— k Ã— M) worst case | `data.table` binary-search join (`on=`) | O(N Ã— k Ã— log M) |
| Per-row `lapply` over 6.46M rows | R-level loop overhead | Fully vectorized `data.table` grouped aggregation | Single C-level pass |
| `do.call(rbind, list_of_6.46M)` | O(NÂ²) copy growth | Direct indexed assignment into pre-allocated columns | O(N) |
END_RESPONSE>>>
