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
run_id: Anthropic_F07_T00_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The `idx_lookup` named vector has 6.46 million entries** â€” every single neighbor key lookup is an O(n) hash probe against a massive character vector.
3. **`lapply` over 6.46M rows is inherently slow in pure R** â€” there is no vectorization; each iteration does allocation, string ops, and subsetting.

`compute_neighbor_stats` is a secondary bottleneck: it also loops over 6.46M elements, extracting subsets of a numeric vector and computing `max/min/mean` one row at a time.

**Combined**, these two stages run in roughly O(N Ã— k) with large constant factors from R's interpreted overhead, string operations, and memory pressure on a 16 GB laptop â€” hence the 86+ hour estimate.

## Optimization Strategy

### 1. Replace string-key lookups with integer-key lookups using `data.table`

Instead of building a giant named character vector (`idx_lookup`), build an integer-keyed `data.table` with `(id, year) â†’ row_index` and use binary-search joins. This eliminates all string pasting inside the per-row loop.

### 2. Vectorize `build_neighbor_lookup` entirely

Expand the neighbor list into an edge table `(source_row, neighbor_cell_id)`, join against the `(id, year) â†’ row_index` table in one vectorized `data.table` merge, and then split back into a list. This replaces 6.46M R-level iterations with a single vectorized join.

### 3. Vectorize `compute_neighbor_stats`

Use the edge table directly: for each `(source_row, neighbor_row)` pair, pull the variable value, then `group by source_row` and compute `max`, `min`, `mean` in one `data.table` aggregation. This replaces 6.46M `lapply` iterations per variable with one grouped aggregation.

### 4. Memory management

The edge table will have ~6.46M Ã— avg_neighbors â‰ˆ 25â€“26 million rows (since ~1.37M directed edges per year Ã— ~19 years on average, but more precisely: each cell-year has its own neighbor set). We reuse the same edge table for all 5 variables. Peak memory stays well within 16 GB.

### 5. Preserve the trained RF model and numerical estimand

We only change **how** the neighbor features are computed, not **what** they are. The output columns are numerically identical (same `max`, `min`, `mean` of the same neighbor values). The RF model is never retrained.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. Vectorized neighbor-lookup construction (replaces build_neighbor_lookup)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table <- function(cell_data_dt, id_order, rook_neighbors_unique) {
  # Map each position in id_order to its neighbor cell IDs
  # id_order[i] has neighbors id_order[rook_neighbors_unique[[i]]]
  
  n_cells <- length(id_order)
  
  # Build cell-level edge list: source_cell_id -> neighbor_cell_id
  source_ids <- rep(
    id_order,
    times = lengths(rook_neighbors_unique)
  )
  neighbor_ids <- id_order[unlist(rook_neighbors_unique)]
  
  cell_edges <- data.table(
    source_cell_id   = source_ids,
    neighbor_cell_id = neighbor_ids
  )
  
  # Build row-index table: (id, year) -> row_idx
  cell_data_dt[, row_idx := .I]
  row_index <- cell_data_dt[, .(id, year, row_idx)]
  
  # Get unique years
  years <- unique(cell_data_dt$year)
  
  # Cross-join cell edges with years, then join to get source and neighbor row indices
  # This is the key vectorized step.
  
  # For each year, every cell edge applies. Use CJ-like expansion:
  cell_edges_expanded <- cell_edges[
    , .(year = years), by = .(source_cell_id, neighbor_cell_id)
  ]
  
  # Join to get source row index
  setkey(row_index, id, year)
  
  cell_edges_expanded[
    row_index,
    source_row := i.row_idx,
    on = .(source_cell_id = id, year = year)
  ]
  
  # Join to get neighbor row index
  cell_edges_expanded[
    row_index,
    neighbor_row := i.row_idx,
    on = .(neighbor_cell_id = id, year = year)
  ]
  
  # Drop edges where either side is missing (masked cells / boundary)
  edge_table <- cell_edges_expanded[
    !is.na(source_row) & !is.na(neighbor_row),
    .(source_row, neighbor_row)
  ]
  
  setkey(edge_table, source_row)
  
  return(edge_table)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Vectorized neighbor-stats computation (replaces compute_neighbor_stats)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_and_add_neighbor_features_vec <- function(cell_data_dt, var_name, edge_table) {
  # Extract the variable values for all neighbor rows
  vals <- cell_data_dt[[var_name]]
  
  # Build a working table with neighbor values
  work <- edge_table[, .(source_row, nval = vals[neighbor_row])]
  
  # Remove NA neighbor values
  work <- work[!is.na(nval)]
  
  # Aggregate by source_row
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = source_row]
  
  # Assign back to cell_data_dt
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  # Initialize with NA
  cell_data_dt[, (max_col)  := NA_real_]
  cell_data_dt[, (min_col)  := NA_real_]
  cell_data_dt[, (mean_col) := NA_real_]
  
  # Fill in computed values
  cell_data_dt[agg$source_row, (max_col)  := agg$nb_max]
  cell_data_dt[agg$source_row, (min_col)  := agg$nb_min]
  cell_data_dt[agg$source_row, (mean_col) := agg$nb_mean]
  
  invisible(cell_data_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Main pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table (in-place if already a data.table; copy if data.frame)
if (!is.data.table(cell_data)) {
  cell_data <- as.data.table(cell_data)
}

# Build the edge table once (replaces build_neighbor_lookup)
message("Building vectorized edge table...")
edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_table), big.mark = ",")))

# Compute neighbor features for all 5 variables
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_vec(cell_data, var_name, edge_table)
}

message("Done. Neighbor features added to cell_data.")

# The trained Random Forest model is untouched.
# The new columns are numerically identical to the original implementation.
```

---

## Performance Comparison

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~60â€“70 hrs (6.46M `lapply` with string ops) | ~2â€“5 min (one vectorized `data.table` join) |
| `compute_neighbor_stats` Ã— 5 vars | ~15â€“20 hrs (5 Ã— 6.46M `lapply`) | ~3â€“8 min (5 grouped aggregations) |
| **Total** | **~86 hrs** | **~5â€“13 min** |

### Why this is safe

- **Numerical equivalence**: `max`, `min`, `mean` are computed over exactly the same neighbor sets. The edge table encodes the identical `(source_row, neighbor_row)` pairs that the original `neighbor_lookup` list would contain.
- **Topology preservation**: The rook-neighbor structure from `rook_neighbors_unique` (the `spdep::nb` object) is faithfully expanded â€” no raster focal assumptions are made.
- **RF model untouched**: No retraining occurs; the feature columns have the same names and identical values.
- **Memory**: The edge table (~25â€“26M rows Ã— 2 integer columns â‰ˆ ~200 MB) plus `cell_data` (~6.46M Ã— 110 cols) fits comfortably in 16 GB.
END_RESPONSE>>>
