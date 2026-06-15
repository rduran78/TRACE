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
run_id: Anthropic_F04_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` â€” Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M â€” each lookup is O(n) hash probe against a very large vector).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop is catastrophically slow. The `idx_lookup` named vector with ~6.46M entries makes each key lookup expensive, and this is repeated for every neighbor of every row.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows (Ã—5 variables)

Each iteration calls `max`, `min`, `mean` on a small vector. The per-call overhead of `lapply` plus anonymous function dispatch, repeated 6.46M Ã— 5 = 32.3M times, is enormous.

### 3. Summary of cost

| Component | Calls | Dominant cost |
|---|---|---|
| `build_neighbor_lookup` | 6.46M iterations | `paste` + named-vector hash lookup on 6.46M-entry table per iteration |
| `compute_neighbor_stats` | 6.46M Ã— 5 = 32.3M iterations | R-level loop overhead, small-vector summary stats |

Estimated wall-clock: 86+ hours is consistent with this analysis.

---

## Optimization Strategy

**Principle: Replace row-level R loops with vectorized and/or `data.table`-based operations.**

### Step A â€” Vectorize `build_neighbor_lookup`

Instead of building a per-row list of neighbor row indices via `lapply`, we construct a **flat edge table** (a two-column data.table: `from_row`, `to_row`) that maps every row to its neighbor rows. This is done entirely with vectorized joins:

1. Expand the `nb` object into a flat edge list of `(cell_id, neighbor_cell_id)` pairs â€” only ~1.37M edges.
2. Join with the panel data to replicate edges across years â€” this produces ~1.37M Ã— 28 â‰ˆ 38.4M `(from_row, to_row)` pairs.
3. All joins are `data.table` keyed merges â€” O(n log n), no R-level loops.

### Step B â€” Vectorize `compute_neighbor_stats`

With the flat edge table, computing `max`, `min`, `mean` of neighbor values is a single **grouped aggregation**:

```
edge_table[, .(max_v = max(val), min_v = min(val), mean_v = mean(val)), by = from_row]
```

This is a single `data.table` grouped operation â€” internally parallelized in C, no R-level row iteration.

### Step C â€” Loop over the 5 variables

Each variable's neighbor stats are computed with one grouped aggregation on the edge table. Five passes total.

### Expected speedup

| Component | Before | After |
|---|---|---|
| Neighbor lookup | 6.46M R-level iterations with string ops | One vectorized `data.table` merge (~38M rows) |
| Neighbor stats (per var) | 6.46M R-level iterations | One `data.table` grouped aggregation |
| **Total estimated time** | **86+ hours** | **~2â€“10 minutes** |

The trained Random Forest model is untouched. The numerical output (max, min, mean of neighbor values per row per variable) is identical.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP A: Build a flat edge table (vectorized, no row-level R loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table <- function(cell_dt, id_order, neighbors) {
  # 1. Expand the nb object into a flat (cell_id -> neighbor_cell_id) edge list.
  #    neighbors is a list of integer index vectors (spdep::nb object),
  #    where each index refers to a position in id_order.
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove zero-length / self-referencing artifacts from nb objects
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edge_cells <- data.table(
    cell_id          = id_order[from_idx],
    neighbor_cell_id = id_order[to_idx]
  )
  # ~1.37 M rows (directed rook edges)

  # 2. Get the unique years present in the panel.
  years <- sort(unique(cell_dt$year))

  # 3. Cross-join edges Ã— years, then map to row indices.
  #    First, build a row-index lookup keyed on (id, year).
  cell_dt[, row_idx := .I]  # preserve original row order
  setkey(cell_dt, id, year)

  # Expand edges across all years (vectorized CJ-style via merge)
  edge_years <- CJ(edge_row = seq_len(nrow(edge_cells)), year = years)
  edge_years[, `:=`(
    cell_id          = edge_cells$cell_id[edge_row],
    neighbor_cell_id = edge_cells$neighbor_cell_id[edge_row]
  )]
  edge_years[, edge_row := NULL]

  # 4. Join to get from_row (the row index of the focal cell-year)
  setnames(edge_years, "cell_id", "id")
  from_map <- cell_dt[, .(id, year, row_idx)]
  setkey(from_map, id, year)
  setkey(edge_years, id, year)
  edge_years <- from_map[edge_years, nomatch = 0L]
  setnames(edge_years, "row_idx", "from_row")

  # 5. Join to get to_row (the row index of the neighbor cell-year)
  setnames(edge_years, "neighbor_cell_id", "id_nb")
  to_map <- cell_dt[, .(id, year, row_idx)]
  setkey(to_map, id, year)
  setkey(edge_years, id_nb, year)
  setnames(edge_years, c("id_nb"), c("id"))
  # re-key for the join on neighbor id + year
  setkey(edge_years, id, year)
  edge_years <- to_map[edge_years, nomatch = 0L]
  setnames(edge_years, "row_idx", "to_row")

  # 6. Return only the two columns we need.
  edge_years[, .(from_row, to_row)]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP A (memory-friendly alternative): chunk the year cross-join
# Use this version if the ~38 M row edge_years table causes memory
# pressure on a 16 GB laptop.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

build_neighbor_edge_table_chunked <- function(cell_dt, id_order, neighbors) {
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)
  valid    <- to_idx > 0L
  edge_cells <- data.table(
    cell_id          = id_order[from_idx[valid]],
    neighbor_cell_id = id_order[to_idx[valid]]
  )

  cell_dt[, row_idx := .I]
  lookup <- cell_dt[, .(id, year, row_idx)]
  setkey(lookup, id, year)

  years <- sort(unique(cell_dt$year))

  # Process one year at a time to limit peak memory
  edge_list <- lapply(years, function(yr) {
    yr_edges <- copy(edge_cells)
    yr_edges[, year := yr]

    # from_row
    setkey(yr_edges, cell_id, year)
    yr_edges <- lookup[yr_edges, on = .(id = cell_id, year), nomatch = 0L]
    setnames(yr_edges, "row_idx", "from_row")

    # to_row
    setkey(yr_edges, neighbor_cell_id, year)
    yr_edges <- lookup[yr_edges, on = .(id = neighbor_cell_id, year), nomatch = 0L]
    setnames(yr_edges, "row_idx", "to_row")

    yr_edges[, .(from_row, to_row)]
  })

  rbindlist(edge_list)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP B: Compute neighbor stats for one variable (single grouped agg)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {
  # Attach the neighbor's value to each edge
  vals <- cell_dt[[var_name]]
  edge_dt[, val := vals[to_row]]

  # Grouped aggregation â€” runs in C inside data.table
  stats <- edge_dt[!is.na(val),
    .(
      max_val  = max(val),
      min_val  = min(val),
      mean_val = mean(val)
    ),
    by = from_row
  ]

  # Map back to full row set (rows with no valid neighbors get NA)
  n <- nrow(cell_dt)
  max_out  <- rep(NA_real_, n)
  min_out  <- rep(NA_real_, n)
  mean_out <- rep(NA_real_, n)

  max_out[stats$from_row]  <- stats$max_val
  min_out[stats$from_row]  <- stats$min_val
  mean_out[stats$from_row] <- stats$mean_val

  # Add columns to cell_dt (by reference)
  set(cell_dt, j = paste0(var_name, "_neighbor_max"),  value = max_out)
  set(cell_dt, j = paste0(var_name, "_neighbor_min"),  value = min_out)
  set(cell_dt, j = paste0(var_name, "_neighbor_mean"), value = mean_out)

  # Clean up temporary column
  edge_dt[, val := NULL]

  invisible(cell_dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP C: Outer loop â€” drop-in replacement for the original pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Convert to data.table if not already
if (!is.data.table(cell_data)) cell_data <- as.data.table(cell_data)

# Build the edge table once (use chunked version on 16 GB laptop)
edge_table <- build_neighbor_edge_table_chunked(
  cell_data, id_order, rook_neighbors_unique
)
setkey(edge_table, from_row)

# Compute neighbor features for each source variable
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_stats_fast(cell_data, edge_table, var_name)
  cat("Done:", var_name, "\n")
}

# cell_data now contains the 15 new columns:
#   ntl_neighbor_max, ntl_neighbor_min, ntl_neighbor_mean,
#   ec_neighbor_max,  ec_neighbor_min,  ec_neighbor_mean,
#   ... etc.
#
# Feed cell_data (with the same 110 predictor column names) into
# predict(trained_rf_model, newdata = cell_data) as before.
# The trained Random Forest model is completely unchanged.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The grouped `max`, `min`, `mean` in `data.table` produce identical IEEE-754 results to the original R `max`, `min`, `mean` calls on the same neighbor value vectors. NA handling (`!is.na`) is replicated. Rows with no valid neighbors receive `NA`, matching the original `c(NA, NA, NA)` return. |
| **Trained RF model untouched** | No model retraining occurs. The output columns have the same names and semantics. `predict()` is called identically downstream. |
| **Neighbor topology preserved** | The flat edge table is a lossless expansion of the same `spdep::nb` object (`rook_neighbors_unique`) and `id_order` used in the original code. |

---

## Expected Performance on 16 GB Laptop

| Stage | Estimated time | Peak memory |
|---|---|---|
| `build_neighbor_edge_table_chunked` | ~30â€“90 seconds | ~2â€“3 GB (one year chunk at a time) |
| `compute_neighbor_stats_fast` Ã— 5 vars | ~30â€“60 seconds each, ~3â€“5 min total | ~1 GB temporary |
| **Total** | **~5â€“8 minutes** | **Well within 16 GB** |

This represents a roughly **600â€“1000Ã— speedup** over the original 86+ hour estimate, achieved entirely by eliminating R-level row iteration in favor of vectorized `data.table` keyed joins and grouped aggregations.
END_RESPONSE>>>
