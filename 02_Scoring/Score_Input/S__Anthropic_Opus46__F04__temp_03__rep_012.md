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
run_id: Anthropic_F04_T03_R012
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 12
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_012.md

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
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop is catastrophically slow. The `idx_lookup` named vector with ~6.46M entries makes each key lookup expensive, and this is repeated for every neighbor of every row.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows (Ã—5 variables)

Each iteration computes `max`, `min`, `mean` on a small vector. The per-call overhead of `lapply` plus anonymous function dispatch, repeated 6.46M Ã— 5 = 32.3M times, is enormous. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also very slow.

### Quantitative estimate of the problem

| Component | Iterations | Estimated wall-clock |
|---|---|---|
| `build_neighbor_lookup` | 6.46M, each with string ops + large hash lookup | ~20â€“40 hours |
| `compute_neighbor_stats` | 6.46M Ã— 5 vars, each with subsetting + summary stats | ~40â€“50 hours |
| `do.call(rbind, ...)` | 5 calls binding 6.46M rows | ~2â€“5 hours |
| **Total** | | **~62â€“95 hours** |

This is consistent with the reported 86+ hour estimate.

---

## Optimization Strategy

**Principle: Replace row-level R loops with vectorized joins and grouped vectorized operations using `data.table`.**

### Step A â€” `build_neighbor_lookup` â†’ Vectorized `data.table` join

Instead of building a list of 6.46M integer vectors (one per row), build a **long-form edge table** `(row_i, neighbor_row_j)` using vectorized operations:

1. Expand the `nb` object into a long edge list `(cell_id, neighbor_cell_id)` â€” only ~1.37M edges.
2. Cross-join with years to get `(cell_id, year, neighbor_cell_id, year)` â€” ~1.37M Ã— 28 = ~38.5M rows.
3. Join against the data to resolve each `(neighbor_cell_id, year)` to its row index.

This replaces 6.46M interpreted iterations with a single keyed `data.table` merge.

### Step B â€” `compute_neighbor_stats` â†’ Grouped `data.table` aggregation

With the long edge table from Step A, computing `max`, `min`, `mean` of neighbor values is a single grouped aggregation:

```
edge_table[data, on = neighbor_row][, .(max_v, min_v, mean_v), by = row_i]
```

This replaces 6.46M `lapply` iterations per variable with one vectorized `data.table` grouped operation.

### Expected speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup | ~20â€“40 hrs | ~1â€“3 min | ~500â€“1000Ã— |
| Neighbor stats (Ã—5) | ~40â€“50 hrs | ~2â€“5 min | ~500â€“1000Ã— |
| **Total** | **~86 hrs** | **~5â€“10 min** | **~500â€“1000Ã—** |

### What is preserved

- The trained Random Forest model is untouched (no retraining).
- The numerical output (max, min, mean of rook-neighbor values per cell-year) is identical â€” the same estimand is computed.

---

## Working R Code

```r
library(data.table)

#' Convert an spdep nb object to a long-form edge data.table.
#' Each row is a directed edge: (focal_id, neighbor_id).
nb_to_edge_dt <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep nb object)
  # id_order is the vector mapping position -> cell id
  lens <- lengths(neighbors)
  focal_idx <- rep(seq_along(neighbors), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

#' Build a long-form edge table with row indices into `cell_data`
#' for both the focal cell-year and the neighbor cell-year.
#'
#' Returns a data.table with columns: (focal_row, neighbor_row)
#' where both are integer row indices into cell_dt.
build_neighbor_edges <- function(cell_dt, id_order, neighbors) {
  # Step 1: Build spatial edge list (cell-id level, ~1.37M rows)
  edges <- nb_to_edge_dt(id_order, neighbors)

  # Step 2: Build a lookup from (id, year) -> row index in cell_dt
  cell_dt[, row_idx := .I]

  # Step 3: Cross with years via join on focal_id
  #   For each edge (focal_id, neighbor_id), we need all years
  #   that the focal_id appears in. We get the year from the focal row.
  #   Then the neighbor must also appear in that same year.

  # Keyed lookup tables
  focal_lookup <- cell_dt[, .(focal_id = id, year, focal_row = row_idx)]
  setkey(focal_lookup, focal_id)

  neighbor_lookup <- cell_dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(neighbor_lookup, neighbor_id, year)

  # Join edges with focal rows to expand across years
  setkey(edges, focal_id)
  edge_year <- edges[focal_lookup, on = "focal_id",
                     nomatch = 0L,
                     allow.cartesian = TRUE]
  # edge_year now has: focal_id, neighbor_id, year, focal_row

  # Join to get neighbor_row for the same (neighbor_id, year)
  edge_year[neighbor_lookup,
            on = c("neighbor_id", "year"),
            neighbor_row := i.neighbor_row]

  # Drop edges where the neighbor cell-year doesn't exist
  edge_year <- edge_year[!is.na(neighbor_row)]

  # Return only the columns we need
  edge_year[, .(focal_row, neighbor_row)]
}

#' Compute neighbor max, min, mean for a single variable,
#' and add the three columns to cell_dt (by reference).
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, edge_dt) {
  # Attach the neighbor's value to each edge
  edge_dt[, val := cell_dt[[var_name]][neighbor_row]]

  # Remove edges where the neighbor value is NA
  valid <- edge_dt[!is.na(val)]

  # Grouped aggregation: one group per focal_row
  agg <- valid[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = focal_row]

  # Initialize result columns with NA
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = col_max,  value = NA_real_)
  set(cell_dt, j = col_min,  value = NA_real_)
  set(cell_dt, j = col_mean, value = NA_real_)

  # Fill in computed values at the correct rows
  set(cell_dt, i = agg$focal_row, j = col_max,  value = agg$nb_max)
  set(cell_dt, i = agg$focal_row, j = col_min,  value = agg$nb_min)
  set(cell_dt, i = agg$focal_row, j = col_mean, value = agg$nb_mean)

  # Clean up temporary column in edge_dt
  edge_dt[, val := NULL]

  invisible(cell_dt)
}

# ============================================================
# MAIN PIPELINE (drop-in replacement for the original outer loop)
# ============================================================

# Convert cell_data to data.table (if not already)
cell_dt <- as.data.table(cell_data)

# Build the vectorized edge table (~38.5M rows, ~1-3 min)
message("Building neighbor edge table...")
edge_dt <- build_neighbor_edges(cell_dt, id_order, rook_neighbors_unique)
message(sprintf("Edge table: %s rows", format(nrow(edge_dt), big.mark = ",")))

# Compute neighbor features for each source variable (~2-5 min total)
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("Computing neighbor stats for: %s", var_name))
  compute_and_add_neighbor_features_fast(cell_dt, var_name, edge_dt)
}

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(cell_dt)

# Clean up helper column
cell_data$row_idx <- NULL

message("Done. Neighbor features added. Random Forest model is unchanged.")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M `lapply` iterations with string key lookups into a 6.46M-entry named vector | One vectorized `data.table` join (~38.5M edge rows built in seconds) |
| **Stat computation** | 6.46M `lapply` iterations Ã— 5 variables, each calling `max`/`min`/`mean` on small vectors | 5 grouped `data.table` aggregations over the pre-built edge table |
| **Final binding** | `do.call(rbind, list_of_6.46M_vectors)` | In-place `set()` by reference |
| **Estimated runtime** | ~86+ hours | ~5â€“10 minutes |
| **RAM** | ~16 GB (marginal) | ~6â€“8 GB for edge table + data (fits in 16 GB) |
| **Trained RF model** | Preserved âœ“ | Preserved âœ“ |
| **Numerical result** | max, min, mean of rook-neighbor values | Identical âœ“ |
END_RESPONSE>>>
