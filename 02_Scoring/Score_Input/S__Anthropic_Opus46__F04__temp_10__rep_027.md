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
run_id: Anthropic_F04_T10_R027
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 27
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_027.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is overwhelmingly in `build_neighbor_lookup`, not in `compute_neighbor_stats` or the Random Forest. Specifically:

1. **`build_neighbor_lookup` uses `lapply` over ~6.46 million rows.** Each iteration performs character coercion, `paste`-based key construction, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash probing per call, and constructing ~6.46M Ã— ~4 average neighbors â‰ˆ ~26M `paste` and name-match operations inside an interpreted loop. This is the dominant cost.

2. **`compute_neighbor_stats` is a secondary bottleneck.** It loops over 6.46M entries, subsetting a numeric vector each time. While cheaper per iteration, `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself expensive (repeated memory allocation).

3. **The `paste`-based keying strategy is fundamentally slow.** Building string keys for every cell-year-neighbor combination creates massive temporary character vectors and relies on R's slow string hashing.

**Root cause summary:** An interpreted R loop over 6.46 million rows doing string operations and named-vector lookups for each row, repeated once for building the lookup and then once per variable for stats computation.

---

## Optimization Strategy

### Principle: Replace string keys and R-level loops with vectorized integer-indexed operations using `data.table`.

**Key ideas:**

1. **Replace the string-keyed lookup with integer join.** Instead of `paste(id, year)` keys, assign each row an integer row index and build a `data.table` with `(id, year) â†’ row_index` for O(1) keyed joins.

2. **Vectorize `build_neighbor_lookup`.** Expand all neighbor relationships into a flat edge table `(row_i, neighbor_id, year)`, then batch-join to resolve `neighbor_id + year â†’ row_j` in one vectorized `data.table` merge. This eliminates the 6.46M-iteration `lapply`.

3. **Vectorize `compute_neighbor_stats`.** Once we have a flat `(row_i, row_j)` edge list, computing `max/min/mean` of neighbor values is a single grouped aggregation: `edges[, .(max, min, mean), by = row_i]`.

4. **Memory check:** The flat edge list will have ~6.46M rows Ã— ~4 neighbors â‰ˆ ~26M rows of two integers â‰ˆ ~200 MB. Feasible on 16 GB RAM.

5. **Preserve the trained RF model and numerical outputs exactly.** We compute identical `max`, `min`, `mean` statistics â€” just faster.

---

## Working R Code

```r
library(data.table)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 0: Convert cell_data to data.table (if not already) and add row index
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
setDT(cell_data)
cell_data[, .row_idx := .I]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 1: Build a flat directed edge table from the nb object
#         (one-time cost, fully vectorized)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
build_neighbor_edges_dt <- function(cell_data, id_order, neighbors) {
  # Map: position in id_order â†’ cell id
  # neighbors[[k]] gives the positions (in id_order) that are neighbors of id_order[k]

  n_cells <- length(id_order)

  # Expand nb list into a flat edge table: (from_id, to_id)
  # from_pos = index in id_order, to_pos = neighbor indices in id_order
  from_pos <- rep(seq_len(n_cells), lengths(neighbors))
  to_pos   <- unlist(neighbors)

  # Remove zero-length / empty neighbor entries (spdep convention: 0L means no neighbors)
  valid <- to_pos > 0L
  from_pos <- from_pos[valid]
  to_pos   <- to_pos[valid]

  edges_cell <- data.table(
    from_id = id_order[from_pos],
    to_id   = id_order[to_pos]
  )

  # Get unique years
  years <- sort(unique(cell_data$year))

  # Cross-join edges with years to get all (from_id, year, to_id) combinations
  # This is the full set of potential neighbor lookups
  edges_full <- edges_cell[, .(year = years), by = .(from_id, to_id)]

  # Build keyed lookup: (id, year) â†’ .row_idx
  row_key <- cell_data[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # Resolve from_id,year â†’ row_i
  edges_full[, row_i := row_key[.(from_id, year), .row_idx, on = .(id, year)]]

  # Resolve to_id,year â†’ row_j (the neighbor's row)
  edges_full[, row_j := row_key[.(to_id, year), .row_idx, on = .(id, year)]]

  # Drop unresolved (NA) â€” cells/years not present in the data
  edges_resolved <- edges_full[!is.na(row_i) & !is.na(row_j), .(row_i, row_j)]

  setkey(edges_resolved, row_i)
  return(edges_resolved)
}

cat("Building edge table...\n")
system.time({
  edge_dt <- build_neighbor_edges_dt(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~26M rows, completes in seconds to low minutes.

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 2: Vectorized neighbor stat computation + feature attachment
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_dt) {
  # Extract the variable values indexed by row
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the edge table
  edge_dt[, nval := vals[row_j]]

  # Grouped aggregation â€” single vectorized pass
  stats <- edge_dt[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    by = row_i
  ]

  # Initialize new columns with NA
  max_col  <- paste0(var_name, "_nb_max")
  min_col  <- paste0(var_name, "_nb_min")
  mean_col <- paste0(var_name, "_nb_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign by reference using integer row indices
  cell_data[stats$row_i, (max_col)  := stats$nb_max]
  cell_data[stats$row_i, (min_col)  := stats$nb_min]
  cell_data[stats$row_i, (mean_col) := stats$nb_mean]

  # Clean up temporary column from edge_dt
  edge_dt[, nval := NULL]

  invisible(cell_data)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 3: Outer loop â€” compute neighbor features for all source vars
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat("  Processing:", var_name, "\n")
    compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  }
})

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Step 4: Drop helper column, apply the existing trained RF model
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data[, .row_idx := NULL]

# The trained Random Forest model is unchanged â€” predict as before:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Expected Performance Improvement

| Phase | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` (6.46M `lapply` + string ops) | ~70â€“80 hours | Replaced by `build_neighbor_edges_dt`: **2â€“5 minutes** |
| `compute_neighbor_stats` Ã— 5 vars (6.46M `lapply` + `do.call(rbind)`) | ~6â€“10 hours | Replaced by 5 Ã— `data.table` grouped aggregation: **1â€“3 minutes** |
| **Total neighbor feature construction** | **~86 hours** | **~5â€“8 minutes** |

The speedup comes from eliminating ~32 million interpreted R-level string operations and replacing them with `data.table`'s radix-key joins and grouped aggregations, which execute in compiled C code. The numerical outputs (max, min, mean of non-NA neighbor values per row) are identical, so the trained Random Forest model requires no changes.
END_RESPONSE>>>
