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
run_id: Anthropic_F07_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_009.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Per-row string pasting and named-vector lookup** â€” `paste()` and named-vector indexing (`idx_lookup[neighbor_keys]`) are called ~6.46 million times inside an `lapply`. Each call does string allocation, hashing, and matching.
2. **The lookup is O(n) in list overhead** â€” `idx_lookup` is a named character vector of length ~6.46M. Named lookup on such a vector is O(n) per probe in the worst case (R uses hashing for named vectors, but the sheer number of calls dominates).
3. **`compute_neighbor_stats`** then iterates over the 6.46M-element list again, extracting values one-at-a-time with per-element `max/min/mean`.

The combined cost is roughly **6.46M Ã— (string ops + hash probes per cell's neighbors)**, repeated 5 times for the 5 variables (though the lookup is built once, the stats loop runs 5Ã—). The 86+ hour estimate is almost entirely attributable to the R-level loop in `build_neighbor_lookup` and the per-element overhead in `compute_neighbor_stats`.

### Root causes:
| Issue | Impact |
|---|---|
| Per-row `paste` + named-vector lookup inside `lapply` over 6.46M rows | ~95% of runtime |
| R-level loop in `compute_neighbor_stats` over 6.46M elements | ~4% of runtime |
| Redundant: neighbor topology is year-invariant but rebuilt per-row across all years | Conceptual waste |

## Optimization Strategy

1. **Vectorize the neighbor lookup entirely using `data.table` joins.** Expand the neighbor list (344K cells Ã— ~4 neighbors each â‰ˆ 1.37M directed edges) into an edge table once. Join against the panel on `(neighbor_id, year)` to get row indices. This replaces 6.46M R-level iterations with a single keyed merge â€” seconds instead of days.

2. **Vectorize the stats computation using `data.table` grouped aggregation.** Group the expanded edge table by the focal row index and compute `max`, `min`, `mean` in one pass per variable. This replaces 6.46M `lapply` calls with a single grouped operation.

3. **Memory management.** The edge table expanded across years is ~6.46M Ã— ~4 â‰ˆ 25.8M rows Ã— a few columns â€” well within 16 GB. We process one variable at a time and discard intermediates.

4. **Preserve the trained RF model and numerical estimand.** The output columns have identical names and identical numerical values (IEEE-754 `max`, `min`, `mean` on the same neighbor sets). The RF model is never touched.

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature engineering
# Drop-in replacement for build_neighbor_lookup + compute_neighbor_stats loop
# =============================================================================

library(data.table)

# ---- Step 0: Ensure cell_data is a data.table with a row index ------------
#   (If cell_data is a data.frame, this converts in place without deep copy)
setDT(cell_data)
cell_data[, .row_id := .I]

# ---- Step 1: Build a year-invariant directed edge table --------------------
#   rook_neighbors_unique is an nb object (list of integer vectors of neighbor
#   positions in id_order). We expand it into a two-column table of
#   (focal_cell_id, neighbor_cell_id).

edge_list <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
}))

cat(sprintf("Edge table: %d directed edges\n", nrow(edge_list)))

# ---- Step 2: Expand edges across years via join ----------------------------
#   We need, for every (focal_id, year) row, the row indices of its neighbors
#   in that same year.  We do this with two keyed joins.

# Keyed lookup: (id, year) -> .row_id
id_year_key <- cell_data[, .(id, year, .row_id)]
setkey(id_year_key, id, year)

# Get unique years
years <- sort(unique(cell_data$year))

# Cross-join edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.5M rows)
# This is the largest object; ~38.5M rows Ã— 3 int cols â‰ˆ 0.9 GB â€” fits in 16 GB.
edges_by_year <- CJ_dt_edges <- edge_list[, .(focal_id, neighbor_id)]
edges_by_year <- edges_by_year[, .(year = years), by = .(focal_id, neighbor_id)]

cat(sprintf("Edges Ã— years: %d rows\n", nrow(edges_by_year)))

# Attach focal row index
setkey(edges_by_year, focal_id, year)
edges_by_year[id_year_key, focal_row := i..row_id, on = .(focal_id = id, year)]

# Attach neighbor row index
setkey(edges_by_year, neighbor_id, year)
edges_by_year[id_year_key, neighbor_row := i..row_id, on = .(neighbor_id = id, year)]

# Drop edges where either focal or neighbor is missing (masked cells / boundary)
edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]

cat(sprintf("Valid directed edgesÃ—years: %d\n", nrow(edges_by_year)))

# We only need (focal_row, neighbor_row) from here
edges_by_year <- edges_by_year[, .(focal_row, neighbor_row)]
setkey(edges_by_year, focal_row)

# Free memory
rm(id_year_key, edge_list, CJ_dt_edges)
gc()

# ---- Step 3: Compute neighbor stats per variable ---------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {

  cat(sprintf("Processing neighbor stats for: %s\n", var_name))

  # Pull the variable values aligned to row index
  edges_by_year[, val := cell_data[[var_name]][neighbor_row]]

  # Remove NA values before aggregation (matches original logic)
  valid_edges <- edges_by_year[!is.na(val)]

  # Grouped aggregation â€” single vectorized pass

  stats <- valid_edges[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = .(focal_row)]

  # Initialize output columns to NA (matches original: cells with no valid

  # neighbors get NA)
  max_col  <- paste0(var_name, "_neighbor_max")
  min_col  <- paste0(var_name, "_neighbor_min")
  mean_col <- paste0(var_name, "_neighbor_mean")

  cell_data[, (max_col)  := NA_real_]
  cell_data[, (min_col)  := NA_real_]
  cell_data[, (mean_col) := NA_real_]

  # Assign results back by row index
  cell_data[stats$focal_row, (max_col)  := stats$nb_max]
  cell_data[stats$focal_row, (min_col)  := stats$nb_min]
  cell_data[stats$focal_row, (mean_col) := stats$nb_mean]

  # Clean up per-variable temporaries
  edges_by_year[, val := NULL]
  rm(valid_edges, stats)
  gc()

  cat(sprintf("  Done: %s, %s, %s\n", max_col, min_col, mean_col))
}

# ---- Step 4: Clean up -----------------------------------------------------
cell_data[, .row_id := NULL]
rm(edges_by_year)
gc()

cat("All neighbor features computed.\n")
```

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Build neighbor lookup | ~80+ hours (6.46M R-level iterations with string ops) | ~30â€“90 seconds (one `data.table` cross-join + two keyed joins) |
| Compute stats (5 vars) | ~6+ hours (5 Ã— 6.46M `lapply` calls) | ~30â€“60 seconds (5 Ã— one grouped aggregation) |
| **Total** | **~86+ hours** | **~2â€“5 minutes** |

## Why Numerical Equivalence Is Preserved

- The edge table encodes **exactly** the same directed rook-neighbor relationships from `rook_neighbors_unique`.
- The join on `(id, year)` reproduces exactly the same `paste(id, year)` key matching.
- `NA` filtering before `max/min/mean` matches the original `neighbor_vals[!is.na(neighbor_vals)]`.
- Cells with zero valid neighbors receive `NA` for all three stats, matching the original `c(NA, NA, NA)` return.
- The column names follow the same `{var}_neighbor_{max|min|mean}` convention assumed by `compute_and_add_neighbor_features`.
- The trained Random Forest model is never modified or retrained.
END_RESPONSE>>>
