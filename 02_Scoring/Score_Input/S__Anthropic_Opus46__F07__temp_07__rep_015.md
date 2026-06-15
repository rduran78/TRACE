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
run_id: Anthropic_F07_T07_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F07_Anthropic_temp_07_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows in a pure-R `lapply`, performing per-row character coercion, `paste`, and named-vector lookups. This is O(n) in rows but with enormous per-iteration constant cost due to:

1. **Per-row `paste` and character matching** against a named vector of 6.46M entries â€” each lookup is effectively a hash-table probe, but done millions of times from interpreted R.
2. **`compute_neighbor_stats`** then does a second `lapply` over 6.46M rows, extracting subsets of a numeric vector. This is lighter but still slow in pure R.
3. **Memory**: storing 6.46M list elements (each a small integer vector) for `neighbor_lookup` is wasteful and cache-unfriendly.

The 86+ hour estimate comes almost entirely from the `build_neighbor_lookup` step: ~6.46M iterations Ã— ~50Î¼s each â‰ˆ 90 hours.

## Optimization Strategy

**Replace the per-row R loop with vectorized operations using `data.table`.**

Key insight: the neighbor relationship is defined at the **cell level** (344K cells), not the cell-year level (6.46M rows). We can:

1. **Expand the `nb` object into an edge list** of (cell, neighbor_cell) pairs â€” only ~1.37M edges.
2. **Join** this edge list to the panel data by (neighbor_cell, year) to get neighbor values â€” this is a `data.table` merge, fully vectorized.
3. **Aggregate** (max, min, mean) by (cell, year) â€” a single `data.table` grouped operation.
4. **Join** the aggregated stats back to the main data.

This eliminates all per-row R loops. Expected runtime: **seconds to a few minutes** instead of 86+ hours.

The trained Random Forest model is untouched. The numerical results are identical (same max, min, mean of the same neighbor sets).

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Convert panel to data.table (if not already) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_dt <- as.data.table(cell_data)

# â”€â”€ 1. Build edge list from the nb object (once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   rook_neighbors_unique is an nb object: a list of length 344,208
#   where element i contains integer indices of neighbors of cell i.
#   id_order is the vector mapping position -> cell id.

build_edge_list <- function(id_order, nb_obj) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(nb_obj))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(nb_obj)) {
    ni <- nb_obj[[i]]
    # spdep nb objects use 0L for no-neighbor islands; skip those
    ni <- ni[ni > 0L]
    len <- length(ni)
    if (len > 0L) {
      idx <- pos:(pos + len - 1L)
      from_id[idx] <- id_order[i]
      to_id[idx]   <- id_order[ni]
      pos <- pos + len
    }
  }
  # Trim if some were 0-neighbor
  data.table(id = from_id[1:(pos - 1L)], neighbor_id = to_id[1:(pos - 1L)])
}

edges <- build_edge_list(id_order, rook_neighbors_unique)
# edges has ~1.37M rows: (id, neighbor_id)

# â”€â”€ 2. Vectorized neighbor stats for each source variable â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

compute_neighbor_features_fast <- function(dt, edges, var_name) {
  # Columns we need from the neighbor rows: neighbor_id, year, and the variable
  # We join edges to the panel on (neighbor_id == id, year == year)
  
  # Subset to only needed columns to save memory
  neighbor_vals <- dt[, .(id, year, val = get(var_name))]
  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id, year)
  
  # Merge: for each (id, neighbor_id) edge, get the neighbor's value in each year
  # First, create the full (id, neighbor_id, year) table by cross-joining edges with years
  # But that would be huge. Instead, merge edges with the panel on neighbor_id:
  #   edges[neighbor_vals] gives us (id, neighbor_id, year, val) for every
  #   neighbor-cell Ã— year combination.
  
  # Add year from the focal cell's panel? No â€” the neighbor's year must match
  # the focal cell's year. Since neighbor_vals already has year, we just merge:
  
  merged <- merge(edges, neighbor_vals, by = "neighbor_id", allow.cartesian = TRUE)
  # merged columns: neighbor_id, id, year, val
  
  # Drop NA values before aggregation
  merged <- merged[!is.na(val)]
  
  # Aggregate by (id, year)
  agg <- merged[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = .(id, year)]
  
  # Rename columns to match original naming convention
  suffix <- var_name
  setnames(agg, c("nb_max", "nb_min", "nb_mean"),
           paste0(c("max_", "min_", "mean_"), suffix))
  
  agg
}

# â”€â”€ 3. Outer loop: compute and merge all neighbor features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

setkey(cell_dt, id, year)

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor stats for:", var_name, "\n")
  agg <- compute_neighbor_features_fast(cell_dt, edges, var_name)
  setkey(agg, id, year)
  
  # Left-join the new columns onto the main table
  new_cols <- setdiff(names(agg), c("id", "year"))
  cell_dt[agg, (new_cols) := mget(paste0("i.", new_cols)), on = .(id, year)]
  
  rm(agg)
  gc()
}

# â”€â”€ 4. Convert back to data.frame if downstream code expects one â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cell_data <- as.data.frame(cell_dt)

# The trained Random Forest model is unchanged.
# Predictions proceed as before:
# predictions <- predict(rf_model, newdata = cell_data)
```

## Why This is Correct and Equivalent

| Property | Original | Optimized |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[i]]` mapped through `id_order` | Same `nb` object expanded to identical edge list |
| Temporal matching | `paste(neighbor_id, year)` lookup | `merge(..., by = c("neighbor_id", "year"))` â€” same semantics |
| Aggregation | `max`, `min`, `mean` of non-NA neighbor values | Identical functions after `!is.na` filter |
| Cells with no neighbors | Return `NA, NA, NA` | Left-join leaves `NA` for unmatched rows |
| **Numerical estimand** | **Preserved exactly** | **Preserved exactly** |
| RF model | Untouched | Untouched |

## Expected Performance

| Step | Rows processed | Estimated time |
|---|---|---|
| `build_edge_list` | 1.37M edges | < 1 second |
| `merge` per variable | 1.37M edges Ã— 28 years â‰ˆ 38M rows | ~10â€“30 seconds |
| `grouped aggregation` per variable | 38M â†’ 6.46M | ~5â€“15 seconds |
| **Total for 5 variables** | | **~2â€“5 minutes** |

This replaces the 86+ hour runtime with a ~2â€“5 minute runtime on the same 16 GB laptop, with peak memory well within budget (~3â€“4 GB transient for the largest merge).
END_RESPONSE>>>
