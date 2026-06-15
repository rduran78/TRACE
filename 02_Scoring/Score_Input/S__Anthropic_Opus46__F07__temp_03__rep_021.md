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
run_id: Anthropic_F07_T03_R021
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 21
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F07_Anthropic_temp_03_rep_021.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **`paste` + named-vector lookup (`idx_lookup[neighbor_keys]`)** â€” For every single row, it constructs character keys and does named-vector indexing. Named-vector lookup in R is O(n) per query against a vector of ~6.46M names. Doing this ~6.46M times yields **O(nÂ²)** complexity â€” roughly 4Ã—10Â¹Â³ character comparisons. This is the 86+ hour wall.

2. **`lapply` over 6.46M rows** in `compute_neighbor_stats` is a secondary cost but is comparatively minor once the lookup is fixed.

3. **Memory**: Storing 6.46M list elements (each a small integer vector) is feasible but wasteful compared to a vectorized/matrix approach.

### Root cause summary

| Component | Problem |
|---|---|
| `idx_lookup[neighbor_keys]` | Named vector lookup is O(n) per call; called ~6.46M times â†’ O(nÂ²) |
| `paste(..., sep="_")` inside the loop | ~6.46M string allocations inside `lapply` |
| `compute_neighbor_stats` with `lapply` | 6.46M R-level iterations; no vectorization |
| Repeated `for` loop over 5 variables | Rebuilds stats 5 times with separate `lapply` passes |

## Optimization Strategy

1. **Replace named-vector lookup with `data.table` hash joins** â€” O(1) amortized per lookup, bringing `build_neighbor_lookup` from O(nÂ²) to O(n).

2. **Eliminate the per-row `lapply` entirely** â€” Expand the neighbor relationships into an edge table (a long-form `data.table` of `(focal_row, neighbor_row)`), then compute all neighbor stats in one vectorized `data.table` grouped aggregation.

3. **Compute all 5 variables in a single pass** over the edge table rather than 5 separate `lapply` loops.

4. **Expected speedup**: From ~86 hours to **~1â€“3 minutes** on a 16 GB laptop.

5. **Numerical equivalence**: The vectorized `max`, `min`, `mean` over the same neighbor sets with the same `na.rm` logic produces bit-identical results. The trained Random Forest model is untouched.

## Working R Code

```r
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optimized neighbor-feature computation using data.table
# Drop-in replacement for the original build_neighbor_lookup +
# compute_neighbor_stats + outer loop.
# Preserves the original numerical estimand exactly.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

library(data.table)

compute_all_neighbor_features <- function(cell_data,
                                          id_order,
                                          rook_neighbors_unique,
                                          neighbor_source_vars) {

  # --- Step 1: Build a long-form edge table of (focal_cell_id, neighbor_cell_id)
  #     from the spdep::nb object.  This is done once (~1.37M edges).

  edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
    nb <- rook_neighbors_unique[[i]]
    nb <- nb[nb > 0L]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
  }))

  # --- Step 2: Convert cell_data to data.table and add a row index.

  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # --- Step 3: Create a keyed lookup:  (id, year) -> row index + variable values.
  #     We only need the neighbor_source_vars columns for the join.

  keep_cols <- c("id", "year", ".row_idx", neighbor_source_vars)
  dt_key <- dt[, ..keep_cols]
  setkey(dt_key, id, year)

  # --- Step 4: For every focal row, get its year, then join to edges to get
  #     neighbor ids, then join again to get neighbor variable values.
  #     All via keyed data.table joins â€” O(n log n) total.

  # Focal side: (focal_id, year, focal_row_idx)
  focal <- dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]

  # Join focal rows to edges to get (focal_row_idx, year, neighbor_id)
  setkey(edges, focal_id)
  setkey(focal, focal_id)
  expanded <- edges[focal, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL,
                    .(focal_row_idx, year = i.year, neighbor_id)]

  # Join to get neighbor variable values in the same year
  setkey(expanded, neighbor_id, year)
  # dt_key is keyed on (id, year); rename for the join
  expanded_with_vals <- dt_key[expanded,
                               on = .(id = neighbor_id, year = year),
                               nomatch = NA]

  # --- Step 5: Grouped aggregation â€” compute max, min, mean for each
  #     (focal_row_idx) across all its neighbors, for all variables at once.

  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  # Build the aggregation call
  stats <- expanded_with_vals[,
    setNames(lapply(agg_exprs, eval, envir = .SD), agg_names),
    by = .(focal_row_idx)
  ]

  # Replace -Inf/Inf (from max/min of zero-length after NA removal) with NA
  for (col in agg_names) {
    set(stats, which(is.infinite(stats[[col]])), col, NA_real_)
  }

  # --- Step 6: Merge back into the original data by row index.

  setkey(stats, focal_row_idx)

  # Ensure original row order
  for (col in agg_names) {
    dt[stats, (col) := get(paste0("i.", col)), on = .(.row_idx = focal_row_idx)]
  }

  # Rows with no neighbors at all won't appear in stats; they stay NA â€” correct.

  dt[, .row_idx := NULL]

  # Return as data.frame to match original pipeline expectations
  as.data.frame(dt)
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Usage (drop-in replacement for the original outer loop)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# cell_data now contains 15 new columns:
#   neighbor_max_ntl, neighbor_min_ntl, neighbor_mean_ntl,
#   neighbor_max_ec,  neighbor_min_ec,  neighbor_mean_ec,
#   ... etc.
#
# The trained Random Forest model is unchanged and can be used directly
# for prediction on the updated cell_data.
```

### Why this is safe and equivalent

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | Same `max`, `min`, `mean` with `na.rm=TRUE` over the identical neighbor sets. The edge table is a lossless representation of `rook_neighbors_unique`. |
| **Irregular topology / masked cells** | The `nomatch = NA` join naturally handles cells with no neighbors or neighbors absent in a given year â€” they become `NA`, exactly as the original code's `is.na` filter did. |
| **Memory** | The expanded edge table is ~(1.37M edges Ã— 28 years) â‰ˆ 38.4M rows Ã— ~7 columns â‰ˆ 2.1 GB, well within 16 GB. Peak usage is transient. |
| **Trained RF model** | Untouched. Only the feature-engineering step is replaced. |
| **Runtime** | ~1â€“3 minutes (data.table hash joins + vectorized grouped aggregation) vs. 86+ hours. |
END_RESPONSE>>>
