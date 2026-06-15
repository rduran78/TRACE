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
run_id: Anthropic_F07_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**. It creates a list of 6.46 million elements, where each element is built by:

1. **Character key construction and lookup** (`paste`, `setNames`, named-vector indexing) â€” for every single row. Named-vector lookup in R is O(n) per query in the worst case because R's named vectors use linear hashing with potential collisions, and here the lookup table has ~6.46M entries. Doing this 6.46M times yields an effective O(nÂ²) cost.
2. **`lapply` over 6.46M rows in pure R** â€” no vectorization, no batching.
3. **`compute_neighbor_stats` then does another `lapply` over 6.46M elements**, extracting variable-length sublists â€” repeated 5 times (once per source variable).

The combined effect: ~86+ hours is almost entirely spent in the `build_neighbor_lookup` function's per-row string operations and named-vector lookups.

## Optimization Strategy

1. **Replace named-vector lookups with `data.table` hash joins.** A `data.table` keyed join is O(1) amortized per lookup and vectorized in C.

2. **Vectorize the neighbor expansion.** Instead of looping row-by-row, expand *all* neighbor relationships into a single edge table (`from_id`, `to_id`), join with year to get (`from_row`, `to_row`), then compute grouped statistics with `data.table` â€” all in bulk.

3. **Compute all 5 variables' stats in a single grouped aggregation pass** instead of 5 separate `lapply` loops.

4. **Memory-safe:** The edge table will have ~1.37M directed edges Ã— 28 years â‰ˆ 38.5M rows of integers â€” roughly 600 MB, well within 16 GB.

This reduces the runtime from ~86 hours to **minutes**.

## Working R Code

```r
library(data.table)

# â”€â”€ 0. Convert to data.table (non-destructive; keeps original object intact) â”€â”€
cell_dt <- as.data.table(cell_data)

# Ensure row identity is preserved so we can write results back
cell_dt[, .row_id := .I]

# â”€â”€ 1. Build a flat edge table from the nb object (one-time, fast) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   rook_neighbors_unique is an nb object: a list of length 344,208
#   where element i contains integer indices of neighbors of cell i
#   (referring to positions in id_order).
#   id_order is a vector of cell IDs of length 344,208.

edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {

  nb_idx <- rook_neighbors_unique[[i]]
  if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) {
    return(NULL)
  }
  data.table(from_id = id_order[i], to_id = id_order[nb_idx])
}))

cat(sprintf("Edge table: %d directed rook-neighbor pairs\n", nrow(edges)))

# â”€â”€ 2. Join edges with panel years to get row-level edge list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   For every (from_id, year) row, we need the values at (to_id, year).
#   Strategy: join edges Ã— cell_dt twice â€” once for "from" rows, once for
#   "to" (neighbor) rows â€” keyed on (id, year).

setkey(cell_dt, id, year)

# Create the neighbor-value table:
#   For each edge (from_id -> to_id) and each year, pull the neighbor's values.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Expand edges by year (vectorized cross-join with unique years)
years <- sort(unique(cell_dt$year))

# Memory-efficient chunked expansion: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows
edge_year <- edges[, CJ(edge_idx = .I, year = years)]
edge_year[, `:=`(
  from_id = edges$from_id[edge_idx],
  to_id   = edges$to_id[edge_idx]
)]
edge_year[, edge_idx := NULL]

# Attach the .row_id of the "from" cell (the cell that will receive the stats)
setkey(edge_year, from_id, year)
from_rows <- cell_dt[, .(from_id = id, year, from_row = .row_id)]
setkey(from_rows, from_id, year)
edge_year <- from_rows[edge_year, nomatch = 0L]

# Attach the neighbor variable values from the "to" cell
keep_cols <- c("id", "year", neighbor_source_vars)
to_vals <- cell_dt[, ..keep_cols]
setnames(to_vals, "id", "to_id")
setkey(to_vals, to_id, year)
setkey(edge_year, to_id, year)
edge_year <- to_vals[edge_year, nomatch = 0L]

cat(sprintf("Expanded edge-year table: %d rows\n", nrow(edge_year)))

# â”€â”€ 3. Compute grouped neighbor stats in one pass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#   Group by from_row (the receiving cell-year row) and compute max/min/mean
#   for every source variable simultaneously.

agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call programmatically
stats <- edge_year[,
  setNames(lapply(neighbor_source_vars, function(v) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      list(NA_real_, NA_real_, NA_real_)
    } else {
      list(max(vals), min(vals), mean(vals))
    }
  }), neighbor_source_vars),
  by = from_row
]

# The above returns nested lists; a cleaner and faster approach:
stats <- edge_year[, {
  out <- vector("list", length(neighbor_source_vars) * 3L)
  k <- 0L
  for (v in neighbor_source_vars) {
    vals <- get(v)
    vals <- vals[!is.na(vals)]
    if (length(vals) == 0L) {
      out[[k + 1L]] <- NA_real_
      out[[k + 2L]] <- NA_real_
      out[[k + 3L]] <- NA_real_
    } else {
      out[[k + 1L]] <- max(vals)
      out[[k + 2L]] <- min(vals)
      out[[k + 3L]] <- mean(vals)
    }
    k <- k + 3L
  }
  names(out) <- agg_names
  out
}, by = from_row]

cat(sprintf("Aggregated stats for %d cell-year rows\n", nrow(stats)))

# â”€â”€ 4. Write results back into cell_data (preserving original structure) â”€â”€â”€â”€â”€â”€
#   stats is keyed by from_row (= .row_id in cell_dt = row index in cell_data)

setkey(stats, from_row)

# Initialize new columns with NA
for (col_name in agg_names) {
  cell_data[[col_name]] <- NA_real_
}

# Assign computed values to the correct rows
cell_data[stats$from_row, agg_names] <- stats[, ..agg_names]

# â”€â”€ 5. Clean up large temporaries â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
rm(cell_dt, edges, edge_year, from_rows, to_vals, stats)
gc()

cat("Done. Neighbor features added to cell_data.\n")
cat(sprintf("cell_data now has %d columns.\n", ncol(cell_data)))
```

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup method** | Named-vector string match (O(n) per query) | `data.table` keyed join (O(1) amortized) |
| **Iteration** | `lapply` over 6.46M rows Ã— 2 passes | Fully vectorized bulk join + grouped aggregation |
| **Aggregation passes** | 5 separate `lapply` loops (one per variable) | 1 single grouped `data.table` aggregation |
| **Estimated runtime** | 86+ hours | ~5â€“15 minutes |
| **Peak memory** | Moderate (but slow) | ~2â€“4 GB (well within 16 GB) |
| **Trained RF model** | âœ… Untouched | âœ… Untouched |
| **Numerical results** | Baseline | Identical (same max/min/mean, same `na.rm` logic) |

The trained Random Forest model is never referenced or modified. The output columns (`neighbor_max_*`, `neighbor_min_*`, `neighbor_mean_*`) are numerically identical to the original implementation â€” the same neighbor sets, the same `na.rm = TRUE` aggregation, the same handling of cells with no neighbors (all `NA`).
END_RESPONSE>>>
