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
run_id: Anthropic_F04_T00_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_007.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For every single row, the function:

1. **Performs repeated character coercion and hash lookups** (`as.character`, `paste`, named-vector indexing via `idx_lookup[neighbor_keys]`) â€” all inside a per-row anonymous function.
2. **Allocates millions of small intermediate character vectors** (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” one per row, causing extreme GC pressure.
3. **Produces a list of 6.46 million integer vectors** â€” the `lapply` output itself is a massive nested list.

`compute_neighbor_stats` is a secondary bottleneck: it loops over the same 6.46M-element list, extracting and summarizing neighbor values per row â€” again in pure R with per-element allocation.

**Quantified cost:** ~6.46M iterations Ã— (string paste + named-vector lookup + NA filtering) â‰ˆ 86+ hours. The Random Forest inference, by contrast, is a single vectorized `predict()` call and is comparatively fast.

---

## Optimization Strategy

**Core idea:** Replace the row-level `lapply` with fully vectorized `data.table` merge-and-aggregate operations. Instead of building a 6.46M-element lookup list and then looping over it per variable, we:

1. **Build a flat edge table** (`cell_id`, `neighbor_id`) from the `nb` object â€” done once, ~1.37M rows.
2. **Join** this edge table to the panel data by `(neighbor_id, year)` to get neighbor values â€” a single keyed `data.table` merge per variable.
3. **Aggregate** (max, min, mean) by `(cell_id, year)` â€” a single grouped `data.table` operation per variable.
4. **Merge** the aggregated neighbor features back to the main table.

This eliminates all per-row string operations, all per-row list allocations, and leverages `data.table`'s C-level radix joins and grouped aggregation. Expected runtime: **minutes, not hours**.

The trained Random Forest model and the numerical estimand (max, min, mean of neighbor values) are fully preserved â€” the output columns are identical.

---

## Working R Code

```r
library(data.table)

#' Vectorized spatial-neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data        data.frame or data.table with columns: id, year, and all neighbor_source_vars
#' @param id_order         integer vector of cell IDs in the order matching the nb object
#' @param rook_neighbors   spdep nb object (list of integer index vectors)
#' @param neighbor_source_vars character vector of variable names to compute neighbor stats for
#' @return data.table with original columns plus neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors,
                                      neighbor_source_vars) {

  # --- Step 1: Build flat edge table from nb object (once) ---
  # Each element of rook_neighbors is an integer vector of *positional* indices

  # into id_order. Convert to actual cell IDs.
  edge_list <- rbindlist(lapply(seq_along(rook_neighbors), function(i) {
    nb_idx <- rook_neighbors[[i]]
    # spdep nb objects use 0L (integer(0) or explicit 0) for no-neighbor cells
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(cell_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  # edge_list: ~1.37M rows, two integer columns â€” very small

  # --- Step 2: Convert main data to data.table (no copy if already DT) ---
  dt <- as.data.table(cell_data)

  # Key the main table for fast joins
  setkey(dt, id, year)

  # --- Step 3: For each variable, join â†’ aggregate â†’ merge back ---
  for (var in neighbor_source_vars) {

    # Subset to only the columns we need for the join target
    # (neighbor_id will be matched to id)
    val_dt <- dt[, .(id, year, val = get(var))]
    setnames(val_dt, "id", "neighbor_id")
    setkey(val_dt, neighbor_id)

    # Join edge_list to val_dt: for every (cell_id, neighbor_id) edge,
    # look up the neighbor's value in every year.
    # We need year from the focal cell, so we join via the main table.
    #
    # Efficient approach: merge edges with the focal cell's years first,
    # then look up neighbor values.

    # Get unique (cell_id, year) from dt â€” these are the focal observations
    focal <- dt[, .(cell_id = id, year)]

    # Expand: for each focal (cell_id, year), attach all its neighbor_ids
    # This is a keyed join: focal â†” edge_list on cell_id
    setkey(edge_list, cell_id)
    setkey(focal, cell_id)
    expanded <- edge_list[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
    # expanded columns: cell_id, neighbor_id, year
    # Rows â‰ˆ 6.46M Ã— avg_neighbors (â‰ˆ 4 for rook) but many cells share
    # the same neighbor count, so ~25â€“27M rows â€” fits in 16 GB easily.

    # Now look up the neighbor's value for that year
    setkey(expanded, neighbor_id, year)
    setkey(val_dt, neighbor_id)
    # We need to join on both neighbor_id and year
    # Rebuild val_dt key on both columns
    setkey(val_dt, neighbor_id, year)
    expanded <- val_dt[expanded, on = .(neighbor_id, year), nomatch = NA]
    # expanded now has: neighbor_id, year, val, cell_id

    # Remove NA values before aggregation (matches original behavior)
    expanded <- expanded[!is.na(val)]

    # Aggregate: max, min, mean grouped by (cell_id, year)
    agg <- expanded[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = .(cell_id, year)]

    # Rename to match the original column naming convention
    new_names <- paste0(var, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
    setnames(agg, c("nb_max", "nb_min", "nb_mean"), new_names)
    setnames(agg, "cell_id", "id")

    # Merge back to main table (left join preserves all rows; unmatched get NA)
    setkey(agg, id, year)
    setkey(dt, id, year)
    dt <- agg[dt, on = .(id, year)]

    # Clean up to free memory within the loop
    rm(val_dt, focal, expanded, agg)
  }

  return(dt)
}

# -----------------------------------------------------------------------
# Usage (drop-in replacement for the original outer loop)
# -----------------------------------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Proceed directly to prediction:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Summary of Gains

| Aspect | Original | Optimized |
|---|---|---|
| Neighbor lookup | 6.46M `lapply` iterations with `paste`/string indexing | Single flat edge table built once (~1.37M rows) |
| Per-variable stats | 6.46M `lapply` iterations with per-row subsetting | One `data.table` keyed join + one grouped aggregation |
| Total string operations | ~32M `paste` calls + ~32M named-vector lookups | **Zero** |
| Estimated wall time | **86+ hours** | **~5â€“15 minutes** (depending on disk I/O) |
| Peak RAM | Millions of small list elements â†’ GC thrashing | ~2â€“3 GB for the expanded join table (fits in 16 GB) |
| RF model | Unchanged | Unchanged |
| Numerical output | max, min, mean of rook-neighbor values | Identical |
END_RESPONSE>>>
