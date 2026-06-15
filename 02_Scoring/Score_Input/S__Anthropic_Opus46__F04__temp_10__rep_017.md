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
run_id: Anthropic_F04_T10_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Looks up the cell's reference index via character matching (`id_to_ref[as.character(...)]`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs character keys by pasting cell ID + year (`paste(..., sep="_")`).
4. Looks up row indices from a named character vector (`idx_lookup[neighbor_keys]`).

This means **~6.46 million iterations**, each performing string allocation, concatenation, and named-vector lookup (which is O(n) hashing on character keys). With ~1.37M neighbor relationships spread over 28 years, the total number of string constructions and lookups is enormous (~38+ million paste operations and named-vector accesses). The second function, `compute_neighbor_stats`, is lighter but still uses a per-row `lapply` with R-level looping.

**Root causes:**
- **Row-level R loop** over 6.46M rows â€” no vectorization.
- **Repeated string construction** (`paste`, `as.character`) inside the loop.
- **Named character vector lookup** (`idx_lookup[neighbor_keys]`) is slow at scale compared to integer-keyed hash or merge-based approaches.
- The lookup is **year-invariant in structure** (same neighbor topology every year), yet it's rebuilt per row rather than exploiting the panel structure.

## Optimization Strategy

1. **Vectorized edge-list expansion**: Expand the `nb` object into an edge list (cell_i â†’ cell_j) once. This is only ~1.37M rows.
2. **Integer-keyed merge via `data.table`**: Instead of per-row string lookup, join the edge list with the data on `(neighbor_id, year)` to pull neighbor variable values, then group-aggregate (max, min, mean) in one pass per variable.
3. **Eliminate `build_neighbor_lookup` entirely**: The merge-based approach makes the row-index lookup unnecessary.
4. **Result**: Replaces ~6.46M R-level iterations with a few vectorized `data.table` joins and grouped aggregations â€” expected runtime drops from 86+ hours to **minutes**.

## Optimized R Code

```r
library(data.table)

# ---------------------------------------------------------------
# Step 1: Convert the nb object to a data.table edge list (once)
# ---------------------------------------------------------------
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector of cell IDs aligned with the nb list
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors)

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows: (from_id, to_id)

# ---------------------------------------------------------------
# Step 2: Vectorized neighbor feature computation
# ---------------------------------------------------------------
compute_neighbor_features_fast <- function(cell_data_dt, edge_dt, var_names) {
  # cell_data_dt: data.table with columns id, year, and all var_names
  # edge_dt:      data.table with columns from_id, to_id
  # var_names:    character vector of source variable names

  # Ensure data.table
  if (!is.data.table(cell_data_dt)) cell_data_dt <- as.data.table(cell_data_dt)

  # Add a row key for final ordered join-back
  cell_data_dt[, .row_idx := .I]

  for (vname in var_names) {
    message("Processing neighbor features for: ", vname)

    # Subset the columns we need from the target (neighbor) side
    # Columns: to_id (as id), year, and the variable value
    neighbor_vals <- cell_data_dt[, .(id, year, val = get(vname))]

    # Join edge list with the focal cell to get (from_id, year) pairs,
    # then join with neighbor_vals to get neighbor variable values.
    #
    # Conceptually:
    #   for each (from_id, year) â€” the focal cell-year â€”
    #     look up all to_id from edge_dt,
    #     retrieve val for each (to_id, year) from the data,
    #     compute max, min, mean of those vals.

    # Build the expanded table: (from_id, to_id, year) for every year
    # We do this by joining focal cell-years with the edge list on from_id.
    focal <- cell_data_dt[, .(from_id = id, year, .row_idx)]

    # Keyed join: focal Ã— edge_dt on from_id
    setkey(edge_dt, from_id)
    setkey(focal, from_id)
    expanded <- edge_dt[focal, on = "from_id", allow.cartesian = TRUE, nomatch = NULL]
    # expanded has columns: from_id, to_id, year, .row_idx

    # Now join to get the neighbor's value for (to_id, year)
    setkey(neighbor_vals, id, year)
    setkey(expanded, to_id, year)
    expanded <- neighbor_vals[expanded, on = c("id" = "to_id", "year" = "year"), nomatch = NA]
    # expanded now has: id (=to_id), year, val, from_id, .row_idx

    # Aggregate: group by .row_idx (the focal cell-year row)
    agg <- expanded[!is.na(val),
                    .(nbr_max  = max(val),
                      nbr_min  = min(val),
                      nbr_mean = mean(val)),
                    by = .row_idx]

    # Name the new columns to match original pipeline conventions
    max_col  <- paste0(vname, "_max")
    min_col  <- paste0(vname, "_min")
    mean_col <- paste0(vname, "_mean")

    # Initialize with NA, then fill matched rows
    cell_data_dt[, (max_col)  := NA_real_]
    cell_data_dt[, (min_col)  := NA_real_]
    cell_data_dt[, (mean_col) := NA_real_]

    cell_data_dt[agg, on = ".row_idx",
                 (max_col)  := i.nbr_max]
    cell_data_dt[agg, on = ".row_idx",
                 (min_col)  := i.nbr_min]
    cell_data_dt[agg, on = ".row_idx",
                 (mean_col) := i.nbr_mean]
  }

  cell_data_dt[, .row_idx := NULL]
  return(cell_data_dt)
}

# ---------------------------------------------------------------
# Step 3: Run it
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_neighbor_features_fast(
  cell_data_dt = cell_data,
  edge_dt      = edge_dt,
  var_names    = neighbor_source_vars
)

# cell_data now has 15 new columns (3 stats Ã— 5 variables),
# numerically identical to the original pipeline output.
# The trained Random Forest model is untouched and can be
# applied directly via predict(rf_model, cell_data).
```

## Memory Note for 16 GB RAM

The `allow.cartesian = TRUE` join expands ~1.37M edges Ã— 28 years â‰ˆ 38.4M rows per variable. Each expanded table has ~4 columns of integers/doubles â‰ˆ ~1.2 GB peak per variable iteration. This fits comfortably within 16 GB. The `agg` intermediate is discarded each iteration. If memory is tight, the loop already processes one variable at a time and intermediate objects can be explicitly removed with `rm(expanded, agg); gc()` inside the loop.

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| Lookup construction | 6.46M R-level iterations with string ops | Eliminated entirely |
| Feature computation | 6.46M R-level `lapply` iterations Ã— 5 vars | 5 vectorized `data.table` joins + grouped aggregations |
| Expected runtime | 86+ hours | ~5â€“15 minutes |
| Numerical result | Baseline | Identical (same max/min/mean over same neighbor sets) |
| Trained RF model | Preserved | Preserved (no retraining) |
END_RESPONSE>>>
