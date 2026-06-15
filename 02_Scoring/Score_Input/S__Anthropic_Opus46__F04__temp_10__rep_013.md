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
run_id: Anthropic_F04_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F04_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, the code:

1. Performs character coercion and named-vector lookup (`id_to_ref`, `idx_lookup`) â€” these are O(1) average-case hash lookups, but the constant factor is large in R because `as.character()`, `paste()`, and named-vector indexing are called **per row** inside an interpreted loop.
2. Builds character key strings (`paste(neighbor_cell_ids, year, "_")`) millions of times.
3. Allocates small intermediate vectors (neighbor keys, NA-filtered results) ~6.46M times, creating massive GC pressure.

**`compute_neighbor_stats`** is a secondary bottleneck: another `lapply` over 6.46M elements, each calling `max`/`min`/`mean` on small vectors. The `do.call(rbind, result)` on a 6.46M-element list is also costly.

**In summary:** The problem is millions of R-level interpreted iterations with per-iteration string allocation and hashing. The algorithm itself is sound; the implementation just needs vectorization.

## Optimization Strategy

1. **Vectorize the neighbor lookup construction** using `data.table` joins instead of per-row `lapply`. Pre-expand all neighbor relationships into a flat edge table, join to get row indices, then group.
2. **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation on the flat edge table â€” no R-level loop at all.
3. **Avoid `lapply` + `do.call(rbind, ...)`** entirely.

Expected speedup: from ~86+ hours to **minutes** (the dominant cost becomes data.table grouped joins/aggregations over ~8â€“10M edge-rows Ã— 28 years).

## Optimized Working R Code

```r
library(data.table)

# ==============================================================
# STEP 1: Build a flat directed edge table from the nb object
#         (one-time cost, independent of year or variable)
# ==============================================================
build_edge_table <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

# ==============================================================
# STEP 2: Build the row-index lookup via a vectorized join
#         Returns a data.table with columns: row_i, neighbor_row_i
# ==============================================================
build_neighbor_lookup_fast <- function(cell_dt, id_order, neighbors) {
  # cell_dt must be a data.table with columns: id, year, and a row index
  # Add row position
  cell_dt[, .row_i := .I]

  # Flat edge table (cell-id to cell-id)
  edges <- build_edge_table(id_order, neighbors)

  # Key the cell data for fast join
  cell_key <- cell_dt[, .(id, year, .row_i)]

  # Join edges to get the focal row index
  # For every (from_id, year) pair, attach the focal row index
  setnames(cell_key, ".row_i", "row_i")
  focal <- cell_key[, .(from_id = id, year, row_i)]
  setkey(focal, from_id, year)

  # Expand edges across all years present in the data
  years <- unique(cell_dt$year)
  edge_year <- CJ_dt(edges, years)   # see helper below

  # Attach focal row
  setkey(edge_year, from_id, year)
  edge_year <- focal[edge_year, nomatch = 0L]

  # Attach neighbor row
  neighbor_key <- cell_key[, .(to_id = id, year, neighbor_row_i = row_i)]
  setkey(neighbor_key, to_id, year)
  setkey(edge_year, to_id, year)
  edge_year <- neighbor_key[edge_year, nomatch = 0L]

  # Return clean edge table: row_i <-> neighbor_row_i
  edge_year[, .(row_i, neighbor_row_i)]
}

# Helper: cross-join edges Ã— years without full CJ on ids
CJ_dt <- function(edges, years) {
  years_dt <- data.table(year = years)
  res <- edges[, .(from_id, to_id)][
    , CJ_year := 1L][
    , CJ_year := NULL]
  # Use a merge-based cross join to avoid massive CJ
  res[, .dummy := 1L]
  years_dt[, .dummy := 1L]
  out <- res[years_dt, on = ".dummy", allow.cartesian = TRUE]
  out[, .dummy := NULL]
  out
}

# ==============================================================
# STEP 3: Compute neighbor statistics fully vectorized
# ==============================================================
compute_and_add_neighbor_features_fast <- function(cell_dt, var_names, edge_lookup) {
  # edge_lookup: data.table with (row_i, neighbor_row_i)
  # Compute all variables at once to reuse the edge table

  for (vn in var_names) {
    # Attach the neighbor's variable value
    edge_lookup[, val := cell_dt[[vn]][neighbor_row_i]]

    # Grouped aggregation â€” one pass per variable, fully vectorized in C
    stats <- edge_lookup[!is.na(val),
      .(
        nbr_max  = max(val),
        nbr_min  = min(val),
        nbr_mean = mean(val)
      ),
      by = row_i
    ]

    # Initialize columns with NA
    max_col  <- paste0("neighbor_max_",  vn)
    min_col  <- paste0("neighbor_min_",  vn)
    mean_col <- paste0("neighbor_mean_", vn)

    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    # Assign computed values by row index
    cell_dt[stats$row_i, (max_col)  := stats$nbr_max]
    cell_dt[stats$row_i, (min_col)  := stats$nbr_min]
    cell_dt[stats$row_i, (mean_col) := stats$nbr_mean]
  }

  invisible(cell_dt)
}

# ==============================================================
# MAIN EXECUTION
# ==============================================================

# Convert to data.table (non-destructive; keeps all columns for RF)
cell_dt <- as.data.table(cell_data)

# Build the vectorized edge lookup (one-time, ~1-3 minutes)
message("Building vectorized edge lookup...")
edge_lookup <- build_neighbor_lookup_fast(cell_dt, id_order, rook_neighbors_unique)
setkey(edge_lookup, row_i)   # speeds grouped aggregation
message(sprintf("Edge lookup: %s rows", format(nrow(edge_lookup), big.mark = ",")))

# Compute all 5 Ã— 3 = 15 neighbor features
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbor statistics...")
compute_and_add_neighbor_features_fast(cell_dt, neighbor_source_vars, edge_lookup)

# Convert back to data.frame if downstream RF predict() expects one
cell_data <- as.data.frame(cell_dt)
cell_data$.row_i <- NULL   # remove helper column

# ---- Prediction with the existing trained RF (unchanged) ----
# e.g., cell_data$pred_gdp <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets; the join logic reproduces the original `paste(id, year)` key matching with NA exclusion. |
| **Trained RF model** | The model object is never touched. The same 15 neighbor feature columns (identical names) are written into `cell_data`, so `predict()` works unchanged. |
| **Column schema** | The output columns (`neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, â€¦) match the original `compute_and_add_neighbor_features` naming convention. Adjust the `paste0` patterns if your wrapper used different names. |

## Expected Performance

| Phase | Original (est.) | Optimized (est.) |
|---|---|---|
| Neighbor lookup build | ~40â€“50 hrs | ~1â€“3 min |
| Stats for 5 variables | ~36â€“40 hrs | ~2â€“5 min |
| **Total** | **~86+ hrs** | **~5â€“10 min** |

The speedup comes from replacing ~6.46M R-interpreter iterations with vectorized `data.table` C-level joins and grouped aggregations. Memory usage peaks at roughly the edge table size (~38M rows Ã— 3 int/numeric columns â‰ˆ 0.9 GB), well within the 16 GB constraint.
END_RESPONSE>>>
