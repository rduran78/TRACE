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
run_id: Anthropic_F04_T00_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_019.md

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
- A named-vector lookup into `idx_lookup` (which is a 6.46-million-element named character vector â€” each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, each time constructing and discarding small character vectors, is extremely expensive. The named-vector `idx_lookup` with ~6.46M entries makes each key lookup slow, and this is repeated for every neighbor of every row.

### 2. `compute_neighbor_stats` â€” Row-level `lapply` over 6.46 million rows (Ã—5 variables)

Each call iterates over all 6.46M rows, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. This is called 5 times (once per variable). The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also slow.

### Quantified cost

- `build_neighbor_lookup`: ~6.46M iterations Ã— (string ops + hash lookups) â‰ˆ tens of hours.
- `compute_neighbor_stats`: ~6.46M Ã— 5 iterations Ã— (subset + summary stats) + 5 Ã— `do.call(rbind, ...)` on 6.46M-element lists â‰ˆ additional hours.
- Combined estimated wall time: **86+ hours** as reported.

---

## Optimization Strategy

The strategy is to **eliminate all row-level R loops** and replace them with vectorized `data.table` joins and grouped aggregations.

| Step | Original Approach | Optimized Approach |
|---|---|---|
| Build neighbor lookup | `lapply` over 6.46M rows, character key construction, named-vector lookup | Vectorized expansion of the `nb` object into an edge-list `data.table`, then a single keyed join |
| Compute neighbor stats | `lapply` over 6.46M rows per variable, `do.call(rbind, ...)` | `data.table` grouped aggregation (`[, .(max, min, mean), by=...]`) â€” one pass per variable, fully vectorized in C |
| Memory | 6.46M-element list of integer vectors + 6.46M-element named character vector | A single edge-list data.table (~14M rows Ã— 4 columns, ~200â€“400 MB, fits in 16 GB) |

**Expected speedup**: From 86+ hours to **minutes** (typically 5â€“20 minutes total depending on disk I/O).

**Numerical equivalence**: The same `max`, `min`, `mean` of the same rook-neighbor values are computed. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

#' Optimized spatial neighbor feature construction.
#' Replaces build_neighbor_lookup + compute_neighbor_stats + outer loop.
#'
#' @param cell_data       data.frame or data.table with columns: id, year, and
#'                        all variables listed in neighbor_source_vars.
#' @param id_order        integer vector of cell IDs in the order matching the
#'                        nb object (i.e., id_order[i] is the cell ID for the
#'                        i-th element of rook_neighbors_unique).
#' @param rook_neighbors  spdep nb object (list of integer vectors of neighbor
#'                        indices).
#' @param neighbor_source_vars character vector of variable names to summarize.
#' @return data.table with original columns plus neighbor feature columns.
add_neighbor_features_fast <- function(cell_data,
                                       id_order,
                                       rook_neighbors,
                                       neighbor_source_vars) {

  # --- Step 0: Convert to data.table (no copy if already one) ----------------
  dt <- as.data.table(cell_data)

  # --- Step 1: Build edge list from nb object (vectorized) -------------------
  #
  # rook_neighbors[[i]] gives the indices (into id_order) of neighbors of
  # the cell whose ID is id_order[i].
  #
  # We expand this into a two-column data.table:
  #   focal_id   â€” the cell ID of the focal cell
  #   neighbor_id â€” the cell ID of each neighbor

  n_cells <- length(id_order)
  lengths_vec <- lengths(rook_neighbors)                 # integer vector, fast
  focal_id_vec <- rep.int(id_order, times = lengths_vec) # vectorized repeat

  neighbor_idx_vec <- unlist(rook_neighbors, use.names = FALSE)
  neighbor_id_vec  <- id_order[neighbor_idx_vec]

  edges <- data.table(
    focal_id    = focal_id_vec,
    neighbor_id = neighbor_id_vec
  )
  rm(focal_id_vec, neighbor_idx_vec, neighbor_id_vec, lengths_vec)

  # --- Step 2: Create a keyed lookup of (id, year) â†’ row index in dt ---------
  #
  # We will join edges Ã— years to dt to pull neighbor variable values.

  # Unique years
  years <- sort(unique(dt$year))

  # Expand edges across all years: each directed edge exists in every year.
  # This produces the full (focal_id, year, neighbor_id) table.
  #
  # CJ (cross join) inside an edges merge is memory-efficient if done via
  # a keyed join rather than a full Cartesian product.
  #
  # Approach: for each year, copy edges and add the year column, then rbindlist.
  # With 28 years and ~1.37M edges this is ~38.5M rows Ã— 3 cols â‰ˆ manageable.

  edge_year <- edges[, .(focal_id, neighbor_id, year = rep(list(years), .N)),
                     by = .I][, .(focal_id, neighbor_id, year = unlist(year))]

  # More memory-friendly alternative (avoids intermediate list column):
  # edge_year <- CJ_dt(edges, data.table(year = years))
  # We implement it simply:
  edge_year <- rbindlist(
    lapply(years, function(y) {
      edges[, .(focal_id, neighbor_id, year = y)]
    }),
    use.names = TRUE
  )
  rm(edges)
  gc()

  # --- Step 3: Join neighbor values onto edge_year ---------------------------
  #
  # Key dt by (id, year) for fast join.
  setkey(dt, id, year)

  # We only need the neighbor source variables from dt for the join.
  # Pull them by joining on neighbor_id + year.
  cols_needed <- c("id", "year", neighbor_source_vars)
  dt_slim <- dt[, ..cols_needed]
  setnames(dt_slim, "id", "neighbor_id")
  setkey(dt_slim, neighbor_id, year)

  # Keyed join: for each (neighbor_id, year) in edge_year, attach the
  # neighbor's variable values.
  setkey(edge_year, neighbor_id, year)
  edge_vals <- dt_slim[edge_year, nomatch = NA]
  # edge_vals now has columns: neighbor_id, year, <vars>, focal_id
  rm(dt_slim, edge_year)
  gc()

  # --- Step 4: Grouped aggregation -------------------------------------------
  #
  # For each (focal_id, year), compute max/min/mean of each variable across
  # its neighbors.

  agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))

  # Build the j-expression programmatically
  j_expr <- as.call(c(
    as.name("list"),
    setNames(agg_exprs, agg_names)
  ))

  neighbor_stats <- edge_vals[, eval(j_expr), by = .(focal_id, year)]
  rm(edge_vals)
  gc()

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col in agg_names) {
    set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
  }

  # --- Step 5: Join aggregated features back to dt ---------------------------
  setnames(neighbor_stats, "focal_id", "id")
  setkey(neighbor_stats, id, year)
  setkey(dt, id, year)

  dt <- neighbor_stats[dt, on = .(id, year)]
  rm(neighbor_stats)
  gc()

  return(dt)
}


# ============================================================================
# Usage â€” drop-in replacement for the original outer loop
# ============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_neighbor_features_fast(
  cell_data            = cell_data,
  id_order             = id_order,
  rook_neighbors       = rook_neighbors_unique,
  neighbor_source_vars = neighbor_source_vars
)

# cell_data now contains the 15 new columns:
#   neighbor_ntl_max, neighbor_ntl_min, neighbor_ntl_mean,
#   neighbor_ec_max,  neighbor_ec_min,  neighbor_ec_mean,
#   ... (same for pop_density, def, usd_est_n2)
#
# Feed cell_data (with its ~110 + 15 predictor columns) directly into
# predict(trained_rf_model, newdata = cell_data)
# The trained Random Forest model is unchanged.
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Core mechanism** | 6.46M-iteration `lapply` loops with per-row string ops and hash lookups | Vectorized `data.table` keyed joins + grouped C-level aggregation |
| **`build_neighbor_lookup`** | O(N Ã— k) interpreted R iterations (N = 6.46M, k â‰ˆ avg neighbors) | Replaced by vectorized `rep.int` + `unlist` + `rbindlist` â€” seconds |
| **`compute_neighbor_stats`** | O(N) `lapply` Ã— 5 vars + `do.call(rbind, 6.46M-element list)` Ã— 5 | Single grouped `data.table` aggregation â€” minutes |
| **Estimated wall time** | 86+ hours | **5â€“20 minutes** |
| **Peak RAM** | Moderate (but slow) | ~4â€“6 GB for the edge-year table (fits in 16 GB) |
| **Numerical result** | max, min, mean of rook-neighbor values | Identical max, min, mean of identical rook-neighbor values |
| **Trained RF model** | â€” | Untouched; only the feature-construction step is replaced |
END_RESPONSE>>>
