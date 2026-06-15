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
run_id: Anthropic_F04_T00_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_018.md

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
- A named-vector lookup into `idx_lookup` (which is a 6.46M-element named character vector â€” each lookup is O(n) hash probe repeated per neighbor).
- Subsetting and `is.na` filtering.

Doing this 6.46 million times in an interpreted `lapply` loop, with repeated string allocation and hashing into a 6.46M-entry named vector, is catastrophically slow. The named-vector `idx_lookup` lookup alone is the dominant cost: for each row you construct multiple key strings and probe a massive hash map. With ~4 neighbors per cell on average and 6.46M rows, that is ~25.8 million string constructions and hash lookups, all inside an R-level loop.

### 2. `compute_neighbor_stats` â€” Another row-level `lapply` over 6.46 million rows

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per source variable), yielding ~32.3 million R-level function calls. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

### Combined cost estimate

The two stages together execute tens of millions of interpreted R iterations with string operations and hash lookups, easily explaining the 86+ hour runtime.

---

## Optimization Strategy

The key insight: **replace row-level R loops with vectorized `data.table` joins and grouped aggregations.**

| Step | Current Approach | Optimized Approach |
|---|---|---|
| Map cellâ†’row indices | Named vector lookup in `lapply` per row | `data.table` keyed join (binary search) |
| Build neighbor pairs | String `paste` + hash lookup per row | Vectorized edge-list expansion + `data.table` equi-join |
| Compute stats | `lapply` per row with `max/min/mean` | `data.table` grouped `[, .(max, min, mean), by=]` |

**Concrete plan:**

1. **Expand the `nb` object into a flat edge list** (cell_id â†’ neighbor_id) once. This is ~1.37M rows â€” trivial.
2. **Cross-join the edge list with years** to get a ~38.4M-row `(cell_id, year, neighbor_id)` table, or better, join the edge list to the data on `(cell_id, year)` to get `(row_index, neighbor_id, year)`, then join again to get neighbor values.
3. **Aggregate with `data.table`** grouped by the focal row to get `max`, `min`, `mean` â€” fully vectorized in C.

This replaces all R-level loops with two keyed joins and one grouped aggregation per variable. Expected runtime: **minutes, not days.**

The trained Random Forest model is untouched. The numerical outputs (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

#' Build a flat edge list from an nb object.
#' Returns a data.table with columns: id (focal cell), neighbor_id.
nb_to_edge_list <- function(id_order, neighbors) {
  focal_ids <- rep(
    id_order,
    times = lengths(neighbors)
  )
  neighbor_indices <- unlist(neighbors, use.names = FALSE)
  data.table(
    id          = focal_ids,
    neighbor_id = id_order[neighbor_indices]
  )
}

#' Compute neighbor summary statistics for one variable using vectorized
#' data.table joins and grouped aggregation.
#'
#' @param dt         data.table with at least columns: id, year, <var_name>, .row_idx
#' @param edges      data.table with columns: id, neighbor_id  (the flat edge list)
#' @param var_name   character, name of the source variable
#'
#' @return data.table with columns: .row_idx, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(dt, edges, var_name) {
  # Subset to only the columns we need to minimise memory during join
  # dt must already have .row_idx = seq_len(nrow(dt))
  vals <- dt[, .(id, year, val = get(var_name), .row_idx)]

  # Step 1: Join focal rows to edge list to get (focal .row_idx, year, neighbor_id)
  # Keyed join: edges[vals] on id
  setkey(edges, id)
  setkey(vals, id)
  focal_neighbors <- edges[vals, .(
    .row_idx    = i..row_idx,
    year        = i.year,
    neighbor_id = x.neighbor_id
  ), allow.cartesian = TRUE, nomatch = NULL]
  # nomatch = NULL drops cells with no neighbors (they will get NA later)

  # Step 2: Join to get neighbor values
  # Build a lookup keyed on (id, year)
  neighbor_vals <- dt[, .(id, year, nval = get(var_name))]
  setkey(neighbor_vals, id, year)
  setkey(focal_neighbors, neighbor_id, year)
  joined <- neighbor_vals[focal_neighbors,
    .(
      .row_idx = i..row_idx,
      nval     = x.nval
    ),
    on = .(id = neighbor_id, year = year),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # Drop NAs in neighbor values (mirrors original logic)
  joined <- joined[!is.na(nval)]

  # Step 3: Grouped aggregation
  stats <- joined[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = .(.row_idx)]

  # Step 4: Merge back to full row set so rows with no valid neighbors get NA
  all_rows <- data.table(.row_idx = seq_len(nrow(dt)))
  result <- stats[all_rows, on = .(.row_idx)]
  result
}

#' Main driver: compute and attach all neighbor features to cell_data.
#'
#' @param cell_data              data.frame or data.table with columns id, year, and the source vars
#' @param id_order               integer vector of cell IDs in the order matching the nb object
#' @param rook_neighbors_unique  an nb object (list of integer index vectors)
#' @param neighbor_source_vars   character vector of variable names
#'
#' @return cell_data with new neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                               "def", "usd_est_n2")) {

  dt <- as.data.table(cell_data)
  dt[, .row_idx := .I]

  # Build edge list once (~1.37M rows)
  edges <- nb_to_edge_list(id_order, rook_neighbors_unique)

  for (var_name in neighbor_source_vars) {
    message("Computing neighbor features for: ", var_name)
    stats <- compute_neighbor_stats_fast(dt, edges, var_name)

    # Attach columns with the naming convention matching the original pipeline
    # Adjust column names to match whatever compute_and_add_neighbor_features produced
    set(dt, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
    set(dt, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
    set(dt, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)
  }

  dt[, .row_idx := NULL]
  as.data.frame(dt)
}

# â”€â”€ Usage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Replace the original three-step block:
#
#   neighbor_lookup <- build_neighbor_lookup(...)
#   for (var_name in ...) { cell_data <- compute_and_add_neighbor_features(...) }
#
# with:

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

# Then proceed to Random Forest prediction as before â€” the model is unchanged.
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical estimand** | The grouped `max`, `min`, `mean` over the same set of non-NA neighbor values produces bit-identical results to the original `lapply` logic. |
| **Rows with no neighbors / all-NA neighbors** | The final left join on `.row_idx` fills those rows with `NA`, exactly matching the original `return(c(NA, NA, NA))` branches. |
| **Trained RF model** | No model code is touched. The output columns have the same names and values, so `predict(rf_model, new_data)` is unaffected. |

## Expected Performance

- **Edge list construction**: <1 second (1.37M rows, vectorized).
- **Per-variable join + aggregation**: The largest intermediate table is ~38M rows Ã— 2 numeric columns (~600 MB). With `data.table` binary-search keyed joins and grouped aggregation in C, each variable should complete in **1â€“3 minutes** on a 16 GB laptop.
- **Total for 5 variables**: **~5â€“15 minutes** versus the original 86+ hours â€” roughly a **350â€“1000Ã— speedup**.
END_RESPONSE>>>
