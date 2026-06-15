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
run_id: Anthropic_F04_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F04_Anthropic_temp_00_rep_022.md

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
- A named-vector lookup into `idx_lookup` (which is a named character vector of length 6.46M â€” each lookup is **O(n)** in the worst case for R's hashed named vectors, but the sheer repetition across 6.46M rows is devastating).
- Subsetting and `as.integer` conversion.

This produces ~6.46 million list elements. The dominant cost is the **per-row string construction and hash lookup repeated millions of times inside an interpreted R loop**.

### 2. `compute_neighbor_stats` â€” Another `lapply` over 6.46 million list elements

Each call subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called **5 times** (once per source variable), so ~32.3 million R-level function invocations.

### 3. `do.call(rbind, result)` on a 6.46M-element list of 3-vectors

This is a well-known R anti-pattern. Binding millions of small vectors row-by-row is extremely slow.

### Estimated cost breakdown (86+ hours):
| Step | Approx. share |
|---|---|
| `build_neighbor_lookup` (string ops, hash lookups Ã—6.46M) | ~40â€“50% |
| `compute_neighbor_stats` Ã—5 vars (lapply + per-row stats) | ~40â€“50% |
| `do.call(rbind, ...)` Ã—5 | ~5â€“10% |

---

## Optimization Strategy

**Core idea:** Replace all per-row R-level loops and string-key lookups with vectorized `data.table` joins and grouped aggregations.

| Original approach | Optimized approach |
|---|---|
| Build a 6.46M-element list of neighbor row indices via `paste` + named-vector lookup | Build an edge-list `data.table` via vectorized integer join â€” no strings |
| `lapply` over 6.46M rows to compute per-row stats | `data.table` grouped aggregation (`[, .(max, min, mean), by=...]`) on the edge-list â€” fully vectorized in C |
| `do.call(rbind, ...)` on millions of small vectors | Result is already a `data.table`; merge back in one join |
| Runs 5 separate passes with separate `lapply` calls | Runs 5 passes but each pass is a fast vectorized `data.table` operation |

**Expected speedup:** From 86+ hours to **~2â€“10 minutes** on the same laptop. The entire computation becomes a handful of vectorized joins and group-by aggregations over ~20â€“30 million edge-rows (6.46M rows Ã— ~4 neighbors average for rook contiguity).

**Preservation guarantees:**
- The trained Random Forest model is untouched.
- The numerical output (max, min, mean of neighbor values per cell-year per variable) is identical to the original.

---

## Working R Code

```r
library(data.table)

#' Build a fully vectorized edge-list of (row_i, neighbor_row_j) pairs.
#' Replaces build_neighbor_lookup entirely â€” no per-row R loop, no string keys.
#'
#' @param cell_data   data.frame / data.table with columns `id` and `year`
#' @param id_order    integer vector: the cell IDs in the order matching the nb object
#' @param neighbors   spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns: row_i (focal row), row_j (neighbor row)
build_neighbor_edgelist <- function(cell_data, id_order, neighbors) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  # --- Step 1: Build a cell-level edge list (focal_id -> neighbor_id) ----------
  #   This is small: length(id_order) cells, ~4 neighbors each â‰ˆ 1.37M edges.
  n_cells <- length(id_order)
  focal_ref <- rep(seq_len(n_cells),
                   times = vapply(neighbors, length, integer(1)))
  neighbor_ref <- unlist(neighbors, use.names = FALSE)

  cell_edges <- data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[neighbor_ref]
  )
  rm(focal_ref, neighbor_ref)

  # --- Step 2: Map (id, year) -> row_idx via keyed join ----------------------
  #   We cross-join cell_edges with every year, then join to get row indices.
  #   But that would explode memory.  Instead, join twice on dt.

  # Keyed lookup: given (id, year) -> row_idx
  setkey(dt, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # Expand cell_edges Ã— years  (~1.37M edges Ã— 28 years â‰ˆ 38.4M rows)
  # This fits comfortably in 16 GB (38.4M Ã— 3 int cols â‰ˆ 0.9 GB).
  edge_year <- cell_edges[, .(focal_id, neighbor_id, year = rep(list(years), .N)),
                          by = .I][, .(focal_id, neighbor_id,
                                       year = unlist(year, use.names = FALSE))]

  # Join to get focal row index
  edge_year[dt, on = .(focal_id = id, year = year), row_i := i.row_idx]

  # Join to get neighbor row index
  edge_year[dt, on = .(neighbor_id = id, year = year), row_j := i.row_idx]

  # Drop edges where either side has no matching row (boundary / missing year)
  edge_year <- edge_year[!is.na(row_i) & !is.na(row_j),
                         .(row_i, row_j)]

  return(edge_year)
}


#' Compute neighbor max, min, mean for one variable using the edge-list.
#' Replaces compute_neighbor_stats â€” fully vectorized via data.table grouping.
#'
#' @param cell_data  data.frame / data.table with the source variable
#' @param edgelist   data.table with columns row_i, row_j
#' @param var_name   character: name of the variable to aggregate
#' @return data.table with columns: row_i, <var>_max, <var>_min, <var>_mean
compute_neighbor_stats_fast <- function(cell_data, edgelist, var_name) {

  vals <- cell_data[[var_name]]

  # Attach neighbor values
  el <- copy(edgelist)
  el[, nval := vals[row_j]]

  # Drop NAs in neighbor values
  el <- el[!is.na(nval)]

  # Grouped aggregation â€” this is the hot path, executed in C by data.table
  stats <- el[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), keyby = row_i]

  setnames(stats,
           c("nb_max",  "nb_min",  "nb_mean"),
           paste0(var_name, c("_max", "_min", "_mean")))

  return(stats)
}


#' Main driver: compute and attach all neighbor features to cell_data.
#' Drop-in replacement for the original outer loop.
#'
#' @param cell_data              data.frame with columns id, year, and all source vars
#' @param id_order               integer vector matching the nb object
#' @param rook_neighbors_unique  spdep nb object
#' @param neighbor_source_vars   character vector of variable names
#' @return cell_data with new neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars = c("ntl", "ec",
                                                               "pop_density",
                                                               "def",
                                                               "usd_est_n2")) {

  dt <- as.data.table(cell_data)
  dt[, row_idx := .I]

  message("Building neighbor edge-list (vectorized)...")
  edgelist <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge-list: %s edges", format(nrow(edgelist), big.mark = ",")))

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))
    stats <- compute_neighbor_stats_fast(cell_data, edgelist, var_name)

    # Left-join back to dt on row index; rows with no neighbors get NA (matches original)
    new_cols <- setdiff(names(stats), "row_i")
    dt[stats, on = .(row_idx = row_i), (new_cols) := mget(paste0("i.", new_cols))]
  }

  dt[, row_idx := NULL]
  return(as.data.frame(dt))
}


# ---- Usage (drop-in replacement for the original outer loop) ----
# cell_data <- add_all_neighbor_features(
#   cell_data,
#   id_order,
#   rook_neighbors_unique,
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
#
# # Then proceed with Random Forest prediction as before â€” model is unchanged.
# # predictions <- predict(trained_rf_model, newdata = cell_data)
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Lookup construction** | 6.46M-iteration `lapply` with `paste` + named-vector hash | Vectorized `data.table` keyed join; zero R-level row loops |
| **Neighbor stats** | 6.46M-iteration `lapply` Ã— 5 vars = 32.3M R function calls | `data.table` grouped `[, .(max,min,mean), by=row_i]` Ã— 5 vars â€” C-level |
| **Result binding** | `do.call(rbind, <6.46M-element list>)` Ã— 5 | Already columnar; single keyed join back |
| **Estimated runtime** | 86+ hours | ~2â€“10 minutes |
| **Peak RAM** | Moderate (6.46M-element list of int vectors) | ~2â€“3 GB for the 38.4M-row edge-list (fits in 16 GB) |
| **Numerical result** | max, min, mean per cell-year | Identical |
| **Trained RF model** | â€” | Untouched |
END_RESPONSE>>>
