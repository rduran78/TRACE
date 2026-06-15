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
run_id: Anthropic_F04_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F04_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference â€” it is the two spatial neighbor feature construction functions, and specifically the way they interact with R's performance characteristics at scale.

### `build_neighbor_lookup` â€” Primary Bottleneck

This function calls `lapply` over **~6.46 million rows**, and inside each iteration it:

1. Performs **character coercion and named-vector lookup** (`id_to_ref[as.character(data$id[i])]`) â€” O(1) per call but with high constant overhead due to repeated `as.character()` conversion and name-based hash lookups across millions of iterations.
2. **Pastes** neighbor cell IDs with the year to create string keys (`paste(neighbor_cell_ids, data$year[i], sep = "_")`) â€” this allocates a new character vector per row.
3. Looks up those keys in `idx_lookup` â€” another named-vector string-hash lookup, repeated for every neighbor of every row.
4. Filters `NA` values and coerces to integer.

The total number of inner operations is proportional to the sum of neighbor counts across all cell-years: ~6.46M rows Ã— ~4 rook neighbors on average â‰ˆ **~25.8 million string paste + hash-lookup operations**, all inside an interpreted R loop. The per-iteration overhead of string allocation, garbage collection, and named-vector lookup makes this extremely slow â€” estimated at **tens of hours** alone.

### `compute_neighbor_stats` â€” Secondary Bottleneck

This function iterates over the 6.46M-element `neighbor_lookup` list, subsetting a numeric vector and computing `max`, `min`, `mean` per element. While each operation is small, the R-level `lapply` loop over millions of elements with per-element function-call overhead, NA filtering, and `c()` allocation is costly. The final `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also expensive.

### Summary

| Component | Operation Count | Core Issue |
|---|---|---|
| `build_neighbor_lookup` | ~6.46M R-loop iterations, ~25.8M `paste` + hash lookups | Per-row string allocation, interpreted loop |
| `compute_neighbor_stats` | ~6.46M R-loop iterations Ã— 5 variables = ~32.3M | Per-row subsetting, NA check, summary stats in interpreted loop |
| `do.call(rbind, ...)` | Binding 6.46M small vectors Ã— 5 | Repeated memory allocation |

---

## Optimization Strategy

**Core idea:** Eliminate all row-level R loops and string-key lookups. Replace them with **vectorized joins and grouped aggregations** using `data.table`.

### Specific changes:

1. **Replace `build_neighbor_lookup`** entirely. Instead of building a 6.46M-element list of integer vectors (which is memory-heavy and slow to construct), build a **flat `data.table` edge list** of `(row_index, neighbor_row_index)` pairs using vectorized merge/join operations. This replaces millions of `paste` + hash-lookup calls with a single keyed `data.table` join.

2. **Replace `compute_neighbor_stats`** with a single **grouped `data.table` aggregation** (`[, .(max, min, mean), by = row_index]`) over the edge list joined to the variable values. This replaces millions of R-level function calls with C-level grouped operations.

3. **Avoid `do.call(rbind, ...)`** â€” the `data.table` aggregation returns a single result table directly.

4. **Preserve the trained Random Forest model** â€” we only change feature construction, not the model or the features themselves. The output columns are numerically identical.

### Complexity comparison:

| Step | Original | Optimized |
|---|---|---|
| Neighbor lookup construction | ~6.46M R-loop iterations with string ops | One vectorized `data.table` join (~25.8M rows) |
| Stats computation (per variable) | ~6.46M R-loop iterations | One grouped aggregation over ~25.8M rows |
| Estimated total time | 86+ hours | **~5â€“15 minutes** |

---

## Working R Code

```r
library(data.table)

#' Build a flat edge-list data.table mapping each row in `data` to its
#' neighbor rows, fully vectorized.  Replaces build_neighbor_lookup().
#'
#' @param data       data.frame / data.table with columns `id` and `year`
#' @param id_order   integer vector of cell IDs in the order used by the nb object
#' @param neighbors  spdep nb object (list of integer index vectors into id_order)
#' @return data.table with columns  focal_row, neighbor_row
build_neighbor_edge_list <- function(data, id_order, neighbors) {


  # --- 1. Build cell-level edge list (id -> neighbor_id) ------------------
  n_neighbors  <- lengths(neighbors)
  focal_idx    <- rep(seq_along(neighbors), n_neighbors)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  edge_cells <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )


  # --- 2. Map (id, year) -> row position in `data` -----------------------
  dt <- as.data.table(data)[, .(id, year)]
  dt[, row_pos := .I]

  # --- 3. Join: attach focal row positions --------------------------------
  #     For every (focal_id, year) combination that exists in `data`,

  #     expand by the cell-level edge list to get (focal_row, neighbor_id, year).
  setkey(dt, id)
  focal_dt <- dt[, .(focal_row = row_pos, id, year)]
  setkey(focal_dt, id)
  setkey(edge_cells, focal_id)

  # Each focal row fans out to its neighbors
  expanded <- edge_cells[focal_dt,
    .(focal_row, neighbor_id, year),
    on = .(focal_id = id),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # --- 4. Join: attach neighbor row positions -----------------------------
  setkey(dt, id, year)
  setkey(expanded, neighbor_id, year)

  result <- dt[expanded,
    .(focal_row, neighbor_row = row_pos),
    on = .(id = neighbor_id, year),
    nomatch = NULL
  ]

  return(result)
}


#' Compute neighbor max, min, mean for a variable using the edge list.
#' Replaces compute_neighbor_stats() + compute_and_add_neighbor_features().
#'
#' @param data       data.frame / data.table (full dataset)
#' @param edge_list  data.table from build_neighbor_edge_list()
#' @param var_name   character, column name in `data`
#' @return data.table with columns: focal_row, nb_max, nb_min, nb_mean
compute_neighbor_stats_fast <- function(data, edge_list, var_name) {

  vals <- data[[var_name]]

  # Attach neighbor values
  el <- copy(edge_list)
  el[, nb_val := vals[neighbor_row]]

  # Drop NAs (matches original behaviour)
  el <- el[!is.na(nb_val)]

  # Grouped aggregation â€” single pass in C
  stats <- el[, .(
    nb_max  = max(nb_val),
    nb_min  = min(nb_val),
    nb_mean = mean(nb_val)
  ), by = focal_row]

  return(stats)
}


#' Main driver: build features for all neighbor source variables and
#' attach them to cell_data.  Drop-in replacement for the outer loop.
#'
#' @param cell_data             data.frame with columns id, year, and the source vars
#' @param id_order              integer vector of cell IDs matching the nb object
#' @param rook_neighbors_unique spdep nb object
#' @param neighbor_source_vars  character vector of variable names
#' @return cell_data with new neighbor feature columns appended
add_all_neighbor_features <- function(cell_data,
                                      id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars) {

  message("Building vectorized edge list ...")
  edge_list <- build_neighbor_edge_list(cell_data, id_order, rook_neighbors_unique)
  message(sprintf("  Edge list: %s rows", format(nrow(edge_list), big.mark = ",")))

  n <- nrow(cell_data)

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for: %s", var_name))

    stats <- compute_neighbor_stats_fast(cell_data, edge_list, var_name)

    # Initialize with NA (matches original for cells with no valid neighbors)
    max_col  <- rep(NA_real_, n)
    min_col  <- rep(NA_real_, n)
    mean_col <- rep(NA_real_, n)

    # Fill in computed values
    max_col[stats$focal_row]  <- stats$nb_max
    min_col[stats$focal_row]  <- stats$nb_min
    mean_col[stats$focal_row] <- stats$nb_mean

    # Use the same column naming convention as the original pipeline
    cell_data[[paste0(var_name, "_nb_max")]]  <- max_col
    cell_data[[paste0(var_name, "_nb_min")]]  <- min_col
    cell_data[[paste0(var_name, "_nb_mean")]] <- mean_col
  }

  return(cell_data)
}


# ========================================================================
# USAGE â€” drop-in replacement for the original outer loop
# ========================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged; proceed directly to prediction:
# predictions <- predict(rf_model, cell_data[, predictor_cols])
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Numerical equivalence** | The same `max`, `min`, `mean` are computed over exactly the same set of non-NA neighbor values per cell-year. The `data.table` join replicates the original `paste(id, year)`-based matching logic, and the grouped aggregation replicates the per-element summary. |
| **NA handling** | Rows with zero valid neighbors produce `NA` for all three stats, identical to the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))`. |
| **Column names** | Output columns follow the same `{var}_nb_max`, `{var}_nb_min`, `{var}_nb_mean` convention. (Adjust the suffix pattern if the original `compute_and_add_neighbor_features` used a different naming scheme.) |
| **Trained RF model** | Untouched. Only the feature engineering step is optimized. |

---

## Expected Performance

On a 16 GB RAM laptop:

- **Edge list construction:** ~25.8M-row `data.table` â‰ˆ ~600 MB. Two keyed joins, each taking ~1â€“2 minutes.
- **Per-variable aggregation:** Grouped `max`/`min`/`mean` over ~25.8M rows â‰ˆ ~30â€“60 seconds each.
- **Total for 5 variables:** **~5â€“15 minutes** versus the original 86+ hours â€” roughly a **350â€“1000Ã— speedup**.
- **Peak RAM:** ~3â€“5 GB for the edge list plus intermediate copies, well within 16 GB.
END_RESPONSE>>>
