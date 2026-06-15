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
run_id: Anthropic_F04_T03_R029
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F04
prompt_family_slug: NEIGHBOR_BOTTLENECK_HINT
family_label: Neighbor bottleneck hint
family_group: Spatial-neighbor cue
replicate: 29
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F04_Anthropic_temp_03_rep_029.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are clear:

### 1. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46 million rows with per-row string operations

For every one of the ~6.46 million rows, the function:
- Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
- Retrieves neighbor cell IDs from the `nb` object.
- Calls `paste()` to construct `"id_year"` key strings for every neighbor of that row.
- Performs named-vector lookup (`idx_lookup[neighbor_keys]`) â€” which is an **O(k)** hash lookup per neighbor key, but the string construction and allocation for ~6.46M Ã— ~4 neighbors (average rook degree â‰ˆ 4) means roughly **25+ million small string allocations and hash probes**.
- Wraps everything in an R-level `lapply`, so there is no vectorization; each iteration pays full R interpreter overhead.

**Estimated cost:** With ~6.46M iterations, each doing string paste + hash lookup for ~4 neighbors, this alone can take many hours in pure R.

### 2. `compute_neighbor_stats` â€” Another O(n) `lapply` with per-row subsetting

For each of the 6.46M rows, it:
- Subsets a numeric vector by index (`vals[idx]`).
- Removes NAs.
- Computes `max`, `min`, `mean`.

This is called **5 times** (once per source variable), so it executes ~32.3 million R-level loop iterations total. The per-row function-call overhead in R's `lapply` is the killer here.

### 3. `do.call(rbind, result)` on a 6.46M-element list of 3-vectors

This is a well-known R anti-pattern. Binding millions of small vectors into a matrix via `do.call(rbind, ...)` is extremely slow due to repeated memory allocation.

### Summary of bottleneck contributions (estimated):

| Component | Estimated share |
|---|---|
| `build_neighbor_lookup` (string ops + hash) | ~40% |
| `compute_neighbor_stats` (R-level loop Ã— 5 vars) | ~50% |
| `do.call(rbind, ...)` Ã— 5 | ~10% |

---

## Optimization Strategy

The core idea: **replace all row-level R loops and string operations with vectorized operations on integer indices using `data.table`.**

Specifically:

1. **Replace `build_neighbor_lookup`** with a single `data.table` join. Instead of building a list-of-integer-vectors (one per row), build a **flat edge table** `(row_i, neighbor_row_j)` using vectorized integer operations â€” no `paste`, no named-vector hash lookups.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation on the flat edge table: group by `row_i`, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass. This replaces 6.46M R-level `lapply` iterations with a single C-level `data.table` grouped operation.

3. **Eliminate `do.call(rbind, ...)`** entirely â€” `data.table` returns the result as a table directly.

4. **Process all 5 variables in one pass** over the edge table if desired, or in 5 fast grouped aggregations (each taking seconds, not hours).

**Expected speedup:** From ~86+ hours to **~2â€“10 minutes** on the same laptop.

**Numerical equivalence:** The operations are identical â€” `max`, `min`, `mean` of the same neighbor values. The trained Random Forest model is untouched.

---

## Working R Code

```r
library(data.table)

#' Build a flat edge table mapping each cell-year row to its neighbor cell-year rows.
#' Replaces build_neighbor_lookup entirely â€” no lapply, no paste, no string hashing.
#'
#' @param cell_dt     data.table with columns: id, year (and all predictor columns).
#'                    Must have a column .row_idx = seq_len(nrow(.)).
#' @param id_order    integer vector of cell IDs in the order matching the nb object.
#' @param neighbors   spdep nb object (list of integer index vectors into id_order).
#' @return data.table with columns: row_i (focal row), row_j (neighbor row).
build_neighbor_edge_table <- function(cell_dt, id_order, neighbors) {

  n_cells <- length(id_order)

  # --- Step 1: Build flat cell-level edge list (vectorized) ---
  # neighbors[[k]] gives integer indices into id_order for cell id_order[k].
  # We need: from_cell_id -> to_cell_id

  n_neighbors <- vapply(neighbors, length, integer(1))
  from_cell_idx <- rep(seq_len(n_cells), times = n_neighbors)
  to_cell_idx   <- unlist(neighbors, use.names = FALSE)

  # Convert positional indices to actual cell IDs
  edge_cells <- data.table(
    from_id = id_order[from_cell_idx],
    to_id   = id_order[to_cell_idx]
  )
  rm(from_cell_idx, to_cell_idx, n_neighbors)

  # --- Step 2: Get the unique years present in the data ---
  years <- sort(unique(cell_dt$year))

  # --- Step 3: Cross-join edges Ã— years, then join to row indices ---
  # Expand edges to all years (each spatial edge exists in every year)
  edge_years <- edge_cells[, .(year = years), by = .(from_id, to_id)]
  rm(edge_cells)

  # Build a lookup: (id, year) -> row index in cell_dt
  # cell_dt must already have .row_idx column
  id_year_lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)

  # Map from_id,year -> row_i
  setnames(id_year_lookup, ".row_idx", "row_i")
  setkey(edge_years, from_id, year)
  edge_years <- id_year_lookup[edge_years, on = .(id = from_id, year), nomatch = 0L]
  # Now edge_years has columns: id, year, row_i, to_id

  # Map to_id,year -> row_j
  setnames(id_year_lookup, "row_i", "row_j")
  edge_years <- id_year_lookup[edge_years, on = .(id = to_id, year), nomatch = 0L]
  # Now edge_years has columns: id, year, row_j, i.id, i.year, row_i, to_id

  # Keep only what we need
  result <- edge_years[, .(row_i, row_j)]
  setkey(result, row_i)

  rm(id_year_lookup, edge_years)
  gc()

  return(result)
}


#' Compute neighbor max, min, mean for a variable using the flat edge table.
#' Replaces compute_neighbor_stats â€” one vectorized data.table grouped aggregation.
#'
#' @param cell_dt    data.table with .row_idx and the variable column.
#' @param edge_dt    data.table with columns row_i, row_j (from build_neighbor_edge_table).
#' @param var_name   character: name of the variable to aggregate.
#' @return data.table with columns: .row_idx, <var>_max, <var>_min, <var>_mean
compute_neighbor_stats_fast <- function(cell_dt, edge_dt, var_name) {

  # Attach neighbor values to edge table
  vals <- cell_dt[[var_name]]
  agg <- edge_dt[, .(neighbor_val = vals[row_j]), by = row_i]

  # Remove NAs before aggregation
  agg <- agg[!is.na(neighbor_val)]

  # Grouped aggregation â€” single pass in C
  stats <- agg[, .(
    nb_max  = max(neighbor_val),
    nb_min  = min(neighbor_val),
    nb_mean = mean(neighbor_val)
  ), by = row_i]

  # Rename columns to match original naming convention
  setnames(stats, c("nb_max", "nb_min", "nb_mean"),
           paste0(var_name, c("_max", "_min", "_mean")))

  return(stats)
}


#' Main driver: build features for all neighbor source variables.
#' Drop-in replacement for the original outer loop.
#'
#' @param cell_data           data.frame or data.table â€” the full panel dataset.
#' @param id_order            integer vector of cell IDs matching the nb object.
#' @param rook_neighbors_unique  spdep nb object.
#' @param neighbor_source_vars   character vector of variable names.
#' @return data.table with all original columns plus neighbor feature columns.
add_all_neighbor_features <- function(cell_data, id_order,
                                      rook_neighbors_unique,
                                      neighbor_source_vars) {

  cell_dt <- as.data.table(cell_data)
  cell_dt[, .row_idx := .I]

  message("Building neighbor edge table (vectorized)...")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edge_table(cell_dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge table: %d edges, built in %.1f seconds.",
                  nrow(edge_dt), (proc.time() - t0)[3]))

  # Compute and join neighbor stats for each variable
  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s'...", var_name))
    t1 <- proc.time()

    stats <- compute_neighbor_stats_fast(cell_dt, edge_dt, var_name)

    # Left-join back to cell_dt on .row_idx = row_i
    # Rows with no valid neighbors will get NA (correct behavior, matches original)
    feat_cols <- paste0(var_name, c("_max", "_min", "_mean"))
    cell_dt <- stats[cell_dt, on = .(row_i = .row_idx)]
    setnames(cell_dt, "row_i", ".row_idx")

    elapsed <- (proc.time() - t1)[3]
    message(sprintf("  Done in %.1f seconds.", elapsed))
  }

  # Clean up helper column
  cell_dt[, .row_idx := NULL]

  rm(edge_dt)
  gc()

  return(cell_dt)
}


# ============================================================
# USAGE â€” drop-in replacement for the original outer loop
# ============================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- add_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is unchanged.
# Proceed directly to prediction:
# predictions <- predict(rf_model, newdata = cell_data)
```

---

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Identical â€” same `max`, `min`, `mean` over the same neighbor sets. NAs handled identically (removed before aggregation; rows with zero valid neighbors get `NA`). |
| **Trained RF model** | Untouched â€” only the feature construction pipeline is changed. Column names and semantics are preserved. |
| **Memory on 16 GB laptop** | The flat edge table has ~1.37M spatial edges Ã— 28 years â‰ˆ 38.5M rows Ã— 2 integer columns â‰ˆ **~0.6 GB**. The aggregation temporary is similar. Total peak overhead â‰ˆ ~2â€“3 GB, well within 16 GB. |
| **Expected runtime** | Edge table construction: ~30â€“60 seconds. Each variable's grouped aggregation: ~10â€“30 seconds. **Total: ~2â€“5 minutes** vs. 86+ hours. |

### Key optimizations summarized:

1. **Eliminated all `paste()`/string hashing** â€” replaced with integer key joins in `data.table`.
2. **Eliminated 6.46M Ã— 5 = 32.3M R-level `lapply` iterations** â€” replaced with 5 `data.table` grouped aggregations (executed in C).
3. **Eliminated `do.call(rbind, ...)`** â€” `data.table` returns results directly as a table.
4. **Edge table is built once and reused** for all 5 variables.
END_RESPONSE>>>
