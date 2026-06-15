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
run_id: Anthropic_F07_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F07
prompt_family_slug: RASTER_INVALID_IRREGULAR
family_label: Raster invalid irregular topology
family_group: Raster/kernel bridge
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F07_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, which iterates over ~6.46 million rows one at a time in an R-level `lapply`. For each row it:

1. Looks up the cell's rook neighbors from the `nb` object.
2. Constructs character keys (`paste(id, year)`) and matches them into a named character vector (`idx_lookup`) of length 6.46M.

**Named-vector lookup via character keys in R is O(n) per probe** (hash-table notwithstanding, the constant is large and the vector is enormous). Doing this ~6.46M Ã— ~4 neighbors â‰ˆ 25 billion character comparisons is what produces the 86+ hour estimate.

`compute_neighbor_stats` is a secondary bottleneck: another R-level `lapply` over 6.46M elements, each extracting a small numeric slice. This is slow but less catastrophic.

**Root cause summary:**

| Component | Complexity | Problem |
|---|---|---|
| `build_neighbor_lookup` | O(rows Ã— avg_neighbors) with expensive character hashing | ~86 h |
| `compute_neighbor_stats` | O(rows Ã— avg_neighbors) in pure R | ~minutesâ€“hour per variable |
| Outer loop | 5 variables Ã— above | Multiplier |

## Optimization Strategy

### 1. Replace character-key lookup with integer arithmetic

Every `(id, year)` pair can be mapped to a row index via a **dense integer matrix** `row_matrix[cell_index, year_index]` where `cell_index âˆˆ 1:344208` and `year_index âˆˆ 1:28`. Building this matrix is O(rows). Looking up neighbors becomes a direct integer-indexed matrix access â€” effectively O(1) per neighbor.

### 2. Vectorize neighbor lookup construction

Pre-expand the `nb` object into a two-column edge list (from, to) once. Then use vectorized operations (no per-row `lapply`) to build the full set of `(row_i, row_j)` directed neighbor pairs across all years simultaneously.

### 3. Vectorize `compute_neighbor_stats`

With the edge list `(row_i, row_j)`, compute `max`, `min`, and `mean` of neighbor values using **`data.table` grouped aggregation** â€” a single vectorized pass per variable, replacing 6.46M R-level function calls.

### 4. Memory budget

- Edge list: ~1.37M edges Ã— 28 years Ã— 2 columns Ã— 8 bytes â‰ˆ ~600 MB (fits in 16 GB).
- Everything else is modest.

**Expected runtime: ~2â€“5 minutes total** (down from 86+ hours).

---

## Working R Code

```r
# =============================================================================
# Optimized neighbor feature engineering
# Preserves the trained RF model and the original numerical estimand.
# =============================================================================

library(data.table)

compute_all_neighbor_features <- function(cell_data, id_order, rook_neighbors_unique,
                                          neighbor_source_vars) {

  # -- Convert to data.table for speed (non-destructive copy) -----------------
  dt <- as.data.table(cell_data)

  # -- Step 1: Build dense cell-index and year-index maps --------------------
  #    id_order is the vector of cell IDs in the same order as the nb object.
  n_cells <- length(id_order)
  id_to_cellidx <- setNames(seq_len(n_cells), as.character(id_order))

  years <- sort(unique(dt$year))
  n_years <- length(years)
  year_to_yridx <- setNames(seq_len(n_years), as.character(years))

  # Map every row to (cell_index, year_index) --------------------------------
  dt[, cellidx := id_to_cellidx[as.character(id)]]
  dt[, yridx   := year_to_yridx[as.character(year)]]

  # -- Step 2: Build row-lookup matrix  row_mat[cellidx, yridx] = row number -
  row_mat <- matrix(NA_integer_, nrow = n_cells, ncol = n_years)
  row_mat[cbind(dt$cellidx, dt$yridx)] <- seq_len(nrow(dt))

  # -- Step 3: Expand nb object into a directed edge list (cell-level) -------
  #    from_cell -> to_cell  (rook neighbors)
  from_cell <- rep(seq_len(n_cells),
                   times = vapply(rook_neighbors_unique, length, integer(1)))
  to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove any 0-neighbor entries (empty integer(0) contributes nothing)
  valid <- to_cell > 0L

  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]

  n_edges_cell <- length(from_cell)   # ~1.37 M directed edges

  # -- Step 4: Tile across all years to get row-level edge list --------------
  #    For every year, map (from_cell, yr) -> row_i  and (to_cell, yr) -> row_j
  #    Vectorised: repeat edge list n_years times, once per year.

  from_cell_rep <- rep(from_cell, times = n_years)
  to_cell_rep   <- rep(to_cell,   times = n_years)
  yr_rep        <- rep(seq_len(n_years), each = n_edges_cell)

  row_i <- row_mat[cbind(from_cell_rep, yr_rep)]
  row_j <- row_mat[cbind(to_cell_rep,   yr_rep)]

  # Keep only edges where both endpoints exist in the data
  keep <- !is.na(row_i) & !is.na(row_j)
  row_i <- row_i[keep]
  row_j <- row_j[keep]

  # Free large temporaries
  rm(from_cell_rep, to_cell_rep, yr_rep, keep)
  gc()

  # Build edge data.table once
  edges <- data.table(row_i = row_i, row_j = row_j)
  rm(row_i, row_j)
  gc()

  # -- Step 5: For each variable, compute neighbor max / min / mean ----------
  n_rows <- nrow(dt)

  for (var_name in neighbor_source_vars) {
    message("Processing neighbor stats for: ", var_name)

    # Attach the neighbor's value to each edge
    edges[, val := dt[[var_name]][row_j]]

    # Remove edges where the neighbor value is NA
    edges_valid <- edges[!is.na(val)]

    # Grouped aggregation â€” single vectorised pass
    agg <- edges_valid[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), keyby = row_i]

    # Initialise result columns with NA
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)

    max_col[agg$row_i]  <- agg$nb_max
    min_col[agg$row_i]  <- agg$nb_min
    mean_col[agg$row_i] <- agg$nb_mean

    # Write into dt with the same column names the original code would produce
    set(dt, j = paste0(var_name, "_neighbor_max"),  value = max_col)
    set(dt, j = paste0(var_name, "_neighbor_min"),  value = min_col)
    set(dt, j = paste0(var_name, "_neighbor_mean"), value = mean_col)
  }

  # -- Step 6: Clean up helper columns and return as data.frame --------------
  dt[, c("cellidx", "yridx") := NULL]
  edges[, val := NULL]   # tidy up

  as.data.frame(dt)
}

# =============================================================================
# Usage  (drop-in replacement for the original outer loop)
# =============================================================================

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# The trained Random Forest model is untouched.
# Predictions proceed exactly as before:
# preds <- predict(rf_model, newdata = cell_data)
```

---

## Why This Preserves the Original Numerical Estimand

| Property | Original code | Optimized code |
|---|---|---|
| Neighbor definition | `rook_neighbors_unique[[cell]]` | Same `nb` object, same directed edges |
| Year matching | Neighbors matched within same year | Same: edges tiled per year identically |
| NA handling | `neighbor_vals[!is.na(neighbor_vals)]` | `edges[!is.na(val)]` â€” identical |
| No-neighbor rows | Return `(NA, NA, NA)` | Rows absent from `agg` â†’ `NA` columns |
| Aggregation functions | `max`, `min`, `mean` | Identical `max`, `min`, `mean` |
| Column names | `{var}_neighbor_max/min/mean` | Identical naming convention |

The output is **numerically identical** (up to floating-point associativity of `mean`, which `data.table` computes in the same double-precision arithmetic). The trained Random Forest model is never modified or retrained.
END_RESPONSE>>>
