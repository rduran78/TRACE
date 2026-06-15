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
run_id: Anthropic_F10_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Optimized R Code

---

## Diagnosis

The current implementation has three critical bottlenecks:

1. **O(nÂ²)-scale `neighbor_lookup` construction**: `build_neighbor_lookup` creates a list of 6.46 million elements. For each of the ~6.46M rows, it performs string concatenation (`paste`), named-vector lookups (`idx_lookup[neighbor_keys]`), and NA filtering. Named character vector lookups in R are O(n) hash probes on 6.46M-length named vectors, repeated 6.46M times.

2. **Row-level R `lapply` in `compute_neighbor_stats`**: For each of the 5 variables, `lapply` iterates over all 6.46M rows in interpreted R, extracting neighbor values, filtering NAs, and computing `max/min/mean`. That is ~32.3 million R-level iterations total.

3. **The lookup is year-aware but redundant**: The spatial topology (rook neighbors) is identical across all 28 years. Yet `build_neighbor_lookup` rebuilds per-row neighbor indices by cross-referencing `(cell_id, year)` keys, inflating a 344K-cell adjacency structure into a 6.46M-row lookup. This is a 28Ã— redundancy.

**Key insight**: The neighbor graph is *time-invariant*. Cell `i`'s rook neighbors in 1992 are the same cells as in 2019. The only thing that changes is the attribute values. Therefore, the aggregation decomposes into: (a) build a sparse adjacency matrix **once** over 344K cells, and (b) for each year, subset the data, perform sparse matrixâ€“vector multiplication (for `mean`) and analogous operations (for `max`, `min`), and write back.

---

## Optimization Strategy

| Aspect | Original | Optimized |
|---|---|---|
| Topology size | 6.46M-row lookup list | 344K Ã— 344K sparse matrix (built once) |
| Aggregation | R-level `lapply` per row | Sparse matrix multiplication (`Matrix` package) + `by-year` vectorized ops |
| `mean` | Manual loop | `A %*% x / degree` (one sparse matvec per variable-year) |
| `max` / `min` | Manual loop | Vectorized via `data.table` grouped operations on edge list |
| Passes over data | 5 vars Ã— 6.46M rows Ã— 3 stats = ~97M scalar ops in R | 5 vars Ã— 28 years Ã— 3 sparse ops on 344K-length vectors |
| Estimated time | 86+ hours | Minutes |

**Approach**:
- Convert the `nb` object into a sparse adjacency matrix (`dgCMatrix`) and an edge-list `data.table` â€” built once.
- For `mean`: sparse matrixâ€“vector product gives the sum of neighbor values; divide by the degree vector.
- For `max` and `min`: use a `data.table` edge-list join, grouped by target node, which is highly optimized in C.
- Loop over years (28) and variables (5), operating on vectors of length 344K rather than 6.46M.
- Write results directly into the full `data.table` by reference.

This preserves exact numerical equivalence: same neighbor sets, same `max`, `min`, `mean(na.rm-style)` semantics, same NA propagation when a node has zero valid neighbors.

---

## Optimized R Code

```r
library(data.table)
library(Matrix)

optimize_neighbor_features <- function(cell_data,
                                        id_order,
                                        rook_neighbors_unique,
                                        neighbor_source_vars) {
  # ---------------------------------------------------------------
  # 0.  Convert cell_data to data.table (by reference if possible)
  # ---------------------------------------------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  n_cells <- length(id_order)
  years   <- sort(unique(cell_data$year))

  # ---------------------------------------------------------------
  # 1.  Build cell-id â†’ integer index mapping (1-based, matches
  #     the ordering in id_order which matches rook_neighbors_unique)
  # ---------------------------------------------------------------
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Map every row's cell id to the spatial index
  cell_data[, spatial_idx := id_to_idx[as.character(id)]]

  # ---------------------------------------------------------------
  # 2.  Build sparse adjacency structures ONCE (344K cells)
  #
  #     rook_neighbors_unique is an nb object: a list of length

  #     n_cells where element [[i]] is an integer vector of
  #     neighbor positions (into id_order).  A value of 0L means
  #     no neighbors (spdep convention).
  # ---------------------------------------------------------------

  # --- 2a. Edge list (for max / min via data.table) ---
  from_vec <- integer(0)
  to_vec   <- integer(0)
  for (i in seq_len(n_cells)) {
    nb_i <- rook_neighbors_unique[[i]]
    nb_i <- nb_i[nb_i != 0L]
    if (length(nb_i) > 0L) {
      # edge direction: neighbors point TO node i
      # (we want to aggregate *neighbor* attributes *for* node i)
      from_vec <- c(from_vec, nb_i)
      to_vec   <- c(to_vec,   rep.int(i, length(nb_i)))
    }
  }
  edge_dt <- data.table(from = from_vec, to = to_vec)

  # --- 2b. Sparse adjacency matrix A (for mean via matvec) ---
  #     A[i, j] = 1 means j is a neighbor of i  â†’  A %*% x gives
  #     the sum of neighbor values for each node.
  A <- sparseMatrix(
    i = to_vec,
    j = from_vec,
    x = 1,
    dims = c(n_cells, n_cells)
  )

  # Degree vector (number of neighbors per node, ignoring NAs for now â€”

  # NA handling is done below).
  degree <- as.numeric(rowSums(A))   # length n_cells

  # ---------------------------------------------------------------
  # 3.  Pre-allocate output columns in cell_data
  # ---------------------------------------------------------------
  for (var_name in neighbor_source_vars) {
    max_col  <- paste0("max_neighbor_",  var_name)
    min_col  <- paste0("min_neighbor_",  var_name)
    mean_col <- paste0("mean_neighbor_", var_name)
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
  }

  # ---------------------------------------------------------------
  # 4.  Keyed index:  for each year, which rows correspond to
  #     which spatial_idx?  (Enables O(1) lookup.)
  #     We assume the panel is balanced or nearly balanced.
  # ---------------------------------------------------------------
  setkey(cell_data, year, spatial_idx)

  # ---------------------------------------------------------------
  # 5.  Main loop: iterate over years Ã— variables
  # ---------------------------------------------------------------
  for (yr in years) {
    # Rows for this year, ordered by spatial_idx
    yr_rows <- cell_data[.(yr)]           # keyed subset
    yr_idx  <- yr_rows$spatial_idx        # which cells are present

    # If panel is unbalanced some cells may be missing.
    # Build a full-length vector (n_cells) padded with NA for
    # missing cells so that sparse indexing is correct.
    # Also build a presence mask.
    present      <- logical(n_cells)
    present[yr_idx] <- TRUE

    for (var_name in neighbor_source_vars) {
      max_col  <- paste0("max_neighbor_",  var_name)
      min_col  <- paste0("min_neighbor_",  var_name)
      mean_col <- paste0("mean_neighbor_", var_name)

      # --- full-length value vector (NA for absent cells) ---
      full_vals <- rep(NA_real_, n_cells)
      full_vals[yr_idx] <- yr_rows[[var_name]]

      # ---- MEAN (sparse matvec) ----
      # Replace NA with 0 for summation; track valid-neighbor counts.
      vals_zero        <- full_vals
      vals_zero[is.na(vals_zero)] <- 0
      valid_indicator  <- as.numeric(!is.na(full_vals))

      neighbor_sum   <- as.numeric(A %*% vals_zero)
      neighbor_count <- as.numeric(A %*% valid_indicator)

      mean_result <- rep(NA_real_, n_cells)
      ok <- neighbor_count > 0
      mean_result[ok] <- neighbor_sum[ok] / neighbor_count[ok]

      # ---- MAX and MIN (data.table edge-list join) ----
      # Attach the "from" node's value to each edge, group by "to"
      edge_vals <- edge_dt[, .(from, to)]
      edge_vals[, val := full_vals[from]]
      edge_vals <- edge_vals[!is.na(val)]  # drop edges with NA source

      if (nrow(edge_vals) > 0L) {
        agg <- edge_vals[, .(max_v = max(val), min_v = min(val)), by = to]
        max_result <- rep(NA_real_, n_cells)
        min_result <- rep(NA_real_, n_cells)
        max_result[agg$to] <- agg$max_v
        min_result[agg$to] <- agg$min_v
      } else {
        max_result <- rep(NA_real_, n_cells)
        min_result <- rep(NA_real_, n_cells)
      }

      # ---- Write results back into cell_data for this year ----
      # yr_idx maps rows in yr_rows to spatial positions
      set(cell_data,
          i = which(cell_data$year == yr),
          j = max_col,
          value = max_result[cell_data[year == yr, spatial_idx]])
      set(cell_data,
          i = which(cell_data$year == yr),
          j = min_col,
          value = min_result[cell_data[year == yr, spatial_idx]])
      set(cell_data,
          i = which(cell_data$year == yr),
          j = mean_col,
          value = mean_result[cell_data[year == yr, spatial_idx]])
    }

    if (interactive()) message("Year ", yr, " done.")
  }

  # Clean up helper column
  cell_data[, spatial_idx := NULL]

  return(cell_data)
}

# ================================================================
#  USAGE
# ================================================================

# -- Load pre-existing objects (assumed already in environment) --
# cell_data                : data.frame / data.table with columns id, year, ntl, ec, ...
# id_order                 : integer/character vector, cell IDs in nb-object order
# rook_neighbors_unique    : spdep nb object (list of integer vectors)
# trained_rf               : pre-trained randomForest / ranger model

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- optimize_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# -- Predict with the existing (frozen) Random Forest model --
# (Column names now match the original pipeline's output.)
cell_data$prediction <- predict(trained_rf, newdata = cell_data)
```

---

## Why This Is Numerically Equivalent

| Statistic | Original | Optimized | Equivalence |
|---|---|---|---|
| **mean** | `mean(neighbor_vals)` after dropping NAs | `sum(vals_zero[neighbors]) / count(!is.na(vals[neighbors]))` via sparse matvec | Identical: both compute arithmetic mean of non-NA neighbor values |
| **max** | `max(neighbor_vals)` after dropping NAs | `data.table` grouped `max(val)` after dropping NA edges | Identical: same set, same function |
| **min** | `min(neighbor_vals)` after dropping NAs | `data.table` grouped `min(val)` after dropping NA edges | Identical |
| **NA rule** | If all neighbors are NA or no neighbors exist â†’ `NA` | `neighbor_count == 0 â†’ NA` for mean; absent from `agg` â†’ stays `NA` for max/min | Identical |

## Expected Performance

| Step | Time estimate |
|---|---|
| Build sparse matrix + edge list (once) | ~5â€“15 seconds |
| Per year-variable: sparse matvec (344K) | ~0.01â€“0.05 s |
| Per year-variable: edge-list max/min (~1.37M edges) | ~0.05â€“0.15 s |
| **Total** (28 years Ã— 5 vars) | **~2â€“5 minutes** |
| Write-back via `data.table::set` | ~1â€“2 minutes |
| **Overall** | **~3â€“8 minutes** (vs. 86+ hours) |

The trained Random Forest model is never modified â€” only `predict()` is called on the enriched dataset.
END_RESPONSE>>>
