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
run_id: Anthropic_F10_T10_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F10
prompt_family_slug: SPARSE_GRAPH_FRAME
family_label: Sparse graph frame
family_group: Graph cue
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F10_Anthropic_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The current implementation suffers from three major bottlenecks:

1. **O(nÂ²) key-hashing in `build_neighbor_lookup`:** For ~6.46 million rows, `paste(id, year)` key creation and named-vector lookups via `idx_lookup[neighbor_keys]` are extremely slow. Named vector lookups in R use linear hashing that degrades at scale.

2. **Row-level `lapply` over 6.46M rows:** Both `build_neighbor_lookup` and `compute_neighbor_stats` iterate row-by-row in pure R. Each iteration has R interpreter overhead (function calls, subsetting, NA checks), multiplied by 6.46M rows Ã— 5 variables = ~32.3M inner loop invocations for stats alone.

3. **Redundant topology recomputation per year:** The rook neighbor graph is **spatial only** â€” it doesn't change across years. Yet the lookup embeds year into the key, rebuilding adjacency relationships across the full panel rather than separating the static spatial topology from the temporal indexing.

**Memory:** The neighbor lookup list alone (6.46M elements, each a variable-length integer vector) consumes several GB, leaving little headroom on 16 GB RAM.

## Optimization Strategy

1. **Separate spatial topology from temporal indexing.** Build the sparse adjacency structure once over 344,208 cells (not 6.46M cell-years). Reuse it identically for every year.

2. **Use sparse matrixâ€“vector multiplication for aggregation.** Construct a sparse CSR/CSC adjacency matrix `A` (344K Ã— 344K, ~1.37M nonzeros). For each year and each variable, extract the variable vector `x` (length 344K), then:
   - `mean_neighbor = (A %*% x) / (A %*% 1_{non-NA})` (weighted by non-NA count)
   - `max_neighbor` and `min_neighbor` via grouped operations on the sparse structure

3. **Vectorize max/min using `data.table` grouped operations** on the edge list, which avoids R-level row iteration entirely.

4. **Process year-by-year** to keep memory footprint at ~344K vectors rather than 6.46M.

5. **Preserve numerical equivalence:** same `max`, `min`, `mean` of non-NA rook-neighbor values, same NA propagation when a node has zero non-NA neighbors.

## Working R Code

```r
# =============================================================================
# Optimized Rook-Neighbor Feature Engineering
# =============================================================================
# Requirements: data.table, Matrix, ranger (or randomForest for prediction)
# Preserves numerical equivalence with original compute_neighbor_stats output.
# =============================================================================

library(data.table)
library(Matrix)

# ---- 1. Build edge list once from the nb object (344K cells) ----------------

build_edge_list <- function(nb_obj) {
  # nb_obj: spdep nb object â€” list of length N, each element is integer vector
  # of neighbor indices (1-based). Rook contiguity is symmetric but we store

  # directed edges (i -> j) for every j in nb_obj[[i]].
  from <- rep(seq_along(nb_obj), lengths(nb_obj))
  to   <- unlist(nb_obj, use.names = FALSE)
  data.table(from = as.integer(from), to = as.integer(to))
}

# ---- 2. Compute neighbor stats for one variable, one year ------------------
#     Fully vectorised via data.table grouped ops on the edge list.

compute_neighbor_stats_fast <- function(vals_vec, edge_dt) {
  # vals_vec: numeric vector length N_cells, ordered by cell index
  # edge_dt:  data.table with columns 'from', 'to' (cell indices)
  #
  # Returns: data.table with columns from, nb_max, nb_min, nb_mean (length N_cells)
  # Rows with no valid neighbors get NA.

  N <- length(vals_vec)

  # Attach neighbor values
  edge_dt[, nval := vals_vec[to]]

  # Drop edges where neighbor value is NA
  valid <- edge_dt[!is.na(nval)]

  if (nrow(valid) == 0L) {
    return(data.table(
      from    = seq_len(N),
      nb_max  = rep(NA_real_, N),
      nb_min  = rep(NA_real_, N),
      nb_mean = rep(NA_real_, N)
    ))
  }

  # Grouped aggregation â€” one pass
  agg <- valid[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from]

  # Expand to full cell set (cells with no valid neighbors -> NA)
  result <- data.table(from = seq_len(N))
  result <- merge(result, agg, by = "from", all.x = TRUE, sort = TRUE)

  # Clean up temp column in edge_dt (in-place modification)
  edge_dt[, nval := NULL]

  result
}

# ---- 3. Main pipeline ------------------------------------------------------

run_neighbor_feature_pipeline <- function(cell_data,         # data.table or data.frame
                                          id_order,          # integer vector: cell IDs in nb index order
                                          rook_neighbors_unique,  # spdep nb object
                                          neighbor_source_vars,   # character vector of variable names
                                          rf_model) {        # pre-trained RF model

  # Convert to data.table for speed (copy to avoid mutating original)
  dt <- as.data.table(copy(cell_data))

  # --- Build cell-index mapping ---
  # id_order[k] is the cell ID for the k-th position in the nb object.
  # We need a map: cell_id -> nb_index
  id_to_idx <- setNames(seq_along(id_order), as.character(id_order))

  # Assign each row its spatial index
  dt[, cell_idx := as.integer(id_to_idx[as.character(id)])]

  N_cells <- length(id_order)
  years   <- sort(unique(dt$year))

  # --- Build edge list once (spatial topology is time-invariant) ---
  cat("Building edge list from nb object...\n")
  edge_dt <- build_edge_list(rook_neighbors_unique)
  cat(sprintf("  %d directed edges across %d cells.\n", nrow(edge_dt), N_cells))

  # --- Pre-sort data by (year, cell_idx) for fast extraction ---
  setkey(dt, year, cell_idx)

  # --- Allocate result columns ---
  for (var_name in neighbor_source_vars) {
    for (suffix in c("_nb_max", "_nb_min", "_nb_mean")) {
      col <- paste0(var_name, suffix)
      if (is.null(dt[[col]])) {
        set(dt, j = col, value = NA_real_)
      }
    }
  }

  # --- Process year-by-year Ã— variable-by-variable ---
  cat(sprintf("Processing %d years Ã— %d variables...\n", length(years), length(neighbor_source_vars)))

  for (yr in years) {
    # Row indices for this year (already keyed)
    yr_rows <- dt[.(yr), which = TRUE]

    if (length(yr_rows) == 0L) next

    # Extract the cell indices present this year
    yr_cell_idx <- dt$cell_idx[yr_rows]

    # For cells present in this year, build a dense vector of length N_cells
    # (cells not present this year will have NA -> neighbors get NA correctly)

    for (var_name in neighbor_source_vars) {
      # Dense vector: position k = value for cell k in this year
      vals_dense <- rep(NA_real_, N_cells)
      vals_dense[yr_cell_idx] <- dt[[var_name]][yr_rows]

      # Compute neighbor stats (vectorised)
      stats <- compute_neighbor_stats_fast(vals_dense, edge_dt)

      # Write back to the data.table rows for this year
      # stats is indexed by cell_idx (1..N_cells); we need to map back
      max_col  <- paste0(var_name, "_nb_max")
      min_col  <- paste0(var_name, "_nb_min")
      mean_col <- paste0(var_name, "_nb_mean")

      set(dt, i = yr_rows, j = max_col,  value = stats$nb_max[yr_cell_idx])
      set(dt, i = yr_rows, j = min_col,  value = stats$nb_min[yr_cell_idx])
      set(dt, i = yr_rows, j = mean_col, value = stats$nb_mean[yr_cell_idx])
    }

    if (yr %% 5 == 0 || yr == years[1] || yr == tail(years, 1)) {
      cat(sprintf("  Year %d complete.\n", yr))
    }
  }

  # --- Clean up helper column ---
  dt[, cell_idx := NULL]

  # --- Predict using pre-trained RF model ---
  cat("Running Random Forest prediction...\n")

  # Get the predictor names the model expects
  if (inherits(rf_model, "ranger")) {
    pred_vars <- rf_model$forest$independent.variable.names
    preds <- predict(rf_model, data = dt[, ..pred_vars])$predictions
  } else if (inherits(rf_model, "randomForest")) {
    preds <- predict(rf_model, newdata = dt)
  } else {
    stop("Unsupported RF model class: ", class(rf_model)[1])
  }

  dt[, rf_prediction := preds]

  cat("Done.\n")
  return(dt)
}

# ---- 4. Usage ---------------------------------------------------------------

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
#
# result <- run_neighbor_feature_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   neighbor_source_vars   = neighbor_source_vars,
#   rf_model               = trained_rf_model
# )
```

## Expected Performance Improvement

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M `paste` + named-vector lookups | Edge list built once (344K cells, ~1.37M edges) |
| **Stats computation** | 6.46M Ã— 5 = 32.3M R-level `lapply` iterations | 28 years Ã— 5 vars = 140 vectorised `data.table` group-bys |
| **Memory for lookup** | ~6.46M-element list of integer vectors (multi-GB) | One 2-column integer `data.table` (~11 MB) |
| **Time complexity** | O(R Ã— E) with R interpreter overhead per row | O(Y Ã— V Ã— E/Y) fully vectorised per year-variable |
| **Estimated wall time** | 86+ hours | **~5â€“15 minutes** on the same laptop |

### Why numerical equivalence is preserved

- `compute_neighbor_stats_fast` computes `max`, `min`, and `mean` over the same set of non-NA neighbor values that the original `compute_neighbor_stats` uses.
- The edge list is derived from the identical `rook_neighbors_unique` nb object.
- Cells with zero non-NA neighbors yield `NA` for all three statistics, matching the original `c(NA, NA, NA)` return.
- The pre-trained Random Forest model is used **only** for `predict()` â€” no retraining occurs.
END_RESPONSE>>>
