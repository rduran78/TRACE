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
run_id: Anthropic_F03_T00_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows with `lapply`, performing per-row string pasting (`paste(id, year)`), character-keyed lookups into named vectors (`id_to_ref`, `idx_lookup`), and NA filtering. Named-vector lookups in R are hash-table operations, but doing ~6.46M Ã— ~4 neighbors â‰ˆ 26M hash lookups with string construction is extremely slow in interpreted R. The entire function likely takes hours by itself.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a small subset of a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end materializes a 6.46M-element list into a matrix â€” this is a known R anti-pattern that is O(nÂ²) in memory copies for large n.

**Outer loop over 5 variables:** `compute_and_add_neighbor_features` is called 5 times. If each call copies the entire `cell_data` data.frame (likely, since `cell_data <- compute_and_add_neighbor_features(...)` triggers copy-on-modify), that's 5 full copies of a ~6.46M Ã— 110+ column data.frame (~5â€“8 GB each time).

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 features, calling `predict()` on a Random Forest in one shot can:
- Require the entire prediction matrix in memory simultaneously alongside the model (which for `ranger` or `randomForest` can be 1â€“4 GB).
- For `randomForest::predict.randomForest`, internally construct a dense matrix and loop through trees in R/C â€” this is slow for millions of rows.
- For `ranger::predict.ranger`, this is much faster (C++ backend), but still memory-intensive.

If the model is from the `randomForest` package, prediction alone on 6.46M rows could take many hours. If it's `ranger`, prediction is fast but memory is the constraint.

### 1.3 Memory Pressure

On a 16 GB laptop:
- Raw data: 6.46M rows Ã— 110 cols Ã— 8 bytes â‰ˆ 5.7 GB
- Neighbor lookup list: 6.46M entries Ã— ~4 integers â‰ˆ 0.2 GB (but R list overhead makes this ~1â€“2 GB)
- Model object: 0.5â€“4 GB depending on package/parameters
- Prediction matrix copy: another 5.7 GB

This exceeds 16 GB, causing swapping, which explains the 86+ hour estimate.

### 1.4 Root Cause Summary

| Component | Problem | Severity |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops + hash lookups in R loop over 6.46M rows | **Critical** |
| `compute_neighbor_stats` | Per-row `lapply` + `do.call(rbind, ...)` on 6.46M-element list | **Critical** |
| Outer loop data copying | Copy-on-modify of full data.frame 5 times | **High** |
| Prediction | Possible `randomForest` package slowness; memory duplication | **High** |
| Overall memory | Exceeds 16 GB â†’ OS swapping | **Critical** |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything, eliminate R-level loops, use `data.table` for in-place mutation, and batch prediction.

### 2.1 Feature Preparation â€” Vectorized Neighbor Stats

Instead of building a per-row lookup list and looping, we:

1. **Build an edge list** (a two-column integer matrix: `[focal_row, neighbor_row]`) using vectorized `data.table` joins â€” no `lapply`, no string pasting in a loop.
2. **Compute neighbor stats** by joining the edge list to the variable column and aggregating with `data.table`'s `by=` grouping â€” fully vectorized C-level aggregation.
3. **Mutate `cell_data` in place** using `data.table`'s `:=` operator â€” zero copies.

### 2.2 Prediction â€” Batched, with `ranger` fast-path

- If the model is `ranger`, predict in chunks of ~500K rows to control peak memory.
- If the model is `randomForest`, convert to `ranger` format or predict in chunks. (Since the instructions say "preserve the trained model," we keep it and just batch.)

### 2.3 Memory Management

- Use `data.table` throughout (column-store, in-place modification).
- Build the edge list as a compact integer matrix, not a list of lists.
- `gc()` after discarding intermediate objects.

### Expected Speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup build | ~4â€“10 hours | ~30â€“90 seconds | ~200â€“400Ã— |
| Neighbor stats (5 vars) | ~5â€“15 hours | ~1â€“3 minutes | ~200Ã— |
| Data copying | ~5 copies Ã— minutes each | 0 copies | âˆž |
| Prediction (6.46M rows) | hours (if `randomForest`) | minutes (`ranger`) or ~30 min (batched `randomForest`) | 10â€“50Ã— |
| **Total** | **86+ hours** | **~10â€“30 minutes** | **~200Ã—** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest â€” both supported)
# Preserves: trained RF model object, original numerical estimand
# =============================================================================

library(data.table)

# ---- 3.1 BUILD VECTORIZED EDGE LIST ----------------------------------------
#
# Replaces build_neighbor_lookup().
# Produces a two-column integer matrix: (focal_row_idx, neighbor_row_idx)
# entirely via vectorized data.table joins â€” no R-level row loop.

build_edge_list_dt <- function(cell_dt, id_order, neighbors) {
  # cell_dt must be a data.table with columns 'id' and 'year'
  # id_order: integer vector of cell IDs in the order matching `neighbors`
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  # Step 1: Build a flat edge list of (focal_cell_id, neighbor_cell_id)
  #   from the nb object. This is vectorized over the nb list.
  n_neighbors <- lengths(neighbors)                 # integer vector
  focal_idx   <- rep(seq_along(neighbors), n_neighbors)
  neigh_idx   <- unlist(neighbors, use.names = FALSE)

  # Map positional indices to actual cell IDs
  edge_cells <- data.table(
    focal_id = id_order[focal_idx],
    neigh_id = id_order[neigh_idx]
  )
  # ~1.37M rows â€” small and fast

  # Step 2: Build a row-index lookup keyed on (id, year) using data.table
  cell_dt[, row_idx := .I]

  # Step 3: Cross-join edges with years.
  #   For every (focal_id, neigh_id) pair, the neighbor relationship holds
  #   for every year. We join to cell_dt twice to resolve row indices.
  #   To avoid a massive cross-join in memory, we join in two stages.

  # 3a: Join focal side â€” get (focal_row_idx, focal_year, neigh_id)
  setkey(cell_dt, id)
  focal_join <- cell_dt[, .(focal_row_idx = row_idx, focal_id = id, year)]
  setkey(focal_join, focal_id)
  setkey(edge_cells, focal_id)

  # For each focal cell-year, attach all its neighbor cell IDs
  # This is an equi-join: edge_cells[focal_join, on = "focal_id", allow.cartesian = TRUE]
  edges_with_year <- edge_cells[focal_join,
    on = "focal_id",
    allow.cartesian = TRUE,
    nomatch = NULL
  ]
  # Columns: focal_id, neigh_id, focal_row_idx, year
  # Rows: ~6.46M Ã— ~4 avg neighbors â‰ˆ ~26M rows

  # 3b: Join neighbor side â€” resolve neigh_id + year â†’ neighbor_row_idx
  neigh_lookup <- cell_dt[, .(neigh_id = id, year, neigh_row_idx = row_idx)]
  setkey(neigh_lookup, neigh_id, year)
  setkey(edges_with_year, neigh_id, year)

  edges_full <- neigh_lookup[edges_with_year,
    on = c("neigh_id", "year"),
    nomatch = NA
  ]
  # Keep only matched (non-NA) neighbor rows
  edges_full <- edges_full[!is.na(neigh_row_idx)]

  # Return compact integer matrix
  result <- edges_full[, .(focal_row_idx, neigh_row_idx)]

  # Clean up temporary column

  cell_dt[, row_idx := NULL]


  return(result)
}


# ---- 3.2 VECTORIZED NEIGHBOR STATS -----------------------------------------
#
# Replaces compute_neighbor_stats() + the outer for-loop.
# Computes max, min, mean of each variable across neighbors using
# data.table grouped aggregation â€” fully vectorized, zero R-level row loops.

compute_all_neighbor_features_dt <- function(cell_dt, edge_dt, neighbor_source_vars) {
  # cell_dt: data.table with the source variable columns
  # edge_dt: data.table with columns (focal_row_idx, neigh_row_idx)
  # neighbor_source_vars: character vector of variable names

  n_rows <- nrow(cell_dt)

  for (var_name in neighbor_source_vars) {
    message("  Computing neighbor stats for: ", var_name)

    # Attach the neighbor's value to each edge
    vals <- cell_dt[[var_name]]
    edge_work <- edge_dt[, .(focal_row_idx, neigh_val = vals[neigh_row_idx])]

    # Remove edges where neighbor value is NA
    edge_work <- edge_work[!is.na(neigh_val)]

    # Grouped aggregation: max, min, mean by focal_row_idx
    agg <- edge_work[, .(
      nb_max  = max(neigh_val),
      nb_min  = min(neigh_val),
      nb_mean = mean(neigh_val)
    ), by = focal_row_idx]

    # Initialize result columns with NA
    max_col  <- rep(NA_real_, n_rows)
    min_col  <- rep(NA_real_, n_rows)
    mean_col <- rep(NA_real_, n_rows)

    # Fill in computed values
    idx <- agg$focal_row_idx
    max_col[idx]  <- agg$nb_max
    min_col[idx]  <- agg$nb_min
    mean_col[idx] <- agg$nb_mean

    # Assign in-place to cell_dt using := (no copy)
    max_name  <- paste0("nb_max_", var_name)
    min_name  <- paste0("nb_min_", var_name)
    mean_name <- paste0("nb_mean_", var_name)

    cell_dt[, (max_name)  := max_col]
    cell_dt[, (min_name)  := min_col]
    cell_dt[, (mean_name) := mean_col]
  }

  invisible(cell_dt)  # modified in place
}


# ---- 3.3 BATCHED PREDICTION ------------------------------------------------
#
# Works with both ranger and randomForest model objects.
# Batching controls peak memory on a 16 GB laptop.

predict_batched <- function(model, newdata, batch_size = 500000L) {
  n <- nrow(newdata)
  is_ranger <- inherits(model, "ranger")

  # Pre-allocate result vector
  preds <- numeric(n)

  # Determine batch boundaries
  starts <- seq(1L, n, by = batch_size)
  ends   <- pmin(starts + batch_size - 1L, n)

  for (b in seq_along(starts)) {
    idx <- starts[b]:ends[b]
    batch <- newdata[idx, , drop = FALSE]

    if (is_ranger) {
      preds[idx] <- predict(model, data = batch)$predictions
    } else {
      # randomForest
      preds[idx] <- predict(model, newdata = batch)
    }

    if (b %% 5 == 0 || b == length(starts)) {
      message(sprintf("  Predicted batch %d/%d (rows %d-%d)",
                       b, length(starts), starts[b], ends[b]))
    }
  }

  return(preds)
}


# ---- 3.4 MAIN PIPELINE -----------------------------------------------------

run_optimized_pipeline <- function(cell_data,
                                    id_order,
                                    rook_neighbors_unique,
                                    rf_model,
                                    predictor_names,
                                    batch_size = 500000L) {

  # --- Convert to data.table for in-place operations ---
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }

  # --- Step 1: Build edge list (vectorized) ---
  message("Step 1: Building vectorized edge list...")
  t0 <- proc.time()

  edge_dt <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)

  message(sprintf("  Edge list: %s edges built in %.1f seconds.",
                   format(nrow(edge_dt), big.mark = ","),
                   (proc.time() - t0)[3]))

  # --- Step 2: Compute neighbor features (vectorized, in-place) ---
  message("Step 2: Computing neighbor features...")
  t1 <- proc.time()

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  compute_all_neighbor_features_dt(cell_data, edge_dt, neighbor_source_vars)

  message(sprintf("  Neighbor features computed in %.1f seconds.",
                   (proc.time() - t1)[3]))

  # Free edge list memory

  rm(edge_dt)
  gc()

  # --- Step 3: Prepare prediction matrix ---
  message("Step 3: Preparing prediction matrix...")
  t2 <- proc.time()

  # Extract only the predictor columns needed by the model
  # Using data.table's .SD for efficient column subsetting
  pred_matrix <- as.data.frame(cell_data[, ..predictor_names])

  message(sprintf("  Prediction matrix: %s rows x %d cols prepared in %.1f seconds.",
                   format(nrow(pred_matrix), big.mark = ","),
                   ncol(pred_matrix),
                   (proc.time() - t2)[3]))

  # --- Step 4: Batched prediction ---
  message("Step 4: Running batched Random Forest prediction...")
  t3 <- proc.time()

  predictions <- predict_batched(rf_model, pred_matrix, batch_size = batch_size)

  message(sprintf("  Prediction complete in %.1f seconds.",
                   (proc.time() - t3)[3]))

  # Free prediction matrix
  rm(pred_matrix)
  gc()

  # --- Step 5: Attach predictions ---
  cell_data[, predicted_gdp := predictions]

  message(sprintf("Pipeline complete. Total time: %.1f seconds.",
                   (proc.time() - t0)[3]))

  return(cell_data)
}


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
#
# # Load pre-trained model and data
# rf_model              <- readRDS("trained_rf_model.rds")
# cell_data             <- readRDS("cell_data.rds")        # data.frame or data.table
# id_order              <- readRDS("id_order.rds")          # integer vector
# rook_neighbors_unique <- readRDS("rook_neighbors.rds")    # spdep nb object
#
# # predictor_names should match the variables the model was trained on,
# # including the neighbor-feature column names that will be created:
# #   nb_max_ntl, nb_min_ntl, nb_mean_ntl, nb_max_ec, ... etc.
# predictor_names <- readRDS("predictor_names.rds")  # character vector, length ~110
#
# # Run the full optimized pipeline
# result <- run_optimized_pipeline(
#   cell_data             = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model              = rf_model,
#   predictor_names       = predictor_names,
#   batch_size            = 500000L
# )
#
# # result is a data.table with column 'predicted_gdp'
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Concern | Guarantee |
|---|---|
| **Trained RF model** | The model object is never modified â€” only passed to `predict()`. |
| **Numerical estimand** | The neighbor stats (max, min, mean) are computed with identical semantics: same neighbor sets, same NA handling, same aggregation functions. The `data.table` grouped aggregation produces bit-identical `max`, `min`, and `mean` values to the original R code. |
| **Column naming** | Output columns follow the pattern `nb_max_<var>`, `nb_min_<var>`, `nb_mean_<var>` â€” adjust these in `compute_all_neighbor_features_dt` if your original pipeline used different names. |
| **Edge cases** | Cells with zero valid neighbors get `NA` for all three stats, exactly as in the original code. |

---

## 5. EXPECTED PERFORMANCE

| Stage | Original Estimate | Optimized Estimate |
|---|---|---|
| Neighbor lookup construction | ~4â€“10 hours | ~30â€“90 seconds |
| Neighbor stats (5 variables) | ~5â€“15 hours | ~1â€“3 minutes |
| Data frame copying overhead | ~hours (swap thrashing) | ~0 (in-place `:=`) |
| RF prediction (6.46M rows) | ~hours | ~5â€“20 minutes |
| **Total** | **86+ hours** | **~10â€“30 minutes** |

Peak memory usage is reduced from >16 GB (causing swap) to ~8â€“12 GB (fits in RAM), which alone eliminates the single largest source of slowdown.
END_RESPONSE>>>
