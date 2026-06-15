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
run_id: Anthropic_F03_T00_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector)
- NA filtering

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this is extremely slow â€” likely **hours** on its own.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also slow (repeated memory allocation). This is called 5 times (once per neighbor source variable), so ~32.3M iterations total.

**`do.call(rbind, ...)` on millions of small vectors** is a classic R anti-pattern â€” it creates millions of tiny row matrices and binds them one by one.

### 1.2 Prediction Workflow Bottlenecks (Inferred)

- **Model loading:** If the serialized Random Forest is large (110 predictors, many trees), `readRDS` can take significant time and memory.
- **Single-call vs. loop prediction:** If `predict()` is called row-by-row or in small batches instead of once on the full data.frame, overhead is enormous.
- **Object copying:** R's copy-on-modify semantics mean that repeatedly doing `cell_data$new_col <- ...` inside a loop triggers full-dataframe copies each time. With ~6.46M rows Ã— 110+ columns, each copy is hundreds of MB.
- **Memory pressure:** 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ 5.7 GB for the numeric matrix alone. Combined with the RF model, neighbor lookup lists, and intermediate objects, 16 GB RAM is tight, causing garbage collection thrashing.

### 1.3 Root Cause Summary

| Component | Problem | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops + named-vector lookup on 6.46M keys | ~hours |
| `compute_neighbor_stats` | Per-row lapply + `do.call(rbind, ...)` Ã— 5 vars | ~hours |
| Column assignment in loop | Copy-on-modify of full data.frame Ã— 15 new columns | ~tens of minutes, GB of RAM churn |
| Prediction (likely) | Possible row-level or batch predict loop; large model load | ~hours if looped |
| Memory | 16 GB RAM saturated â†’ GC thrashing | Multiplier on all above |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything, eliminate per-row R loops, use `data.table` for zero-copy column addition, and call `predict()` once.

| Step | Action | Speedup Factor |
|---|---|---|
| **A.** Replace `build_neighbor_lookup` | Build a `data.table` join between (id, year) and neighbor-id, yielding a two-column integer matrix of (row_index, neighbor_row_index). No per-row loop. | ~100â€“500Ã— |
| **B.** Replace `compute_neighbor_stats` | Group-by aggregation on the edge table using `data.table`: group by `row_idx`, compute max/min/mean of neighbor values. Fully vectorized. | ~50â€“200Ã— |
| **C.** Use `data.table` for `cell_data` | Eliminates copy-on-modify. Column addition by reference (`:=`). | ~10Ã— for column ops |
| **D.** Single `predict()` call | Ensure prediction is one call: `predict(model, newdata = cell_data)`. | Critical |
| **E.** Memory management | Convert to matrix for predict if ranger; `gc()` after large intermediates; remove lookup objects when done. | Prevents thrashing |

**Expected total runtime: ~5â€“20 minutes** (down from 86+ hours).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Preserves: trained RF model (no retraining), original numerical estimand
# =============================================================================

library(data.table)

# ---- A. Optimized neighbor lookup: returns a data.table edge list -----------
#
# Instead of a list of length nrow(data), we build a two-column data.table:
#   from_row  : integer row index in cell_data
#   to_row    : integer row index of the neighbor in cell_data
#
# This is fully vectorized and avoids all per-row string operations.

build_neighbor_edges <- function(dt, id_order, neighbors) {
  # dt must be a data.table with columns 'id' and 'year'
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # Step 1: Expand the nb object into a two-column data.table of
  #         (focal_cell_id, neighbor_cell_id)
  n_cells <- length(id_order)
  from_lengths <- vapply(neighbors, length, integer(1))  # fast C-level
  edge_from_idx <- rep(seq_len(n_cells), times = from_lengths)
  edge_to_idx   <- unlist(neighbors, use.names = FALSE)

  edges_cell <- data.table(
    focal_id    = id_order[edge_from_idx],
    neighbor_id = id_order[edge_to_idx]
  )
  rm(edge_from_idx, edge_to_idx, from_lengths)

  # Step 2: Map (id, year) -> row index in dt
  dt[, row_idx := .I]

  # Step 3: Cross-join edges with all years present for each focal cell.
  #         We need (focal_id, year) -> row_idx  AND  (neighbor_id, year) -> row_idx
  id_year_map <- dt[, .(id, year, row_idx)]
  setkey(id_year_map, id, year)

  # Join focal side: get focal row_idx for every (focal_id, year) combination
  # First, get all (focal_id, year) pairs by joining edges_cell with the years
  # present for each focal_id.
  focal_years <- dt[, .(year), by = .(focal_id = id)]
  setkey(focal_years, focal_id)
  setkey(edges_cell, focal_id)

  # Merge: for each edge, replicate across all years of the focal cell
  edge_year <- edges_cell[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: focal_id, neighbor_id, year
  rm(edges_cell, focal_years)
  gc()

  # Join to get focal row index
  edge_year[id_year_map, on = .(focal_id = id, year = year), from_row := i.row_idx]


  # Join to get neighbor row index
  edge_year[id_year_map, on = .(neighbor_id = id, year = year), to_row := i.row_idx]

  # Drop edges where either side is missing
  edge_year <- edge_year[!is.na(from_row) & !is.na(to_row),
                         .(from_row, to_row)]

  rm(id_year_map)
  gc()

  return(edge_year)
}


# ---- B. Optimized neighbor stats: vectorized group-by aggregation -----------

compute_neighbor_stats_vec <- function(dt, edge_dt, var_name, nrow_dt) {
  # dt       : data.table with the variable column
  # edge_dt  : data.table with columns from_row, to_row
  # var_name : character, name of the variable
  # nrow_dt  : total number of rows in dt
  #
  # Returns a data.table with columns:
  #   <var_name>_neighbor_max, <var_name>_neighbor_min, <var_name>_neighbor_mean

  vals <- dt[[var_name]]

  # Attach neighbor values to edge table (by reference-safe copy of needed cols)
  work <- edge_dt[, .(from_row, to_row)]
  work[, nval := vals[to_row]]

  # Remove edges where neighbor value is NA

  work <- work[!is.na(nval)]

  # Aggregate by focal row
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]

  # Build full-length result (NA for rows with no valid neighbors)
  result <- data.table(
    nb_max  = rep(NA_real_, nrow_dt),
    nb_min  = rep(NA_real_, nrow_dt),
    nb_mean = rep(NA_real_, nrow_dt)
  )
  result[agg$from_row, `:=`(
    nb_max  = agg$nb_max,
    nb_min  = agg$nb_min,
    nb_mean = agg$nb_mean
  )]

  col_max  <- paste0(var_name, "_neighbor_max")
  col_min  <- paste0(var_name, "_neighbor_min")
  col_mean <- paste0(var_name, "_neighbor_mean")
  setnames(result, c("nb_max", "nb_min", "nb_mean"),
                   c(col_max,  col_min,  col_mean))

  return(result)
}


# ---- C. Full pipeline ------------------------------------------------------

run_optimized_pipeline <- function(cell_data_input,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model_path,
                                   neighbor_source_vars = c("ntl", "ec",
                                                            "pop_density",
                                                            "def",
                                                            "usd_est_n2")) {

  # --- 0. Convert to data.table (no copy if already data.table) ---
  if (!is.data.table(cell_data_input)) {
    cell_data <- as.data.table(cell_data_input)
  } else {
    cell_data <- copy(cell_data_input)
  }

  cat("Rows:", nrow(cell_data), " Cols:", ncol(cell_data), "\n")
  nrow_cd <- nrow(cell_data)

  # --- 1. Build vectorized neighbor edge list ---
  cat("Building neighbor edge list...\n")
  t0 <- proc.time()
  edge_dt <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
  cat("  Edge list:", nrow(edge_dt), "edges. Time:",
      round((proc.time() - t0)[3], 1), "s\n")

  # Remove temporary row_idx if added
  if ("row_idx" %in% names(cell_data)) {
    cell_data[, row_idx := NULL]
  }

  # --- 2. Compute neighbor features (vectorized) ---
  cat("Computing neighbor features...\n")
  t0 <- proc.time()
  for (var_name in neighbor_source_vars) {
    cat("  Variable:", var_name, "...")
    stats_dt <- compute_neighbor_stats_vec(cell_data, edge_dt, var_name, nrow_cd)

    # Add columns by reference â€” no copy of cell_data
    new_cols <- names(stats_dt)
    for (col in new_cols) {
      set(cell_data, j = col, value = stats_dt[[col]])
    }
    rm(stats_dt)
    cat(" done\n")
  }
  cat("  Neighbor features time:", round((proc.time() - t0)[3], 1), "s\n")

  # Free edge table
  rm(edge_dt)
  gc()

  # --- 3. Load trained Random Forest model ---
  cat("Loading RF model...\n")
  t0 <- proc.time()
  rf_model <- readRDS(rf_model_path)
  cat("  Model load time:", round((proc.time() - t0)[3], 1), "s\n")

  # --- 4. Predict â€” single vectorized call ---
  cat("Running prediction on", nrow_cd, "rows...\n")
  t0 <- proc.time()

  # Identify the predictor columns the model expects
  # Works for randomForest, ranger, and most RF implementations
  if (inherits(rf_model, "ranger")) {
    # ranger: predict expects a data.frame or data.table
    pred <- predict(rf_model, data = cell_data)$predictions
  } else if (inherits(rf_model, "randomForest")) {
    # randomForest: predict expects newdata as data.frame
    pred <- predict(rf_model, newdata = cell_data)
  } else {
    # Generic fallback
    pred <- predict(rf_model, newdata = cell_data)
  }

  cat("  Prediction time:", round((proc.time() - t0)[3], 1), "s\n")

  # --- 5. Attach predictions ---
  cell_data[, predicted_gdp := pred]

  # Clean up model from memory
  rm(rf_model, pred)
  gc()

  cat("Pipeline complete.\n")
  return(cell_data)
}


# ---- D. Example invocation -------------------------------------------------
#
# result <- run_optimized_pipeline(
#   cell_data_input      = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model_path        = "path/to/trained_rf_model.rds",
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Loaded via `readRDS`; never modified or retrained. `predict()` is called exactly once on the full dataset. |
| **Numerical estimand** | `max`, `min`, `mean` are computed on exactly the same neighbor sets as the original code. The edge-list join replicates the same (focal_id, year) â†” (neighbor_id, year) mapping. NA handling is identical: edges with NA neighbor values are excluded before aggregation; rows with no valid neighbors get NA. |
| **Column names** | The naming convention `<var>_neighbor_max/min/mean` matches the pattern from `compute_and_add_neighbor_features`. Adjust the suffix pattern if your original wrapper uses different names. |

---

## 5. EXPECTED PERFORMANCE

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~10â€“30 hours | ~1â€“3 minutes |
| `compute_neighbor_stats` Ã— 5 | ~30â€“50 hours | ~2â€“5 minutes |
| Column assignment (Ã—15 cols) | ~2â€“5 hours (copy-on-modify) | ~seconds (by-reference) |
| RF prediction | Unknown (depends on implementation) | ~2â€“10 minutes (single call) |
| **Total** | **86+ hours** | **~5â€“20 minutes** |

The key transformation is replacing **6.46 million R-level loop iterations** with **vectorized `data.table` joins and group-by aggregations** that execute in compiled C code internally.
END_RESPONSE>>>
