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
run_id: Anthropic_F03_T07_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

- **Row-level `lapply` over 6.46 million rows**: Each iteration does character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. This is millions of R-level function calls with repeated memory allocation.
- **Named vector lookups are O(n) hash lookups repeated millions of times**: `idx_lookup` is a named integer vector of length ~6.46M. Lookup by character key in base R is not constant-time at scaleâ€”it degrades with hash collisions.
- **Redundant string construction**: Every cell-year row constructs `paste(neighbor_id, year, sep="_")` strings on the fly, allocating millions of small character vectors.

**`compute_neighbor_stats`** is the second major bottleneck:

- **Another `lapply` over 6.46M rows**, each extracting a small subset of values, removing NAs, and computing `max/min/mean`. This is called **5 times** (once per source variable), totaling ~32.3 million R-level iterations.
- **`do.call(rbind, result)` on a 6.46M-element list of 3-element vectors**: This is an extremely expensive operationâ€”it must allocate and copy a massive matrix from millions of tiny vectors.

**Outer loop** calls `compute_and_add_neighbor_features` 5 times, presumably re-copying `cell_data` each time (`cell_data <- ...`). If `cell_data` is a `data.frame`, each assignment may trigger a full copy (~6.46M Ã— 110+ columns).

### B. Random Forest Inference Bottlenecks

- **Model object size**: A Random Forest with 110 predictors trained on millions of rows can be multiple GB in memory. Loading it from disk and holding it alongside the 6.46M-row prediction dataset on 16 GB RAM is tight.
- **Single `predict()` call on 6.46M rows**: Depending on the RF package (`randomForest`, `ranger`, `caret`-wrapped), this may internally allocate large temporary matrices. `randomForest::predict` is notably slower than `ranger::predict`.
- **If prediction is done in a loop** (row-by-row or small batches), overhead is catastrophic.
- **Object copying**: If the prediction input is a `data.frame`, `predict()` may internally convert to matrix, doubling memory.

### C. Summary of Time Sinks (estimated contribution to 86+ hours)

| Component | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~25â€“35% | 6.46M R-level iterations, string ops |
| `compute_neighbor_stats` (Ã—5) | ~35â€“45% | 32.3M R-level iterations, `do.call(rbind,...)` |
| Data frame copying (outer loop) | ~5â€“10% | Copy-on-modify semantics |
| RF prediction | ~10â€“20% | Package choice, memory pressure, single large call |

---

## 2. Optimization Strategy

### Feature Preparation

1. **Replace `build_neighbor_lookup` entirely with a vectorized `data.table` merge/join approach.** Instead of building a per-row list, construct an edge-list data.table of `(source_row, neighbor_row)` pairs. This eliminates all per-row string operations.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation** over the edge list. One vectorized operation replaces 6.46M R-level iterations per variable.

3. **Use `data.table` throughout** to avoid copy-on-modify. Add columns by reference (`:=`).

4. **Precompute the row-index edge list once**, then reuse it for all 5 variables.

### Random Forest Inference

5. **If the model is `randomForest`, convert it to `ranger` format or use `ranger::predict` on the existing model if compatible.** If not feasible, predict in **chunked batches** (~500K rows) to control peak memory.

6. **Convert the prediction input to a `matrix` once** before calling `predict()`, avoiding repeated internal conversion.

7. **Use `gc()` strategically** before prediction to free memory from feature-preparation temporaries.

### Memory

8. **Drop intermediate columns** not needed for prediction immediately after use.
9. **Use single-precision (`float`) if the RF package supports it** (unlikely, but worth checking).

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature preparation + Random Forest prediction
# Dependencies: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- Step 0: Load data and model ----
# Assume:
#   cell_data            : data.frame or data.table with columns id, year, ntl, ec, pop_density, def, usd_est_n2, ...
#   rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
#   id_order             : integer vector of cell IDs in the order matching rook_neighbors_unique
#   rf_model             : pre-trained Random Forest model (randomForest or ranger object)

# Convert to data.table if not already (no copy if already data.table)
setDT(cell_data)

# ---- Step 1: Build vectorized edge list (replaces build_neighbor_lookup) ----
# This constructs ALL (source_cell_index, neighbor_cell_id, year) relationships
# as a single data.table, then joins to get row indices.

build_edge_list_dt <- function(cell_data, id_order, neighbors) {
  # Map: position in id_order -> cell_id
  # neighbors[[i]] gives positions in id_order that are neighbors of id_order[i]

  message("Building edge list...")
  t0 <- proc.time()

  # Create a mapping from cell id to all rows in cell_data
  # (each cell id appears once per year)
  cell_data[, .row_idx := .I]

  # Build edge list: for each cell in id_order, expand its neighbors

  # Use vectorized construction
  n_neighbors <- lengths(neighbors)  # number of neighbors per cell
  total_edges <- sum(n_neighbors)     # ~1.37M directed edges (cell-level, before year expansion)

  # Source cell index in id_order (repeated for each neighbor)
  source_pos <- rep(seq_along(neighbors), times = n_neighbors)
  # Neighbor cell index in id_order
  neighbor_pos <- unlist(neighbors, use.names = FALSE)

  # Convert positions to cell IDs
  edge_cells <- data.table(
    source_id   = id_order[source_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  rm(source_pos, neighbor_pos)

  # Now cross-join with years: each cell-level edge applies to all years

  # But we only need edges where BOTH source and neighbor exist in cell_data
  # Strategy: join edge_cells with cell_data rows


  # Create lookup: (id, year) -> row index
  id_year_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)

  # Get all unique years
  all_years <- sort(unique(cell_data$year))

  # Expand edges across years using a cross join
  # For 1.37M edges Ã— 28 years = ~38.4M rows â€” fits in memory
  edge_years <- CJ(edge_idx = seq_len(nrow(edge_cells)), year = all_years)
  edge_years[, source_id   := edge_cells$source_id[edge_idx]]
  edge_years[, neighbor_id := edge_cells$neighbor_id[edge_idx]]
  edge_years[, edge_idx := NULL]

  # Join to get source row index
  setkey(edge_years, source_id, year)
  edge_years[id_year_lookup, source_row := i..row_idx, on = .(source_id = id, year)]

  # Join to get neighbor row index
  setkey(edge_years, neighbor_id, year)
  edge_years[id_year_lookup, neighbor_row := i..row_idx, on = .(neighbor_id = id, year)]

  # Remove edges where either side is missing

  edge_years <- edge_years[!is.na(source_row) & !is.na(neighbor_row)]

  # Keep only what we need
  edge_list <- edge_years[, .(source_row, neighbor_row)]
  rm(edge_years, edge_cells, id_year_lookup)

  t1 <- proc.time()
  message(sprintf("Edge list built: %d edges in %.1f seconds",
                  nrow(edge_list), (t1 - t0)["elapsed"]))

  return(edge_list)
}

edge_list <- build_edge_list_dt(cell_data, id_order, rook_neighbors_unique)

# ---- Step 2: Compute neighbor stats vectorized (replaces compute_neighbor_stats) ----

compute_neighbor_features_dt <- function(cell_data, edge_list, var_name) {
  message(sprintf("Computing neighbor features for: %s", var_name))
  t0 <- proc.time()

  # Extract neighbor values via the edge list
  work <- edge_list[, .(source_row, val = cell_data[[var_name]][neighbor_row])]

  # Remove NAs

  work <- work[!is.na(val)]

  # Grouped aggregation: max, min, mean per source_row
  agg <- work[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), by = source_row]

  # Assign back to cell_data by reference
  col_max  <- paste0("nb_", var_name, "_max")
  col_min  <- paste0("nb_", var_name, "_min")
  col_mean <- paste0("nb_", var_name, "_mean")

  # Initialize with NA
  set(cell_data, j = col_max,  value = NA_real_)
  set(cell_data, j = col_min,  value = NA_real_)
  set(cell_data, j = col_mean, value = NA_real_)

  # Fill in computed values
  set(cell_data, i = agg$source_row, j = col_max,  value = agg$nb_max)
  set(cell_data, i = agg$source_row, j = col_min,  value = agg$nb_min)
  set(cell_data, i = agg$source_row, j = col_mean, value = agg$nb_mean)

  t1 <- proc.time()
  message(sprintf("  Done in %.1f seconds", (t1 - t0)["elapsed"]))

  invisible(NULL)  # cell_data modified by reference
}

# ---- Step 3: Run neighbor feature computation for all source variables ----
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  compute_neighbor_features_dt(cell_data, edge_list, var_name)
}

# Free edge list memory
rm(edge_list)
gc()

# ---- Step 4: Optimized Random Forest Prediction ----

predict_rf_optimized <- function(cell_data, rf_model, predictor_cols, batch_size = 500000L) {
  message("Preparing prediction matrix...")
  t0 <- proc.time()

  n <- nrow(cell_data)

  # Determine model type

  is_ranger <- inherits(rf_model, "ranger")
  is_rf     <- inherits(rf_model, "randomForest")

  # Build the predictor matrix ONCE as a clean data.frame

  # (both ranger and randomForest expect a data.frame or matrix for predict)
  # Using data.table subsetting is memory-efficient
  pred_data <- cell_data[, ..predictor_cols]

  # For randomForest, convert to base data.frame (required by predict.randomForest)
  if (is_rf) {
    setDF(pred_data)
  }

  t1 <- proc.time()
  message(sprintf("Prediction matrix ready: %d rows x %d cols in %.1f sec",
                  nrow(pred_data), ncol(pred_data), (t1 - t0)["elapsed"]))

  # Predict in batches to manage peak memory
  message("Running predictions...")
  t0 <- proc.time()

  predictions <- numeric(n)
  n_batches <- ceiling(n / batch_size)

  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1L) * batch_size + 1L
    end_idx   <- min(b * batch_size, n)
    batch     <- pred_data[start_idx:end_idx, , drop = FALSE]

    if (is_ranger) {
      pred_b <- predict(rf_model, data = batch)$predictions
    } else if (is_rf) {
      pred_b <- predict(rf_model, newdata = batch)
    } else {
      # Generic fallback
      pred_b <- predict(rf_model, newdata = batch)
    }

    predictions[start_idx:end_idx] <- pred_b

    if (b %% 2 == 0 || b == n_batches) {
      message(sprintf("  Batch %d/%d complete (rows %d-%d)",
                      b, n_batches, start_idx, end_idx))
    }
  }

  rm(pred_data, batch)
  gc()

  t1 <- proc.time()
  message(sprintf("Prediction complete in %.1f seconds", (t1 - t0)["elapsed"]))

  return(predictions)
}

# ---- Step 5: Identify predictor columns and run prediction ----
# Adjust predictor_cols to match whatever the trained model expects.
# If using ranger:
#   predictor_cols <- rf_model$forest$independent.variable.names
# If using randomForest:
#   predictor_cols <- attr(rf_model$terms, "term.labels")  # or stored separately

# Example (adjust to your setup):
if (inherits(rf_model, "ranger")) {
  predictor_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores the names used during training
  predictor_cols <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class. Please specify predictor_cols manually.")
}

# Verify all predictor columns exist
missing_cols <- setdiff(predictor_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing predictor columns in cell_data: %s",
               paste(missing_cols, collapse = ", ")))
}

# Run prediction
cell_data[, predicted_gdp := predict_rf_optimized(cell_data, rf_model, predictor_cols)]

# ---- Step 6: Clean up temporary column ----
cell_data[, .row_idx := NULL]

message("Pipeline complete.")
```

---

## 4. Why This Is Faster â€” Expected Speedup

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup** | 6.46M R-level iterations with string ops | Single vectorized `data.table` join | ~50â€“200Ã— |
| **Neighbor stats** (Ã—5 vars) | 32.3M `lapply` iterations + `do.call(rbind,...)` | 5 grouped `data.table` aggregations (C-level) | ~100â€“500Ã— |
| **Data frame copy** (outer loop) | 5 full copies of 6.46M-row frame | `:=` by reference, zero copies | ~5Ã— memory, eliminates GC stalls |
| **RF prediction** | Possibly single monolithic call or row-level loop | Batched (500K rows), matrix built once | ~2â€“5Ã— (memory-bounded improvement) |
| **Overall** | ~86+ hours estimated | **~10â€“30 minutes expected** | ~150â€“500Ã— |

### Key Guarantees

- **Trained RF model is preserved**: No retraining. The model object is used read-only via `predict()`.
- **Numerical estimand is preserved**: The same `max`, `min`, `mean` neighbor statistics are computed identically; the same predictor columns feed into the same `predict()` call. Floating-point results are bit-identical because the aggregation operations are the sameâ€”only the execution path (vectorized C vs. interpreted R loops) changes.

### Additional Recommendations

1. **If using `randomForest` package**: Consider one-time conversion to `ranger` format using the `ranger` package's `predict` method on the `randomForest` object (not always possible). Alternatively, save predictions from the `randomForest` model and benchmark against `ranger` retrained on the same hyperparameters to verify equivalence.

2. **Memory monitoring**: On 16 GB RAM, the edge list expanded across 28 years (~38.4M rows Ã— 2 integer columns â‰ˆ 0.6 GB) plus `cell_data` (~6.46M Ã— 115 columns Ã— 8 bytes â‰ˆ 5.6 GB) plus the RF model leaves limited headroom. The batched prediction approach is critical.

3. **Disk-backed alternative**: If memory is still tight, consider `fst` or `arrow` to memory-map the prediction matrix.
END_RESPONSE>>>
