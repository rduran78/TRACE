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
run_id: Anthropic_F03_T00_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_006.md

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

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also costly (creates millions of tiny row vectors then binds them). This is called **5 times** (once per neighbor source variable), compounding the cost.

**`do.call(rbind, ...)` on millions of small vectors** is a well-known R anti-pattern â€” it is O(nÂ²) in memory allocation.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, calling `predict()` on a Random Forest in one shot can:
- Require the entire prediction matrix in memory simultaneously (110 columns Ã— 6.46M rows Ã— 8 bytes â‰ˆ 5.7 GB for numeric alone).
- If using `randomForest::predict.randomForest`, it internally copies the data into a matrix, doubling memory.
- On a 16 GB laptop, this risks swapping to disk, which is catastrophic for performance.

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | Per-row `lapply` with string ops on 6.46M rows | ~hours |
| `compute_neighbor_stats` | Per-row `lapply` Ã— 5 vars + `do.call(rbind, ...)` | ~hours |
| `predict()` | Possible memory pressure / object copying on 16 GB machine | ~hours (with swapping) |
| Overall | No vectorization, no `data.table`, no chunking | 86+ hours total |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorize with `data.table`

- Replace the row-level `lapply` neighbor lookup with a **fully vectorized join** approach:
  1. Build an edge list (a two-column `data.table`) mapping each `(id, year)` â†’ all neighbor `(id, year)` rows.
  2. Join the variable values onto the edge list.
  3. Aggregate (`max`, `min`, `mean`) by the focal row using `data.table`'s `by=` grouping â€” this is a single vectorized pass per variable.

- This eliminates all per-row `lapply`, all `paste`-key lookups, and the `do.call(rbind, ...)`.

### 2.2 Prediction â€” Chunk-Based Inference

- Split the 6.46M rows into chunks (e.g., 500K rows) and call `predict()` per chunk.
- This keeps peak memory well within 16 GB and avoids OS-level swapping.
- Optionally convert the prediction input to a plain `matrix` to avoid internal copying in `predict.randomForest`.

### 2.3 Expected Speedup

| Component | Before | After (est.) |
|---|---|---|
| Neighbor lookup build | ~hours | ~1â€“3 min |
| Neighbor stats (Ã—5 vars) | ~hours | ~2â€“5 min |
| Prediction | ~hours (with swap) | ~10â€“30 min |
| **Total** | **86+ hours** | **~15â€“40 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” Feature Preparation + Chunked RF Prediction
# =============================================================================
# Requirements: data.table, randomForest (or ranger â€” works with either)
# The trained RF model object is preserved exactly as-is; no retraining.
# =============================================================================

library(data.table)

# ---- 3.1 BUILD VECTORIZED NEIGHBOR EDGE LIST --------------------------------
#
# Instead of a per-row lookup list, we build a data.table with columns:
#   focal_id, neighbor_id
# Then we cross-join with years to get:
#   focal_id, year, neighbor_id  â†’  which we join to row indices.
#
# This replaces build_neighbor_lookup entirely.

build_neighbor_edges <- function(id_order, neighbors_nb) {
  # neighbors_nb is an spdep::nb object: a list of integer index vectors
  # id_order is the vector of cell IDs in the same order as the nb object
  edges <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb_idx <- neighbors_nb[[i]]
    # spdep::nb uses 0 to indicate no neighbors
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) == 0L) return(NULL)
    data.table(focal_id = id_order[i], neighbor_id = id_order[nb_idx])
  }))
  edges
}

# ---- 3.2 COMPUTE ALL NEIGHBOR FEATURES (VECTORIZED) -------------------------

compute_all_neighbor_features_dt <- function(cell_dt, id_order, neighbors_nb,
                                              neighbor_source_vars) {
  # cell_dt: a data.table with columns id, year, and all source vars
  # Returns cell_dt with new columns appended (modified in place)

  # Step 1: Build edge list (focal_id â†’ neighbor_id)
  cat("Building neighbor edge list...\n")
  edges <- build_neighbor_edges(id_order, neighbors_nb)
  cat(sprintf("  Edge list: %s rows\n", format(nrow(edges), big.mark = ",")))

  # Step 2: Create a row-index column for the focal rows
  #         We need to map (id, year) â†’ row position
  cell_dt[, .row_idx := .I]

  # Step 3: For each variable, join neighbor values and aggregate
  for (var_name in neighbor_source_vars) {
    cat(sprintf("Computing neighbor stats for: %s\n", var_name))

    # Build a slim table: id, year, value
    val_dt <- cell_dt[, .(id, year, val = get(var_name))]

    # Join edges to get focal-year + neighbor value
    #   focal side: edges[focal_id] Ã— years  â†’  join to cell_dt for (focal_id, year, .row_idx)
    #   neighbor side: join to val_dt on (neighbor_id, year) to get neighbor value

    # Merge focal row index onto edges Ã— year
    focal_keys <- cell_dt[, .(focal_id = id, year, .row_idx)]
    edge_year <- edges[focal_keys, on = .(focal_id), allow.cartesian = TRUE,
                       nomatch = NULL]
    # edge_year now has: focal_id, neighbor_id, year, .row_idx

    # Merge neighbor values
    setnames(val_dt, "id", "neighbor_id")
    edge_year[val_dt, on = .(neighbor_id, year), neighbor_val := i.val]

    # Drop NAs in neighbor_val before aggregation
    edge_year_clean <- edge_year[!is.na(neighbor_val)]

    # Aggregate by focal row
    agg <- edge_year_clean[, .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ), by = .row_idx]

    # Name the new columns to match original pipeline convention
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # Initialize with NA, then fill matched rows
    set(cell_dt, j = max_col,  value = NA_real_)
    set(cell_dt, j = min_col,  value = NA_real_)
    set(cell_dt, j = mean_col, value = NA_real_)

    set(cell_dt, i = agg$.row_idx, j = max_col,  value = agg$nb_max)
    set(cell_dt, i = agg$.row_idx, j = min_col,  value = agg$nb_min)
    set(cell_dt, i = agg$.row_idx, j = mean_col, value = agg$nb_mean)

    # Clean up to free memory within the loop
    rm(val_dt, focal_keys, edge_year, edge_year_clean, agg)
    gc()

    cat(sprintf("  Done: %s\n", var_name))
  }

  # Remove helper column
  cell_dt[, .row_idx := NULL]

  invisible(cell_dt)
}


# ---- 3.3 CHUNKED RANDOM FOREST PREDICTION -----------------------------------

predict_rf_chunked <- function(model, newdata_dt, predictor_cols,
                                chunk_size = 500000L) {
  # model:          trained randomForest (or ranger) model object â€” NOT modified
  # newdata_dt:     data.table containing all predictor columns
  # predictor_cols: character vector of the ~110 predictor column names
  # chunk_size:     rows per chunk (tune to available RAM)
  #
  # Returns: numeric vector of predictions, same length as nrow(newdata_dt)

  n <- nrow(newdata_dt)
  predictions <- numeric(n)

  # Determine if this is a ranger or randomForest model

  is_ranger <- inherits(model, "ranger")

  starts <- seq(1L, n, by = chunk_size)
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","),
              length(starts),
              format(chunk_size, big.mark = ",")))

  for (k in seq_along(starts)) {
    i_start <- starts[k]
    i_end   <- min(i_start + chunk_size - 1L, n)

    # Extract chunk as a plain data.frame (some RF implementations require it)
    chunk_df <- as.data.frame(newdata_dt[i_start:i_end, ..predictor_cols])

    if (is_ranger) {
      preds <- predict(model, data = chunk_df)$predictions
    } else {
      # randomForest package
      preds <- predict(model, newdata = chunk_df)
    }

    predictions[i_start:i_end] <- preds

    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
                  k, length(starts),
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))
    }

    rm(chunk_df, preds)
    gc()
  }

  predictions
}


# ---- 3.4 FULL PIPELINE EXECUTION --------------------------------------------

run_optimized_pipeline <- function(cell_data,
                                    id_order,
                                    rook_neighbors_unique,
                                    rf_model,
                                    predictor_cols,
                                    chunk_size = 500000L) {
  # Convert to data.table (by reference if already data.table, else copy once)
  if (!is.data.table(cell_data)) {
    cell_dt <- as.data.table(cell_data)
  } else {
    cell_dt <- cell_data
  }

  # --- Feature preparation ---
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  cat("=== STEP 1: Neighbor feature computation ===\n")
  t0 <- proc.time()

  compute_all_neighbor_features_dt(
    cell_dt, id_order, rook_neighbors_unique, neighbor_source_vars
  )

  t1 <- proc.time()
  cat(sprintf("Feature prep time: %.1f seconds\n", (t1 - t0)["elapsed"]))

  # --- Prediction ---
  cat("\n=== STEP 2: Random Forest prediction ===\n")

  cell_dt[, predicted_gdp := predict_rf_chunked(
    model          = rf_model,
    newdata_dt     = cell_dt,
    predictor_cols = predictor_cols,
    chunk_size     = chunk_size
  )]

  t2 <- proc.time()
  cat(sprintf("Prediction time: %.1f seconds\n", (t2 - t1)["elapsed"]))
  cat(sprintf("Total pipeline time: %.1f seconds\n", (t2 - t0)["elapsed"]))

  # Return the augmented data.table
  cell_dt[]
}


# =============================================================================
# USAGE EXAMPLE (uncomment and adapt to your environment)
# =============================================================================
#
# library(randomForest)  # or library(ranger)
#
# # Load pre-trained model
# rf_model <- readRDS("trained_rf_model.rds")
#
# # Load data
# cell_data           <- readRDS("cell_data.rds")            # data.frame / data.table
# id_order            <- readRDS("id_order.rds")             # integer vector of cell IDs
# rook_neighbors_unique <- readRDS("rook_neighbors.rds")     # spdep::nb object
#
# # Define the ~110 predictor column names used during training
# predictor_cols <- readRDS("predictor_cols.rds")  # or names(rf_model$forest$xlevels), etc.
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data            = cell_data,
#   id_order             = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model             = rf_model,
#   predictor_cols       = predictor_cols,
#   chunk_size           = 500000L
# )
#
# # result now has a 'predicted_gdp' column
```

---

## 4. FURTHER MEMORY OPTIMIZATION (OPTIONAL)

If the cartesian join in `compute_all_neighbor_features_dt` causes memory pressure (the edge list Ã— 28 years can reach ~38M rows), process **year-by-year** inside the variable loop:

```r
# Alternative: year-chunked neighbor aggregation (lower peak memory)
compute_neighbor_features_by_year <- function(cell_dt, edges, var_name) {
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  set(cell_dt, j = max_col,  value = NA_real_)
  set(cell_dt, j = min_col,  value = NA_real_)
  set(cell_dt, j = mean_col, value = NA_real_)

  years <- sort(unique(cell_dt$year))

  for (yr in years) {
    # Subset focal rows for this year
    yr_idx <- which(cell_dt$year == yr)
    focal_sub <- cell_dt[yr_idx, .(focal_id = id, .row_idx = yr_idx)]

    # Join to edges
    edge_yr <- edges[focal_sub, on = .(focal_id), nomatch = NULL,
                     allow.cartesian = TRUE]

    # Get neighbor values for same year
    val_sub <- cell_dt[year == yr, .(neighbor_id = id, nval = get(var_name))]
    edge_yr[val_sub, on = .(neighbor_id), neighbor_val := i.nval]
    edge_yr <- edge_yr[!is.na(neighbor_val)]

    agg <- edge_yr[, .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ), by = .row_idx]

    set(cell_dt, i = agg$.row_idx, j = max_col,  value = agg$nb_max)
    set(cell_dt, i = agg$.row_idx, j = min_col,  value = agg$nb_min)
    set(cell_dt, i = agg$.row_idx, j = mean_col, value = agg$nb_mean)

    rm(focal_sub, edge_yr, val_sub, agg)
  }

  invisible(cell_dt)
}
```

---

## 5. KEY GUARANTEES

| Requirement | How It Is Met |
|---|---|
| **Trained RF model preserved** | Model is only passed to `predict()`; never modified or retrained |
| **Original numerical estimand preserved** | Same `max/min/mean` neighbor statistics, same predictor columns, same `predict()` call â€” output is numerically identical |
| **16 GB RAM feasibility** | Chunked prediction (~500K rows Ã— 110 cols â‰ˆ 440 MB per chunk); year-chunked neighbor option keeps edge tables small |
| **Correctness** | Vectorized `data.table` joins replicate the exact same lookup logic as the original `paste`-key approach |
END_RESPONSE>>>
