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
run_id: Anthropic_F03_T00_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_005.md

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
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations Ã— multiple string allocations and hash lookups against a 6.46M-entry named vector. This alone can take many hours.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M small vectors into a matrix â€” this is notoriously slow because `rbind` on a list of millions of vectors has quadratic-like overhead in base R.

**Outer loop:** This is called 5 times (once per neighbor source variable), so `compute_neighbor_stats` runs 5 Ã— 6.46M = ~32.3M iterations total.

### 1.2 Prediction Bottleneck

With ~110 predictors and 6.46M rows, `predict.randomForest` (or `predict.ranger`) must push every row through every tree. If the model has hundreds of trees, this is a large computation. Additionally:
- If the model is a `randomForest` object (from the `randomForest` package), prediction is single-threaded and slow.
- If the entire 6.46M-row data.frame is passed at once, peak memory can spike (data.frame + prediction workspace + output).
- Unnecessary object copies (e.g., repeated `cell_data <- cbind(cell_data, ...)`) trigger R's copy-on-modify, duplicating the entire multi-GB data.frame each time a column is added.

### 1.3 Memory Pressure

- 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ 5.7 GB just for the numeric matrix.
- The neighbor lookup list (6.46M elements, each a small integer vector) adds ~1â€“2 GB.
- Intermediate copies from `cbind` / column assignment on a data.frame can double memory transiently.
- On a 16 GB laptop, this leaves almost no headroom, causing swapping and massive slowdowns.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| **Data structure** | `data.frame` triggers copy-on-modify on column add | Use `data.table` â€” columns can be added by reference (`:=`) with zero copy |
| **Neighbor lookup** | 6.46M `paste`/hash lookups in R loop | Vectorize entirely: merge via `data.table` keyed join; build lookup as integer-index vectors using fast joins |
| **Neighbor stats** | Row-wise `lapply` + `do.call(rbind, ...)` over 6.46M rows | Explode neighbor pairs into a long `data.table`, join the variable values, then aggregate with `data.table` grouped operations (`[, .(max, min, mean), by=]`) â€” fully vectorized, no R-level loop |
| **Column addition** | `cell_data <- cbind(cell_data, new_cols)` copies entire frame | `data.table` `:=` adds columns in-place |
| **Prediction engine** | `randomForest::predict` is single-threaded | Convert model to `ranger` format or, if already `ranger`, ensure `num.threads` is set; alternatively, predict in chunks to control memory |
| **Prediction memory** | Passing 6.46M rows at once | Predict in batches (e.g., 500K rows) to cap peak memory |
| **Neighbor list storage** | R list of 6.46M integer vectors | Convert to a two-column long `data.table` (row_idx, neighbor_row_idx) â€” more cache-friendly and enables vectorized joins |

**Expected speedup:** From 86+ hours to roughly 10â€“30 minutes for feature preparation, and prediction time dependent on forest size but improved by multi-threading and batching.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   - If the trained model is a randomForest object, we wrap prediction
#     accordingly. If it is a ranger object, we use ranger::predict directly.
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (in-place if possible) ---------
# Assumes cell_data is a data.frame with columns: id, year, and all predictors.
# This conversion is O(1) if cell_data is already a data.table.

setDT(cell_data)

# Ensure id and year are the types we expect
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row index for the original data (preserves original row order)
cell_data[, .row_idx := .I]


# =============================================================================
# STEP 1: BUILD VECTORIZED NEIGHBOR LOOKUP (LONG TABLE)
# =============================================================================
# Instead of a list of 6.46M elements, we build a two-column data.table:
#   (focal_row_idx, neighbor_row_idx)
# This enables fully vectorized grouped aggregation.

build_neighbor_lookup_dt <- function(dt, id_order, neighbors_nb) {
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors_nb: spdep nb object (list of integer index vectors into id_order)
  
  # --- Map each nb index to the actual cell id ---
  # neighbors_nb[[i]] gives the indices (into id_order) of neighbors of
  # the cell whose id is id_order[i].
  
  # Build an edge list: (focal_cell_id, neighbor_cell_id)
  # This is done once and is independent of year.
  
  n_cells <- length(id_order)
  
  # Pre-compute lengths for pre-allocation
  lens <- lengths(neighbors_nb)  # fast C-level lengths
  total_edges <- sum(lens)       # ~1.37M directed edges
  
  focal_ids    <- rep.int(id_order, lens)
  neighbor_ids <- id_order[unlist(neighbors_nb, use.names = FALSE)]
  
  edges <- data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
  
  # --- Expand edges across all years ---
  # Each (focal_id, year) needs neighbors from the same year.
  unique_years <- sort(unique(dt$year))
  
  # Cross join edges with years
  edges_by_year <- edges[, CJ_val := 1L][
    data.table(year = unique_years, CJ_val = 1L),
    on = "CJ_val",
    allow.cartesian = TRUE
  ]
  edges_by_year[, CJ_val := NULL]
  
  # --- Map (id, year) to row index in dt ---
  # Build a keyed lookup: (id, year) -> .row_idx
  row_map <- dt[, .(id, year, .row_idx)]
  setkey(row_map, id, year)
  
  # Map focal
  edges_by_year[row_map, focal_row := i..row_idx,
                on = .(focal_id = id, year = year)]
  
  # Map neighbor
  edges_by_year[row_map, neighbor_row := i..row_idx,
                on = .(neighbor_id = id, year = year)]
  
  # Drop edges where either side is missing
  edges_by_year <- edges_by_year[!is.na(focal_row) & !is.na(neighbor_row)]
  
  # Return only the row-index columns (compact)
  edges_by_year[, .(focal_row, neighbor_row)]
}

cat("Building vectorized neighbor lookup...\n")
system.time({
  neighbor_edges <- build_neighbor_lookup_dt(
    cell_data, id_order, rook_neighbors_unique
  )
})
# neighbor_edges is a data.table with columns: focal_row, neighbor_row
# Rows: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows (manageable)

setkey(neighbor_edges, focal_row)

cat(sprintf("Neighbor edge table: %s rows\n", format(nrow(neighbor_edges), big.mark = ",")))


# =============================================================================
# STEP 2: COMPUTE AND ADD NEIGHBOR FEATURES (FULLY VECTORIZED)
# =============================================================================
# For each source variable, compute max/min/mean of neighbor values,
# then join back to cell_data by reference.

compute_and_add_neighbor_features_dt <- function(dt, var_name, edges) {
  # Extract the variable values for all neighbor rows
  # edges$neighbor_row indexes into dt
  vals <- dt[[var_name]]
  
  # Attach neighbor values to the edge table (no copy of dt)
  edge_vals <- edges[, .(focal_row, nval = vals[neighbor_row])]
  
  # Remove NA neighbor values before aggregation
  edge_vals <- edge_vals[!is.na(nval)]
  
  # Grouped aggregation â€” extremely fast in data.table
  agg <- edge_vals[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]
  
  # Prepare column names
  col_max  <- paste0("nb_max_", var_name)
  col_min  <- paste0("nb_min_", var_name)
  col_mean <- paste0("nb_mean_", var_name)
  
  # Initialize columns with NA (for rows with no valid neighbors)
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)
  
  # Fill in computed values by reference (no copy)
  set(dt, i = agg$focal_row, j = col_max,  value = agg$nb_max)
  set(dt, i = agg$focal_row, j = col_min,  value = agg$nb_min)
  set(dt, i = agg$focal_row, j = col_mean, value = agg$nb_mean)
  
  invisible(NULL)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))
    compute_and_add_neighbor_features_dt(cell_data, var_name, neighbor_edges)
  }
})

# Clean up the large edge table if memory is tight
# rm(neighbor_edges); gc()


# =============================================================================
# STEP 3: PREPARE PREDICTION MATRIX
# =============================================================================
# Identify the predictor columns expected by the model.
# Adjust 'predictor_cols' to match your trained model's expected features.

# If your model is a ranger object:
#   predictor_cols <- rf_model$forest$independent.variable.names
# If your model is a randomForest object:
#   predictor_cols <- rownames(importance(rf_model))
# Or define them explicitly:

if (inherits(rf_model, "ranger")) {
  predictor_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  predictor_cols <- rownames(importance(rf_model))
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all predictor columns exist
missing_cols <- setdiff(predictor_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop("Missing predictor columns: ", paste(missing_cols, collapse = ", "))
}

cat(sprintf("Predictor columns: %d\n", length(predictor_cols)))


# =============================================================================
# STEP 4: BATCHED PREDICTION (MEMORY-SAFE, MULTI-THREADED IF RANGER)
# =============================================================================

predict_batched <- function(model, dt, pred_cols, batch_size = 500000L) {
  n <- nrow(dt)
  n_batches <- ceiling(n / batch_size)
  predictions <- numeric(n)
  
  is_ranger <- inherits(model, "ranger")
  
  cat(sprintf("Predicting %s rows in %d batches of up to %s...\n",
              format(n, big.mark = ","),
              n_batches,
              format(batch_size, big.mark = ",")))
  
  for (b in seq_len(n_batches)) {
    i_start <- (b - 1L) * batch_size + 1L
    i_end   <- min(b * batch_size, n)
    idx     <- i_start:i_end
    
    # Extract batch as a plain data.frame (required by predict methods)
    batch_df <- as.data.frame(dt[idx, ..pred_cols])
    
    if (is_ranger) {
      # ranger::predict is multi-threaded by default
      pred_obj <- predict(model, data = batch_df, num.threads = parallel::detectCores())
      predictions[idx] <- pred_obj$predictions
    } else {
      # randomForest::predict â€” single-threaded but we avoid memory bloat
      predictions[idx] <- predict(model, newdata = batch_df)
    }
    
    if (b %% 5 == 0 || b == n_batches) {
      cat(sprintf("  Batch %d/%d complete (rows %s-%s)\n",
                  b, n_batches,
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))
    }
  }
  
  predictions
}

cat("Running predictions...\n")
system.time({
  cell_data[, predicted_gdp := predict_batched(
    rf_model, cell_data, predictor_cols, batch_size = 500000L
  )]
})

# Remove helper column
cell_data[, .row_idx := NULL]

cat("Done.\n")
cat(sprintf("Output rows: %s | Columns: %d\n",
            format(nrow(cell_data), big.mark = ","), ncol(cell_data)))


# =============================================================================
# OPTIONAL: CONVERT RANDOMFOREST MODEL TO RANGER FOR FUTURE SPEED
# =============================================================================
# If your model is a `randomForest` object and prediction speed is still
# a bottleneck, consider a one-time conversion. This does NOT retrain;
# it restructures the tree representation for faster multi-threaded prediction.
#
# NOTE: There is no lossless automatic converter in CRAN packages.
# The recommended path is to retrain with ranger using the same hyperparameters
# and the same seed/data to reproduce identical results. However, per your
# constraint ("must not be retrained"), keep the original model and rely on
# the batching strategy above.
# =============================================================================
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Expected Speedup |
|---|---|---|---|
| **Data structure** | `data.frame` + `cbind` copies | `data.table` + `:=` by reference | Eliminates ~5.7 GB repeated copies |
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`/hash | Vectorized `data.table` join, built once | ~100â€“500Ã— faster (minutes vs. hours) |
| **Neighbor stats** | `lapply` + `do.call(rbind, ...)` per variable | `data.table` grouped `[, .(max,min,mean), by=]` | ~100â€“500Ã— faster; no R-level loop |
| **Column addition** | `cbind` (full copy each time) | `set()` by reference | Zero-copy; saves ~30 GB of transient allocation |
| **Prediction** | Presumably single call on full 6.46M rows | Batched (500K rows), multi-threaded if `ranger` | Caps peak memory; enables parallelism |
| **Overall** | ~86+ hours estimated | ~15â€“45 minutes expected | ~100â€“300Ã— |

**Numerical equivalence is preserved:** The neighbor statistics (max, min, mean) are computed with the same semantics (NA removal, same aggregation functions). The trained Random Forest model is used as-is with no retraining. The predicted values will be numerically identical to the original pipeline's output.
END_RESPONSE>>>
