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
run_id: Anthropic_F03_T03_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - `as.character()` conversion and named-vector lookup (`id_to_ref`).
  - Subsetting `neighbors[[ref_idx]]` (an `nb` list element).
  - `paste()` to build composite keys.
  - Named-vector lookup into `idx_lookup` (a named character vector of length 6.46M â€” every lookup is **O(n) hash probe on a massive vector**).
  - `is.na` filtering and `as.integer` coercion.
- The result is a **list of 6.46 million integer vectors**, consuming enormous memory and taking hours to build because of per-row string operations and hash lookups on a 6.46M-entry named vector.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M elements, each extracting neighbor values, removing NAs, and computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3 million R-level function calls total.
- `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is itself slow (creates a huge temporary matrix row-by-row).

**Memory pressure**: The neighbor lookup list alone (6.46M entries, each a small integer vector) plus repeated `data.frame` column-binding (`compute_and_add_neighbor_features` likely does `cbind` or `$<-` on a 6.46M-row data.frame) causes repeated full-copy of the data.frame (R's copy-on-modify semantics). With ~110 columns Ã— 6.46M rows Ã— 8 bytes â‰ˆ 5.7 GB, each copy can exhaust 16 GB RAM and trigger swapping.

### B. Random Forest Inference Bottlenecks

- Calling `predict()` on a single large Random Forest model over 6.46M rows with 110 features is memory-intensive: `ranger` or `randomForest` must build the full prediction matrix and traverse every tree.
- If the model is a `randomForest` object (not `ranger`), prediction is single-threaded and slow.
- If prediction is done inside a loop (e.g., year-by-year or chunk-by-chunk) with repeated `predict()` calls, each call has overhead from data validation, factor-level checking, and matrix construction.
- Passing a `data.frame` to `predict()` instead of a `matrix` adds coercion overhead on every call.

### C. Summary of Root Causes

| Root Cause | Impact |
|---|---|
| Per-row `paste`/string-key lookups in `build_neighbor_lookup` | ~hours on 6.46M rows |
| Named-vector lookup on 6.46M-entry vector (hash table) | Slow per-probe |
| `lapply` + R-level loops for neighbor stats (Ã—5 vars) | ~hours |
| `do.call(rbind, ...)` on millions of list elements | Slow matrix assembly |
| Copy-on-modify of large data.frame when adding columns | RAM exhaustion, swapping |
| Possible single-threaded RF prediction (`randomForest::predict`) | Hours for 6.46M rows |
| Possible row-by-row or small-batch prediction loop | Overhead per call |

---

## 2. Optimization Strategy

### Feature Preparation: Vectorize Everything with `data.table`

1. **Replace the neighbor lookup list** with a flat `data.table` join. Instead of building a 6.46M-element list, create a long-form neighbor-pair table (`cell_row â†’ neighbor_row`) using vectorized integer arithmetic. Avoid all `paste()`/string keys â€” use integer-keyed joins on `(id, year)`.

2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation: join the neighbor-pair table to the variable column, then `[, .(max, min, mean), by = cell_row]`. This replaces 6.46M R-level function calls with a single vectorized C-level operation.

3. **Use `data.table` in-place column assignment (`:=`)** to add the 15 new columns (5 vars Ã— 3 stats) without copying the entire table.

### Random Forest Inference: Batch Prediction with Matrix Input

4. **Convert to matrix once** before prediction. Avoid repeated `data.frame` â†’ `matrix` coercion.

5. **If the model is `randomForest`**: convert it to `ranger` format using the same trees (not possible directly), OR chunk the prediction into blocks of ~500K rows to control peak memory, OR simply accept single-threaded prediction but ensure the input is a pre-built numeric matrix.

6. **If the model is `ranger`**: use `num.threads` to parallelize prediction.

7. **Predict in moderately large chunks** (~500Kâ€“1M rows) to stay within RAM while minimizing per-call overhead.

### Memory Management

8. Target peak RAM â‰ˆ 8â€“10 GB (within 16 GB). The `data.table` approach avoids copies. Chunked prediction avoids materializing all tree outputs simultaneously.

### Expected Speedup

| Stage | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~20â€“40 hrs | ~1â€“3 min |
| `compute_neighbor_stats` (Ã—5) | ~30â€“40 hrs | ~2â€“5 min |
| Column binding / copies | ~hours (swapping) | ~seconds (`:=`) |
| RF prediction (6.46M rows) | ~1â€“6 hrs | ~10â€“30 min |
| **Total** | **86+ hrs** | **~15â€“40 min** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# =============================================================================
# Requirements: data.table, ranger (or randomForest), Matrix (optional)
# Preserves: trained RF model object, original numerical estimand (GDP)
# =============================================================================

library(data.table)

# ---- STEP 0: Convert cell_data to data.table (in-place, no copy) -----------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure id and year are integer for fast keyed joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Add a row index (will be used as the primary key for aggregation)
cell_data[, .row_idx := .I]


# ---- STEP 1: Build flat neighbor-pair table (vectorized, no strings) --------
# rook_neighbors_unique is an nb object: a list of length = # cells,
# where element i contains integer indices of neighbors of cell i
# id_order is the vector mapping position -> cell id

build_neighbor_pairs_dt <- function(cell_data, id_order, neighbors) {
  # id_order[i] is the cell id at position i in the nb object
  # neighbors[[i]] contains integer positions of neighbors of cell at position i

  n_cells <- length(id_order)

  # Map: cell_id -> position in nb object
  id_to_pos <- integer(max(id_order))
  id_to_pos[id_order] <- seq_len(n_cells)

  # Build flat edge list: (focal_cell_id, neighbor_cell_id)
  # Vectorized expansion of the nb list
  n_neighbors <- lengths(neighbors)  # integer vector, length = n_cells
  focal_pos   <- rep.int(seq_len(n_cells), n_neighbors)
  nbr_pos     <- unlist(neighbors, use.names = FALSE)

  focal_ids <- id_order[focal_pos]
  nbr_ids   <- id_order[nbr_pos]

  # Create edge table (cell-level, year-independent)
  edge_dt <- data.table(
    focal_id = as.integer(focal_ids),
    nbr_id   = as.integer(nbr_ids)
  )
  rm(focal_pos, nbr_pos, focal_ids, nbr_ids)

  # Now cross-join with years to get (focal_id, year, nbr_id, year) pairs

# But that would be 1.37M edges Ã— 28 years = 38.5M rows â€” manageable.
  # Instead, join edges to cell_data rows for both focal and neighbor.

  # Key cell_data for fast join
  setkey(cell_data, id, year)

  # Get unique years
  years <- sort(unique(cell_data$year))

  # Expand edges across all years
  # CJ-like expansion: each edge exists for every year
  year_dt <- data.table(year = as.integer(years))
  edge_year_dt <- edge_dt[, CJ_wrapper := TRUE][
    year_dt, on = "CJ_wrapper", allow.cartesian = TRUE
  ]
  # Cleaner approach:
  edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years)
  edge_year_dt[, focal_id := edge_dt$focal_id[edge_idx]]
  edge_year_dt[, nbr_id   := edge_dt$nbr_id[edge_idx]]
  edge_year_dt[, edge_idx := NULL]

  # Join to get focal row index
  focal_map <- cell_data[, .(id, year, focal_row = .row_idx)]
  setkey(focal_map, id, year)
  edge_year_dt <- focal_map[edge_year_dt, on = .(id = focal_id, year = year),
                            nomatch = NULL]
  setnames(edge_year_dt, "id", "focal_id")

  # Join to get neighbor row index
  nbr_map <- cell_data[, .(id, year, nbr_row = .row_idx)]
  setkey(nbr_map, id, year)
  edge_year_dt <- nbr_map[edge_year_dt, on = .(id = nbr_id, year = year),
                           nomatch = NULL]
  setnames(edge_year_dt, "id", "nbr_id")

  # Result: data.table with columns (focal_row, nbr_row) â€” all we need

  edge_year_dt[, .(focal_row, nbr_row)]
}

# --- More memory-efficient version (avoids 38M-row CJ) ---
build_neighbor_pairs_dt <- function(cell_data, id_order, neighbors) {

  n_cells <- length(id_order)

  # Flat edge list at cell level
  n_nbrs    <- lengths(neighbors)
  focal_pos <- rep.int(seq_len(n_cells), n_nbrs)
  nbr_pos   <- unlist(neighbors, use.names = FALSE)

  focal_ids <- as.integer(id_order[focal_pos])
  nbr_ids   <- as.integer(id_order[nbr_pos])

  edge_dt <- data.table(focal_id = focal_ids, nbr_id = nbr_ids)
  rm(focal_pos, nbr_pos, focal_ids, nbr_ids)

  # Build a row-index lookup: (id, year) -> .row_idx
  row_map <- cell_data[, .(id, year, .row_idx)]
  setkey(row_map, id, year)

  # For each edge, we need to replicate across all years where BOTH

  # focal and neighbor exist. Use a merge-based approach:
  # Join edge_dt to row_map for focal side, then join for neighbor side.

  # Focal join: get all (focal_id, year, focal_row) combinations
  setkey(edge_dt, focal_id)
  # Expand: for each edge, find all years the focal cell has data
  focal_expanded <- row_map[edge_dt,
                            on = .(id = focal_id),
                            allow.cartesian = TRUE,
                            nomatch = NULL]
  # Columns: id (=focal_id), year, .row_idx (=focal_row), nbr_id
  setnames(focal_expanded, c("id", ".row_idx"), c("focal_id", "focal_row"))

  # Neighbor join: match (nbr_id, year) to get nbr_row
  setkey(focal_expanded, nbr_id, year)
  setkey(row_map, id, year)

  pair_dt <- row_map[focal_expanded,
                     on = .(id = nbr_id, year = year),
                     nomatch = NULL]
  # Columns: id (=nbr_id), year, .row_idx (=nbr_row), focal_id, focal_row
  setnames(pair_dt, c("id", ".row_idx"), c("nbr_id", "nbr_row"))

  pair_dt[, .(focal_row = as.integer(focal_row),
              nbr_row   = as.integer(nbr_row))]
}

cat("Building neighbor pair table...\n")
system.time({
  pair_dt <- build_neighbor_pairs_dt(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~38M rows, ~300 MB, built in 1-3 minutes
cat(sprintf("Neighbor pairs: %s rows\n", format(nrow(pair_dt), big.mark = ",")))


# ---- STEP 2: Compute all neighbor features (vectorized) --------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))

    # Extract the variable values for all neighbor rows
    pair_dt[, nbr_val := cell_data[[var_name]][nbr_row]]

    # Aggregate: max, min, mean per focal_row (excluding NAs)
    stats <- pair_dt[!is.na(nbr_val),
                     .(v_max  = max(nbr_val),
                       v_min  = min(nbr_val),
                       v_mean = mean(nbr_val)),
                     by = focal_row]

    # Assign back to cell_data using := (in-place, no copy)
    max_col  <- paste0(var_name, "_max_neighbor")
    min_col  <- paste0(var_name, "_min_neighbor")
    mean_col <- paste0(var_name, "_mean_neighbor")

    # Initialize with NA
    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Fill in computed values
    cell_data[stats$focal_row, (max_col)  := stats$v_max]
    cell_data[stats$focal_row, (min_col)  := stats$v_min]
    cell_data[stats$focal_row, (mean_col) := stats$v_mean]

    # Clean up the temporary column
    pair_dt[, nbr_val := NULL]
    rm(stats)
  }
})
# Expected: ~2-5 minutes total for all 5 variables

# Clean up
cell_data[, .row_idx := NULL]
rm(pair_dt)
gc()


# ---- STEP 3: Random Forest Prediction (chunked, memory-safe) ---------------

# Detect model type and set up prediction accordingly
predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)

  # Determine model class
  model_class <- class(model)[1]
  cat(sprintf("Model class: %s | Rows: %s | Chunks: %d\n",
              model_class, format(n, big.mark = ","), n_chunks))

  # For ranger: ensure num.threads is set for parallel prediction
  is_ranger <- inherits(model, "ranger")

  # Pre-convert to matrix if all predictors are numeric (avoids per-chunk coercion)
  # Identify predictor columns (exclude response, id, year if present)
  # This depends on your model; adjust as needed.

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    idx       <- start_row:end_row

    chunk <- newdata[idx, ]

    if (is_ranger) {
      pred <- predict(model, data = chunk, num.threads = parallel::detectCores())
      predictions[idx] <- pred$predictions
    } else {
      # randomForest
      pred <- predict(model, newdata = chunk)
      predictions[idx] <- as.numeric(pred)
    }

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %s-%s)\n",
                  i, n_chunks,
                  format(start_row, big.mark = ","),
                  format(end_row, big.mark = ",")))
    }
  }

  predictions
}

# Prepare prediction data: select only predictor columns, convert to data.table
# Adjust 'predictor_cols' to match your trained model's expected features
# Example: if the model was trained on all columns except "id", "year", "gdp":
response_col <- "gdp"  # adjust to your actual response variable name
exclude_cols <- c("id", "year", response_col)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# Verify all expected predictors are present
if (inherits(rf_model, "ranger")) {
  expected_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  expected_vars <- rownames(rf_model$importance)
} else {
  expected_vars <- predictor_cols
}

missing_vars <- setdiff(expected_vars, names(cell_data))
if (length(missing_vars) > 0) {
  warning(sprintf("Missing predictor variables: %s",
                  paste(missing_vars, collapse = ", ")))
}

# Build prediction input (only needed columns, as data.table)
pred_input <- cell_data[, .SD, .SDcols = intersect(expected_vars, names(cell_data))]

cat("Starting Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model      = rf_model,
    newdata    = pred_input,
    chunk_size = 500000L
  )]
})

rm(pred_input)
gc()

cat("Pipeline complete.\n")
cat(sprintf("Predictions: %s rows\n",
            format(sum(!is.na(cell_data$predicted_gdp)), big.mark = ",")))


# =============================================================================
# OPTIONAL: If model is randomForest and prediction is still too slow,
# convert prediction input to a matrix (avoids internal data.frame -> matrix
# coercion on every predict() call)
# =============================================================================

# pred_matrix <- as.matrix(pred_input)
# Then in predict_rf_chunked, pass pred_matrix[idx, , drop = FALSE] to predict()
```

---

## 4. Summary of Key Changes

| Original | Optimized | Why |
|---|---|---|
| `paste(id, year)` string keys for 6.46M rows | Integer-keyed `data.table` joins on `(id, year)` | Eliminates millions of string allocations and hash lookups |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Vectorized `data.table` cartesian join + merge | Single C-level operation replaces 6.46M R function calls |
| `lapply` over 6.46M rows in `compute_neighbor_stats` (Ã—5) | `data.table` grouped aggregation `[, .(max, min, mean), by = focal_row]` | Vectorized C-level groupby replaces 32.3M R function calls |
| `do.call(rbind, list_of_6.46M_vectors)` | `data.table` returns result directly as columns | No row-by-row matrix assembly |
| `data.frame$new_col <- ...` (copy-on-modify, ~5.7 GB copy each time) | `data.table[, (col) := val]` in-place assignment | Zero-copy column addition; stays within 16 GB RAM |
| Possible single `predict()` call on 6.46M rows | Chunked prediction (500K rows/chunk) with parallel threads | Controls peak memory; enables `ranger` multi-threading |
| Possible `data.frame` input to `predict()` | Pre-selected `data.table` / optional matrix conversion | Avoids repeated internal coercion |

**Estimated total runtime: ~15â€“40 minutes** (down from 86+ hours), depending on RF model size and whether `ranger` or `randomForest` is used. The trained model and original GDP estimand are fully preserved â€” no retraining or transformation of the response variable occurs.
END_RESPONSE>>>
