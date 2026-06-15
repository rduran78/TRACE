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
run_id: Anthropic_F03_T07_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows. For each row it:
- Performs character coercion and named-vector lookups (`id_to_ref`, `idx_lookup`) â€” these are O(n) hash lookups but repeated 6.46M times with string pasting and `paste(..., sep="_")` allocation each iteration.
- Creates intermediate character vectors (`neighbor_keys`) per row.
- Net effect: millions of small allocations, string concatenations, and named-vector lookups. This alone can take **hours**.

**`compute_neighbor_stats`:** Called 5 times (once per source variable). Each call iterates over 6.46M entries in `neighbor_lookup`, subsetting a numeric vector and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M 3-element vectors is extremely slow â€” `do.call(rbind, list_of_6.46M_vectors)` alone creates a massive temporary list-to-matrix conversion.

**Outer loop:** Calls `compute_and_add_neighbor_features` 5 times, each presumably calling `compute_neighbor_stats`. Each call copies the entire `cell_data` data.frame when assigning new columns (`cell_data <- ...`), triggering R's copy-on-modify semantics on a ~6.46M Ã— 110+ column object.

### 1.2 Random Forest Inference Bottlenecks

- Predicting 6.46M rows Ã— 110 features through a Random Forest (likely `ranger` or `randomForest`) in a single `predict()` call can require **massive memory** (the model object itself + prediction workspace). On 16 GB RAM this can cause swapping.
- If `randomForest::predict.randomForest` is used (rather than `ranger`), it is single-threaded and substantially slower.
- If prediction is done row-by-row or in a naive loop, that compounds the problem enormously.

### 1.3 Memory Pressure

- 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB** just for the numeric matrix. With R's overhead, copies, and the model in memory, 16 GB is tight. Any unnecessary copy doubles consumption and triggers GC thrashing or swapping.

### Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named-vector lookups | ~hours |
| `compute_neighbor_stats` | `lapply` + `do.call(rbind, ...)` over 6.46M elements, called 5Ã— | ~hours |
| Column assignment in loop | Copy-on-modify of full data.frame 5Ã— | ~tens of minutes + RAM spikes |
| RF prediction | Possibly single-threaded, possibly un-batched, full matrix copy | ~hours |
| Overall memory | Repeated large copies on 16 GB machine | Swapping, GC thrashing |

---

## 2. OPTIMIZATION STRATEGY

### A. Replace data.frame with `data.table` (eliminate copy-on-modify)

`data.table` supports **in-place column addition** via `:=`, eliminating the repeated ~5.7 GB copies.

### B. Vectorize `build_neighbor_lookup` entirely

Instead of building a per-row R list (6.46M entries), build a **flat edge-list** (a two-column integer matrix: `from_row â†’ to_row`) using vectorized operations. This replaces 6.46M `paste` + lookup iterations with a single vectorized join.

### C. Vectorize `compute_neighbor_stats` with `data.table` grouped operations

Use the flat edge-list as a `data.table`, join in the variable values, and compute `max/min/mean` grouped by `from_row` â€” fully vectorized C-level aggregation.

### D. Batch RF prediction with `ranger`

- If the model is `randomForest`, convert it or re-wrap prediction.
- If `ranger`, use `predict()` with `num.threads` and process in **chunks** (~500K rows) to control peak memory.

### E. Memory discipline

- Remove intermediate objects aggressively (`rm()` + `gc()`).
- Convert prediction input to a matrix once, predict in chunks, write results back.

### Projected speedup

| Component | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~2â€“4 hrs | ~20â€“60 sec |
| `compute_neighbor_stats` Ã—5 | ~3â€“6 hrs | ~2â€“5 min total |
| Column assignment Ã—5 | ~30 min + RAM | ~seconds (in-place) |
| RF prediction | ~1â€“4 hrs | ~5â€“20 min |
| **Total** | **~86+ hrs** | **~30â€“60 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest), spdep (for nb object)
# =============================================================================

library(data.table)

# ---- 0. Convert cell_data to data.table (if not already) --------------------
#    Assumes cell_data is a data.frame/data.table with columns: id, year, ...
#    Assumes rook_neighbors_unique is an nb object (list of integer index vectors)
#    Assumes id_order is the vector of cell IDs corresponding to nb indices
#    Assumes rf_model is the pre-trained Random Forest model (ranger or randomForest)

if (!is.data.table(cell_data)) {
  setDT(cell_data)  # convert in place â€” no copy
}

# ---- 1. VECTORIZED NEIGHBOR LOOKUP (flat edge-list) -------------------------

build_neighbor_edgelist <- function(dt, id_order, nb_obj) {
  # Map each nb index to its cell id
  # nb_obj[[k]] gives the neighbor indices (in id_order) for cell id_order[k]

  n_cells <- length(id_order)
  stopifnot(length(nb_obj) == n_cells)

  # Build cell-level edge list: from_cell_id -> to_cell_id
  from_cell <- rep(
    id_order,
    times = lengths(nb_obj)
  )
  to_cell <- id_order[unlist(nb_obj, use.names = FALSE)]

  cell_edges <- data.table(from_id = from_cell, to_id = to_cell)

  # Build row-index lookup: (id, year) -> row index in dt
  dt[, .row_idx := .I]

  # For each row in dt, we need to find its neighbors' rows (same year).
  # Strategy: join cell_edges with dt twice â€” once for "from" rows, once for "to" rows.

  # Unique years
  years <- unique(dt$year)

  # Create a keyed lookup: id, year -> row_idx
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Expand cell_edges across all years (vectorized cross-join)
  # This creates the full (from_row, to_row) edge list.

  # More memory-efficient: iterate by year (28 years â€” trivial loop)
  edge_list_parts <- vector("list", length(years))

  for (yi in seq_along(years)) {
    yr <- years[yi]
    # Row indices for this year
    yr_lookup <- row_lookup[year == yr]
    setkey(yr_lookup, id)

    # Join from_id to get from_row
    merged <- cell_edges[yr_lookup, on = .(from_id = id), nomatch = 0L,
                         .(from_row = i..row_idx, to_id)]

    # Join to_id to get to_row
    setkey(yr_lookup, id)
    merged <- yr_lookup[merged, on = .(id = to_id), nomatch = 0L,
                        .(from_row, to_row = .row_idx)]

    edge_list_parts[[yi]] <- merged
  }

  edge_dt <- rbindlist(edge_list_parts, use.names = TRUE)

  # Clean up temporary column
  dt[, .row_idx := NULL]

  return(edge_dt)
}

cat("Building vectorized neighbor edge list...\n")
t0 <- proc.time()
edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("  Edge list built: %d edges in %.1f sec\n",
            nrow(edge_dt), (proc.time() - t0)[3]))


# ---- 2. VECTORIZED NEIGHBOR STATS -------------------------------------------

compute_and_add_neighbor_features_fast <- function(dt, var_name, edge_dt) {
  # Pull the variable values for the "to" (neighbor) rows
  vals <- dt[[var_name]]
  edge_dt[, nval := vals[to_row]]

  # Drop NAs before aggregation
  valid_edges <- edge_dt[!is.na(nval)]

  # Grouped aggregation â€” fully vectorized at C level
  stats <- valid_edges[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = from_row]

  # Prepare output columns (default NA)
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Assign in-place using data.table := (no copy of dt)
  dt[, (max_col)  := NA_real_]
  dt[, (min_col)  := NA_real_]
  dt[, (mean_col) := NA_real_]

  dt[stats$from_row, (max_col)  := stats$nb_max]
  dt[stats$from_row, (min_col)  := stats$nb_min]
  dt[stats$from_row, (mean_col) := stats$nb_mean]

  # Clean temp column from edge_dt
  edge_dt[, nval := NULL]

  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
t0 <- proc.time()
for (var_name in neighbor_source_vars) {
  t1 <- proc.time()
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_dt)
  cat(sprintf("  %s done in %.1f sec\n", var_name, (proc.time() - t1)[3]))
}
cat(sprintf("All neighbor features done in %.1f sec\n", (proc.time() - t0)[3]))

# Free the edge list if no longer needed
rm(edge_dt)
gc()


# ---- 3. BATCHED RANDOM FOREST PREDICTION ------------------------------------

predict_rf_batched <- function(model, dt, feature_cols, batch_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)

  # Determine if model is ranger or randomForest
  is_ranger <- inherits(model, "ranger")

  # Pre-select feature columns as a matrix (or data.frame for randomForest)
  # Process in chunks to control memory
  n_batches <- ceiling(n / batch_size)

  cat(sprintf("Predicting %d rows in %d batches (batch_size=%d)...\n",
              n, n_batches, batch_size))

  for (b in seq_len(n_batches)) {
    idx_start <- (b - 1L) * batch_size + 1L
    idx_end   <- min(b * batch_size, n)
    idx       <- idx_start:idx_end

    # Extract batch â€” data.table subsetting is efficient
    batch_df <- dt[idx, ..feature_cols]

    if (is_ranger) {
      # ranger::predict returns a list with $predictions
      batch_pred <- predict(model, data = batch_df,
                            num.threads = parallel::detectCores(logical = FALSE))$predictions
    } else {
      # randomForest::predict
      batch_pred <- predict(model, newdata = batch_df)
    }

    preds[idx] <- batch_pred

    if (b %% 5 == 0 || b == n_batches) {
      cat(sprintf("  Batch %d/%d complete (rows %d-%d)\n",
                  b, n_batches, idx_start, idx_end))
    }

    # Free batch memory
    rm(batch_df, batch_pred)
    if (b %% 10 == 0) gc()
  }

  return(preds)
}

# Identify feature columns (all predictors used by the model)
# Adjust this to match your actual feature column names:
if (inherits(rf_model, "ranger")) {
  feature_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores predictor names in the model
  feature_cols <- attr(rf_model$terms, "term.labels")
  if (is.null(feature_cols)) {
    # If trained with x/y interface, rownames of importance
    feature_cols <- rownames(rf_model$importance)
  }
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features exist
missing_cols <- setdiff(feature_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop("Missing feature columns in cell_data: ",
       paste(missing_cols, collapse = ", "))
}

cat("Starting Random Forest prediction...\n")
t0 <- proc.time()
cell_data[, predicted_gdp := predict_rf_batched(rf_model, cell_data,
                                                 feature_cols,
                                                 batch_size = 500000L)]
cat(sprintf("Prediction complete in %.1f sec\n", (proc.time() - t0)[3]))


# ---- 4. FINAL OUTPUT --------------------------------------------------------
# cell_data now has the column `predicted_gdp` with the original numerical
# estimand preserved (same RF model, same features, same predictions).

cat(sprintf("Done. Output: %d rows, %d columns\n", nrow(cell_data), ncol(cell_data)))
cat(sprintf("Prediction range: [%.4f, %.4f]\n",
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

| Decision | Rationale |
|---|---|
| **`setDT()` in place** | Zero-copy conversion; all subsequent `:=` assignments are in-place (no 5.7 GB copies). |
| **Flat edge-list instead of per-row list** | Eliminates 6.46M `paste`/string-lookup iterations. The 28-year loop (trivial overhead) keeps peak memory bounded. |
| **`data.table` grouped aggregation** | `max`/`min`/`mean` by group uses data.table's radix-sort + C-level aggregation â€” orders of magnitude faster than `lapply` + `do.call(rbind, ...)`. |
| **Batched prediction (500K rows)** | Keeps peak memory for the prediction matrix at ~500K Ã— 110 Ã— 8 bytes â‰ˆ 440 MB per batch, well within 16 GB even with the model loaded. |
| **`num.threads` for ranger** | Exploits all physical cores for tree-level parallelism during inference. |
| **Model is never retrained** | Only `predict()` is called; the model object is read-only throughout. |
| **Numerical estimand preserved** | Same model + same features + same `predict()` call = identical numerical output. |
END_RESPONSE>>>
