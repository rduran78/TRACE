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
run_id: Anthropic_F03_T07_R018
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 18
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_018.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string operations**

The function creates ~6.46 million character-paste keys and performs named-vector lookups (`idx_lookup[neighbor_keys]`) inside a row-level `lapply`. Named vector lookup in R is hash-based but still involves repeated character allocation, hashing, and NA-checking for every single row. With ~1.37M neighbor relationships spread across 6.46M rows, this is extremely expensive.

**B. `compute_neighbor_stats` â€” repeated per-variable `lapply` over 6.46M rows**

Called 5 times (once per neighbor source variable). Each call iterates over all 6.46M rows, subsets a numeric vector, removes NAs, and computes `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also a well-known R anti-pattern that creates massive intermediate object copying.

**C. Object copying in the outer loop**

`cell_data <- compute_and_add_neighbor_features(cell_data, ...)` likely triggers full-copy of the data.frame on each column addition (R's copy-on-modify semantics). With ~110+ columns Ã— 6.46M rows, each copy is hundreds of MB. Doing this 15 times (5 vars Ã— 3 stats) compounds the problem.

**D. Random Forest prediction over 6.46M rows**

If `predict()` is called row-by-row or in very small batches, overhead dominates. Even in a single call, `predict.randomForest` in R constructs a full data.frame copy internally and loops through trees in R-level code. With 110 features Ã— 6.46M rows, memory pressure alone can cause swapping on a 16 GB machine.

### Summary of Time Sinks

| Component | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~25-35% | Row-level `lapply`, character key hashing |
| `compute_neighbor_stats` (Ã—5) | ~25-35% | Row-level `lapply`, `do.call(rbind, ...)` |
| Data.frame copying in loop | ~10-15% | Copy-on-modify, repeated column binding |
| RF `predict()` | ~15-25% | Large matrix construction, possible memory swapping |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything; eliminate row-level R loops; use `data.table` for in-place column operations; batch RF prediction.

| Bottleneck | Strategy |
|---|---|
| `build_neighbor_lookup` | Build a `data.table` edge list (cell-year â†’ neighbor-cell-year) with integer joins. No character keys. |
| `compute_neighbor_stats` | One vectorized `data.table` grouped aggregation per variable (or all at once), using the edge list. |
| Column addition / copying | Use `data.table` `:=` for in-place column creation â€” zero copies. |
| RF prediction | Convert features to a matrix once; predict in chunks (~500K rows) to control memory; use `ranger` re-import if possible for 10-50Ã— faster predict. |

**Expected speedup:** From 86+ hours to roughly **10â€“40 minutes** depending on RF library.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================
# Prerequisites:
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec,
#                pop_density, def, usd_est_n2, ... (110 predictor cols)
#   - id_order: vector of cell IDs in the order matching rook_neighbors_unique
#   - rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
#   - rf_model: trained randomForest (or ranger) model object
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table in place ---------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---- Step 1: Build vectorized edge list (replaces build_neighbor_lookup) ---
build_edge_list <- function(id_order, neighbors) {
  # neighbors is an nb object: list of integer index vectors
  # id_order[i] is the cell id for the i-th element of neighbors
  n <- length(neighbors)

  # Pre-compute total edges for pre-allocation
  n_edges <- sum(lengths(neighbors))

  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len <- length(nb_i)
    from_idx[pos:(pos + len - 1L)] <- i
    to_idx[pos:(pos + len - 1L)]   <- nb_i
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-sentinel in spdep)
  actual <- pos - 1L
  data.table(
    from_cell_id = id_order[from_idx[1:actual]],
    to_cell_id   = id_order[to_idx[1:actual]]
  )
}

cat("Building edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)

# ---- Step 2: Compute all neighbor features vectorized ----------------------
compute_all_neighbor_features <- function(dt, edge_dt, source_vars) {
  # dt must have columns: id, year, and all source_vars
  # edge_dt has columns: from_cell_id, to_cell_id

  # Create a row key for fast joining
  dt[, row_idx := .I]

  # Lookup table: (id, year) -> row_idx
  key_dt <- dt[, .(id, year, row_idx)]
  setkey(key_dt, id, year)

  # Get unique years
  years <- sort(unique(dt$year))

  # For each year, expand edges to (from_row, to_row) pairs
  # This avoids character key construction entirely.
  cat("Expanding edge list across years...\n")

  # Cross join edges with years
  edge_year <- CJ_edge_year <- edge_dt[, .(from_cell_id, to_cell_id)]
  # Replicate edges for each year efficiently:
  # Each edge applies to every year in the panel.
  edge_year <- edge_dt[, .(year = years), by = .(from_cell_id, to_cell_id)]

  # Join to get the "from" row index (the focal cell)
  setnames(edge_year, "from_cell_id", "id")
  setkey(edge_year, id, year)
  edge_year <- key_dt[edge_year, on = .(id, year), nomatch = 0L]
  setnames(edge_year, c("row_idx", "id"), c("from_row", "from_id"))

  # Join to get the "to" row index (the neighbor cell)
  setnames(edge_year, "to_cell_id", "id")
  setkey(edge_year, id, year)
  edge_year <- key_dt[edge_year, on = .(id, year), nomatch = 0L]
  setnames(edge_year, c("row_idx", "id"), c("to_row", "to_id"))

  cat("  Edge-year pairs: ", nrow(edge_year), "\n")

  # Now compute grouped stats: for each (from_row), aggregate neighbor values
  for (vname in source_vars) {
    cat("  Computing neighbor stats for:", vname, "\n")

    # Extract neighbor values via to_row
    edge_year[, nval := dt[[vname]][to_row]]

    # Grouped aggregation â€” fully vectorized
    agg <- edge_year[!is.na(nval),
      .(
        nb_max  = max(nval),
        nb_min  = min(nval),
        nb_mean = mean(nval)
      ),
      by = from_row
    ]

    # Initialize new columns with NA
    max_col  <- paste0("neighbor_max_", vname)
    min_col  <- paste0("neighbor_min_", vname)
    mean_col <- paste0("neighbor_mean_", vname)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign in place â€” no copy
    set(dt, i = agg$from_row, j = max_col,  value = agg$nb_max)
    set(dt, i = agg$from_row, j = min_col,  value = agg$nb_min)
    set(dt, i = agg$from_row, j = mean_col, value = agg$nb_mean)
  }

  # Clean up helper column
  edge_year[, nval := NULL]
  dt[, row_idx := NULL]

  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# ---- Step 3: Optimized Random Forest Prediction ---------------------------

# Option A: If the model is a `ranger` object (fastest path)
# Option B: If the model is a `randomForest` object (still optimized)

predict_rf_chunked <- function(model, dt, feature_cols, chunk_size = 500000L) {
  # Pre-build the feature matrix ONCE (avoids repeated data.frame copies
  # inside predict.randomForest)
  cat("Building prediction matrix...\n")
  pred_mat <- as.matrix(dt[, ..feature_cols])

  n <- nrow(pred_mat)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)

  is_ranger <- inherits(model, "ranger")

  cat("Predicting in", n_chunks, "chunks...\n")
  for (ch in seq_len(n_chunks)) {
    start_i <- (ch - 1L) * chunk_size + 1L
    end_i   <- min(ch * chunk_size, n)
    chunk_data <- pred_mat[start_i:end_i, , drop = FALSE]

    if (is_ranger) {
      # ranger::predict is much faster (C++ backend, no data.frame overhead)
      preds <- predict(model, data = chunk_data)$predictions
    } else {
      # randomForest::predict â€” pass matrix to avoid internal as.data.frame
      preds <- predict(model, newdata = chunk_data)
    }

    predictions[start_i:end_i] <- preds

    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %d-%d)\n", ch, n_chunks, start_i, end_i))
    }
    # Explicit gc every N chunks to stay within 16 GB
    if (ch %% 10 == 0) gc(verbose = FALSE)
  }

  predictions
}

# Get feature column names (exclude id, year, and the response variable)
# Adjust 'response_var' to your actual target column name.
response_var <- "gdp"  # <-- adjust if needed
meta_cols    <- c("id", "year", response_var)
feature_cols <- setdiff(names(cell_data), meta_cols)

cat("Generating predictions...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model        = rf_model,
    dt           = cell_data,
    feature_cols = feature_cols,
    chunk_size   = 500000L
  )]
})

cat("Done.\n")
```

---

## 4. OPTIONAL BUT RECOMMENDED: Re-wrap Existing `randomForest` Model as `ranger` for Prediction

If the trained model is a `randomForest` object, you can extract it and re-import into `ranger` format for **10â€“50Ã— faster** `predict()` without retraining. This preserves the original estimand because the tree structures and splits are identical:

```r
# Convert randomForest â†’ ranger (prediction-only, no retraining)
# Requires: ranger >= 0.14.0
# This reads the tree structure from the randomForest object.

library(ranger)

convert_rf_to_ranger <- function(rf_model) {
  # ranger can read randomForest tree structures directly
  # via ranger::treeInfo applied in reverse â€” but the simplest
  # production-safe path is to save predictions from both and verify.
  #
  # If direct conversion is not available in your ranger version,
  # keep using randomForest::predict with the matrix-input chunked
  # approach above, which already eliminates the largest overhead.
  message("Note: If ranger conversion is unavailable, the chunked ",
          "matrix-input approach already provides major speedups.")
}
```

---

## 5. EXPECTED PERFORMANCE

| Stage | Original (est.) | Optimized (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~15â€“25 hrs | **~30 sec** (vectorized edge list + keyed join) |
| `compute_neighbor_stats` (Ã—5) | ~25â€“35 hrs | **~2â€“5 min** (grouped `data.table` aggregation) |
| Data.frame column binding | ~10 hrs | **~0 sec** (`:=` in place) |
| RF `predict()` (6.46M rows) | ~20+ hrs | **~5â€“30 min** (`matrix` input, chunked, or `ranger`) |
| **Total** | **86+ hrs** | **~10â€“40 min** |

### Key Invariants Preserved
- âœ… **Trained RF model is never retrained** â€” only used for `predict()`.
- âœ… **Original numerical estimand is preserved** â€” identical tree splits produce identical predictions; only the data-handling wrapper changes.
- âœ… **All 15 neighbor features** (5 vars Ã— max/min/mean) are computed identically via the same mathematical operations, just vectorized.
END_RESPONSE>>>
