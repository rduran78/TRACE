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
run_id: Anthropic_F03_T10_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy & Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup` is catastrophically slow at scale.**

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - `as.character()` conversion and named-vector lookup (`id_to_ref`),
  - subsetting a list (`neighbors[[ref_idx]]`),
  - `paste()` to build composite keys,
  - named-vector lookup on `idx_lookup` (a 6.46M-element named character vector â€” every lookup is **O(n) hash probe on a huge vector**),
  - `is.na` filtering.
- Named-vector lookups in R are backed by hashing, but constructing and probing a **6.46 million-entry** names table millions of times is extremely expensive. Each call to `idx_lookup[neighbor_keys]` must hash every key and scan buckets.
- The entire output is a **list of 6.46 million integer vectors** â€” heavy memory allocation and GC pressure.

**`compute_neighbor_stats` is moderately slow.**

- Another `lapply` over 6.46M elements, each extracting a subset of a numeric vector, removing NAs, and computing `max/min/mean`. The subsetting and NA-removal are fine for small neighbor sets (~4 rook neighbors), but doing this 6.46M Ã— 5 variables = **32.3 million** iterations adds up.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself a significant bottleneck; it must allocate and copy a huge matrix incrementally.

**The outer loop repeats `compute_neighbor_stats` 5 times**, each time producing 3 new columns (max, min, mean) â€” 15 new columns total. Each pass scans all 6.46M rows.

### 1.2 Random Forest Inference Bottleneck

- `predict.randomForest` (or `predict.ranger`) on **6.46 million rows Ã— 110 features** is inherently expensive.
- If using the `randomForest` package, `predict()` is implemented in R-level loops over trees â€” it is **extremely slow** at this scale. A 500-tree forest on 6.46M rows can take hours.
- If the model is a `randomForest` object, every call to `predict()` also **copies the entire data.frame** internally (R's copy-on-modify semantics).
- Predicting in one monolithic call forces the entire prediction matrix (~6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.3 GB for doubles) into RAM simultaneously, likely exceeding 16 GB when combined with tree structures and intermediate allocations.

### 1.3 Memory Pressure

| Component | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 125 cols) | ~6.1 GB |
| `neighbor_lookup` (6.46M lists) | ~1â€“2 GB |
| RF model (500 trees) | ~0.5â€“2 GB |
| Prediction temporaries | ~5+ GB |
| **Total** | **>14 GB â†’ thrashing** |

On a 16 GB laptop this will cause heavy swapping, which alone can explain the 86-hour estimate.

---

## 2. OPTIMIZATION STRATEGY

### Layer 1: Replace `build_neighbor_lookup` with vectorized `data.table` joins

Instead of 6.46M R-level iterations with named-vector lookups, we:
1. Expand the `nb` object into a two-column edge table (`from_id`, `to_id`) â€” done once.
2. Join with `cell_data` on `(to_id, year)` to get the row indices of every neighbor for every row â€” one `data.table` merge.
3. Compute all neighbor stats (max, min, mean) for **all 5 variables simultaneously** using `data.table` grouped aggregation â€” replacing the 5-pass `lapply` loop.

This collapses the entire feature-preparation stage into **two vectorized operations** (one join, one grouped aggregation).

### Layer 2: Convert the RF model from `randomForest` â†’ `ranger` for prediction

`ranger::ranger` can consume a `randomForest` object or we can re-wrap it. But since we **cannot retrain**, we use the approach:
- If the model is class `randomForest`: convert prediction to chunked calls to `predict()`, **or** extract the forest and use the C-level `ranger` prediction engine via `ranger::predict.ranger` on a "transplanted" object. The simplest robust approach: **chunk the prediction** into batches of ~500K rows and use `predict()` on each chunk, freeing memory between chunks.
- If the model is class `ranger`: simply use `predict()` in chunks.

Either way, chunked prediction keeps peak memory manageable.

### Layer 3: Memory management

- Use `data.table` in-place column assignment (`:=`) instead of `cbind` / column-appending copies.
- Remove intermediate objects and call `gc()` between stages.
- Feed `predict()` only the required columns (110 predictors), not the full table.

### Expected speedup

| Stage | Before | After | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~30â€“50 hrs | ~2â€“5 min | ~500Ã— |
| `compute_neighbor_stats` (Ã—5) | ~20â€“30 hrs | ~1â€“3 min (combined) | ~500Ã— |
| RF prediction (6.46M rows) | ~5â€“10 hrs | ~10â€“40 min (chunked) | ~10â€“20Ã— |
| **Total** | **86+ hrs** | **~15â€“50 min** | **~100â€“300Ã—** |

---

## 3. WORKING R CODE

```r
# ==============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (optional), randomForest
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# STEP 0: Prepare the nb edge list (run ONCE, can be cached to disk)
# --------------------------------------------------------------------------
#' Convert an spdep::nb object + id_order vector into a data.table edge list.
#' @param id_order integer/character vector: the cell IDs in the same order as
#'   the nb object (i.e., id_order[i] is the cell ID for neighbors[[i]]).
#' @param neighbors an spdep::nb object (list of integer index vectors).
#' @return data.table with columns (from_id, to_id)
build_edge_table <- function(id_order, neighbors) {
  # Pre-allocate vectors for speed
  n <- length(neighbors)
  from_list <- vector("list", n)
  to_list   <- vector("list", n)

  for (i in seq_len(n)) {
    nb_idx <- neighbors[[i]]
    # spdep::nb stores 0L for no-neighbor entries in some cases
    nb_idx <- nb_idx[nb_idx > 0L]
    if (length(nb_idx) > 0L) {
      from_list[[i]] <- rep(id_order[i], length(nb_idx))
      to_list[[i]]   <- id_order[nb_idx]
    }
  }

  data.table(
    from_id = unlist(from_list, use.names = FALSE),
    to_id   = unlist(to_list,   use.names = FALSE)
  )
}

# --------------------------------------------------------------------------
# STEP 1: Build neighbor features via vectorized data.table joins
# --------------------------------------------------------------------------
#' Compute neighbor-aggregated features for multiple variables at once.
#'
#' @param cell_dt   data.table with at least columns: id, year, and all
#'                  columns named in `var_names`.
#' @param edge_dt   data.table with columns (from_id, to_id) â€” the directed
#'                  neighbor edge list produced by build_edge_table().
#' @param var_names character vector of variable names to aggregate.
#' @return cell_dt, modified in place with new columns:
#'         {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
#'         for each var in var_names.
compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {

  # -- 1. Ensure data.table and create a row-key ----------------------------
  if (!is.data.table(cell_dt)) setDT(cell_dt)
  if (!is.data.table(edge_dt)) setDT(edge_dt)

  # Keyed lookup table: (id, year) -> values of interest
  # We only need id, year, and the source variables for the join.
  cols_needed <- c("id", "year", var_names)
  lookup <- cell_dt[, ..cols_needed]
  setnames(lookup, "id", "to_id")              # rename for join
  setkey(lookup, to_id, year)

  # -- 2. Expand edges Ã— years: join edge_dt with cell_dt to get the year
  #       for each "from" row, then join with lookup to get neighbor values. --


  # We need each (from_id, year) â†’ all its neighbors' values.
  # Strategy:
  #   a) Get (row_index, from_id, year) from cell_dt.
  #   b) Join with edge_dt on from_id â†’ gives (row_index, to_id, year).
  #   c) Join with lookup on (to_id, year) â†’ gives neighbor variable values.
  #   d) Aggregate by row_index â†’ max, min, mean per variable.

  # (a) Mapping from cell_dt rows to (from_id, year)
  cell_dt[, .row_idx := .I]
  from_keys <- cell_dt[, .(`.row_idx` = .row_idx, from_id = id, year = year)]
  setkey(from_keys, from_id)

  # (b) Join with edge list
  setkey(edge_dt, from_id)
  expanded <- edge_dt[from_keys, on = "from_id",
                      allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded has columns: from_id, to_id, .row_idx, year

  # (c) Join with lookup to get neighbor variable values
  setkey(expanded, to_id, year)
  expanded <- lookup[expanded, on = c("to_id", "year"), nomatch = NA]
  # Now expanded has: to_id, year, <var_names>, from_id, .row_idx

  # (d) Aggregate by .row_idx
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0(v, c("_neighbor_max", "_neighbor_min", "_neighbor_mean"))
  }))

  names(agg_exprs) <- agg_names

  cat("Aggregating neighbor stats for", length(var_names), "variables...\n")
  agg <- expanded[, eval(as.call(c(as.name("list"),
                                    lapply(agg_exprs, function(e) e)))),
                  by = .(.row_idx)]

  # Replace infinite values (from max/min of empty after NA removal) with NA
  for (col in agg_names) {
    set(agg, which(is.infinite(agg[[col]])), col, NA_real_)
  }

  # -- 3. Merge back into cell_dt by row index ------------------------------
  setkey(agg, .row_idx)
  for (col in agg_names) {
    set(cell_dt, j = col, value = NA_real_)
    set(cell_dt, i = agg[[".row_idx"]], j = col, value = agg[[col]])
  }

  # Clean up helper column
  cell_dt[, .row_idx := NULL]

  invisible(cell_dt)
}

# --------------------------------------------------------------------------
# STEP 2: Chunked Random Forest Prediction
# --------------------------------------------------------------------------
#' Predict in chunks to avoid memory blowup.
#'
#' @param model       A trained randomForest or ranger model object.
#' @param newdata_dt  data.table of prediction data (all columns needed by model).
#' @param pred_vars   character vector of the predictor column names (in model order).
#' @param chunk_size  number of rows per prediction batch.
#' @return numeric vector of predictions (same length as nrow(newdata_dt)).
predict_chunked <- function(model, newdata_dt, pred_vars, chunk_size = 500000L) {
  n <- nrow(newdata_dt)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  model_class <- class(model)[1]
  cat("Running chunked prediction (", model_class, ") on", n,
      "rows in", length(starts), "chunks ...\n")

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + chunk_size - 1L, n)
    chunk <- as.data.frame(newdata_dt[i1:i2, ..pred_vars])

    if (model_class == "ranger") {
      preds[i1:i2] <- predict(model, data = chunk)$predictions
    } else {
      # randomForest or other
      preds[i1:i2] <- predict(model, newdata = chunk)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  chunk %d/%d done (rows %dâ€“%d)\n", k, length(starts), i1, i2))
    }

    # Free chunk memory
    rm(chunk)
    if (k %% 10 == 0) gc(verbose = FALSE)
  }

  return(preds)
}

# ==========================================================================
# MAIN PIPELINE
# ==========================================================================

run_optimized_pipeline <- function(cell_data,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model,
                                   predictor_names,
                                   chunk_size = 500000L) {

  cat("=== Optimized GDP Prediction Pipeline ===\n")
  t0 <- proc.time()

  # --- Convert to data.table (in place if already) -------------------------
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
    cat("Converted cell_data to data.table.\n")
  }

  # --- Step 0: Build edge table from nb object (fast, one-time) ------------
  cat("Building edge table from nb object ...\n")
  edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
  cat(sprintf("  Edge table: %d directed edges.\n", nrow(edge_dt)))

  # --- Step 1: Compute all neighbor features at once -----------------------
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  cat("Computing neighbor features for:", paste(neighbor_source_vars, collapse = ", "), "\n")

  t1 <- proc.time()
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
  cat(sprintf("  Neighbor features computed in %.1f seconds.\n",
              (proc.time() - t1)[3]))

  # Free edge table memory

  rm(edge_dt)
  gc(verbose = FALSE)

  # --- Step 2: Chunked RF Prediction ---------------------------------------
  cat("Starting Random Forest prediction ...\n")
  t2 <- proc.time()

  cell_data[, predicted_gdp := predict_chunked(
    model      = rf_model,
    newdata_dt = cell_data,
    pred_vars  = predictor_names,
    chunk_size = chunk_size
  )]

  cat(sprintf("  Prediction completed in %.1f seconds.\n",
              (proc.time() - t2)[3]))

  total_time <- (proc.time() - t0)[3]
  cat(sprintf("=== Pipeline finished in %.1f seconds (%.1f minutes). ===\n",
              total_time, total_time / 60))

  return(cell_data)
}

# ==========================================================================
# USAGE EXAMPLE (adapt paths/names to your environment)
# ==========================================================================
#
# library(data.table)
# library(randomForest)   # or library(ranger)
#
# # Load pre-trained model (do NOT retrain)
# rf_model <- readRDS("path/to/trained_rf_model.rds")
#
# # Load data
# cell_data            <- fread("path/to/cell_data.csv")
# id_order             <- readRDS("path/to/id_order.rds")
# rook_neighbors_unique <- readRDS("path/to/rook_neighbors_unique.rds")
#
# # The predictor column names the model expects (110 variables)
# predictor_names <- readRDS("path/to/predictor_names.rds")
# # OR: predictor_names <- colnames(rf_model$forest$xlevels)  # for randomForest
# # OR: predictor_names <- rf_model$forest$independent.variable.names  # for ranger
#
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   predictor_names        = predictor_names,
#   chunk_size             = 500000L
# )
#
# # result now has a `predicted_gdp` column
```

---

## 4. SUMMARY OF CHANGES & WHY THEY WORK

| Original Problem | Root Cause | Fix |
|---|---|---|
| `build_neighbor_lookup`: 6.46M R-level iterations with `paste()`+named-vector lookup on a 6.46M-entry table | R named vector lookups are slow hash probes; `paste` allocates millions of strings | Replace with a **single `data.table` keyed join** â€” C-level radix join, O(n log n) |
| `compute_neighbor_stats`: 5 separate `lapply` passes Ã— 6.46M iterations, each ending with `do.call(rbind, ...)` | R-level loop; `do.call(rbind, 6.46M-element list)` is quadratic-ish in memory copies | **One grouped `data.table` aggregation** computes all 15 stats (5 vars Ã— 3 stats) in a single pass â€” parallelized C internals |
| RF `predict()` on 6.46M rows at once | Entire feature matrix (~5+ GB) + tree traversal intermediates exceed 16 GB RAM â†’ swap thrashing | **Chunked prediction** (500K rows/chunk) keeps peak memory â‰¤ ~2 GB for prediction workspace |
| Repeated `cell_data <- cbind(cell_data, ...)` | R copies the entire data.frame on each column addition (copy-on-modify) | `data.table::set()` modifies **in place** â€” zero copies |
| Using `randomForest::predict()` | R-level tree traversal is ~10Ã— slower than C-level `ranger` | If feasible, convert model to `ranger` format; otherwise chunking still brings major improvement |

**The trained Random Forest model is never modified or retrained.** The numerical predictions (the estimand) are preserved exactly â€” only the infrastructure around data preparation and prediction dispatch is optimized.
END_RESPONSE>>>
