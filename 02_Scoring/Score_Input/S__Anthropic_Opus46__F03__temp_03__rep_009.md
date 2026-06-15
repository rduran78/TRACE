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
run_id: Anthropic_F03_T03_R009
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 9
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_009.md

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

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix â€” this is slow because `rbind` on a list of millions of small vectors has quadratic-like overhead. This is called **5 times** (once per neighbor source variable).

**`compute_and_add_neighbor_features` outer loop:** Likely copies the entire `cell_data` data.frame on each assignment (`cell_data <- ...`), which for 6.46M Ã— 110+ columns is a multi-GB copy â€” **5 times**.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, calling `predict()` on a Random Forest in one shot will:
- Require the entire feature matrix in memory simultaneously alongside the model object (potentially 10+ GB).
- On a 16 GB laptop, this risks swapping to disk.
- If prediction is done row-by-row or in a naive loop, it's catastrophically slow.

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string pastes + named-vector lookups | ~hours |
| `compute_neighbor_stats` | 6.46M `lapply` + `do.call(rbind, ...)` Ã—5 | ~hours |
| Data.frame copy-on-modify | `cell_data <-` in loop copies entire frame Ã—5 | ~tens of minutes |
| RF prediction | Possible memory pressure / naive chunking | ~hours if swapping |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorized with `data.table`

1. **Replace `build_neighbor_lookup`** with a `data.table` join-based approach. Instead of building a per-row list of neighbor indices, build a long-format **edge table** (`id`, `year`, `neighbor_id`) and join it to the data to get neighbor values directly. This eliminates all per-row string operations.

2. **Replace `compute_neighbor_stats`** with a grouped `data.table` aggregation on the edge table: `dt_edges[dt_values, on=...][, .(max, min, mean), by=.(id, year)]`. This is fully vectorized in C.

3. **Use `data.table` set-by-reference** (`:=`) to add columns, eliminating all data.frame copies.

### 2.2 Prediction â€” Batched with Memory Control

1. **Predict in chunks** (e.g., 500K rows) to keep peak memory well under 16 GB.
2. **Load the model once**, reuse for all chunks.
3. Use `data.table` to hold results and bind via `rbindlist`.

### 2.3 Expected Speedup

| Component | Before | After (est.) |
|---|---|---|
| Neighbor lookup build | ~hours | ~1â€“3 min |
| Neighbor stats (Ã—5 vars) | ~hours | ~2â€“5 min |
| Data copies | ~30+ min | ~0 (in-place) |
| RF prediction (6.46M rows) | variable | ~10â€“30 min |
| **Total** | **86+ hours** | **~15â€“45 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest), spdep (for nb object)
# =============================================================================

library(data.table)

# ---- 3.1 BUILD VECTORIZED NEIGHBOR EDGE TABLE ----

#' Converts an spdep::nb neighbor list + id_order into a long-format edge
#' data.table with columns: id, neighbor_id.
#'
#' @param id_order Integer/numeric vector of cell IDs in the order matching
#'   the nb object (length = number of spatial cells, e.g. 344,208).
#' @param neighbors An spdep::nb object (list of integer index vectors).
#' @return data.table with columns `id` (integer) and `neighbor_id` (integer).

build_neighbor_edges <- function(id_order, neighbors) {
  # Number of neighbors per cell
  n_neighbors <- vapply(neighbors, length, integer(1))

  # Pre-allocate vectors
  total_edges <- sum(n_neighbors)
  from_id     <- integer(total_edges)
  to_id       <- integer(total_edges)

  pos <- 1L
  for (i in seq_along(neighbors)) {
    nn <- n_neighbors[i]
    if (nn > 0L) {
      idx_range <- pos:(pos + nn - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[neighbors[[i]]]
      pos <- pos + nn
    }
  }

  data.table(id = from_id, neighbor_id = to_id)
}


# ---- 3.2 COMPUTE NEIGHBOR FEATURES VIA GROUPED JOIN ----

#' For a given variable, compute max/min/mean of neighbor values for every
#' (id, year) combination and add them as columns to dt in place.
#'
#' @param dt data.table with at least columns: id, year, and `var_name`.
#' @param var_name Character string â€” name of the variable.
#' @param edges data.table with columns: id, neighbor_id (the edge table).
#' @return Invisible NULL. Columns are added to `dt` by reference.

compute_and_add_neighbor_features_dt <- function(dt, var_name, edges) {
  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Subset: only the columns we need from dt for the join
  # Key: neighbor_id + year  â†’  we look up the neighbor's value
  vals <- dt[, .(neighbor_id = id, year, val = get(var_name))]
  setkey(vals, neighbor_id, year)

  # Join edges to dt to get (id, year, neighbor_id), then join to vals
  # Step 1: cross edges with years present in dt for each id
  #   â€” but more efficient: join dt's (id, year) to edges to get
  #     (id, year, neighbor_id), then join neighbor_id+year â†’ val.

  # dt_iy: every (id, year) row index
  dt_iy <- dt[, .(id, year, .row_idx = .I)]
  setkey(edges, id)

  # Expand: for each row in dt, get its neighbors
  # This produces a long table: (id, year, neighbor_id, .row_idx)
  expanded <- edges[dt_iy, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: id, neighbor_id, year, .row_idx

  # Join to get neighbor values
  expanded[vals, val := i.val, on = .(neighbor_id, year)]

  # Aggregate by the original row
  agg <- expanded[!is.na(val),
    .(nmax = max(val), nmin = min(val), nmean = mean(val)),
    by = .row_idx
  ]

  # Initialize columns with NA
  set(dt, j = col_max,  value = NA_real_)
  set(dt, j = col_min,  value = NA_real_)
  set(dt, j = col_mean, value = NA_real_)

  # Fill in computed values by row index
  set(dt, i = agg$.row_idx, j = col_max,  value = agg$nmax)
  set(dt, i = agg$.row_idx, j = col_min,  value = agg$nmin)
  set(dt, i = agg$.row_idx, j = col_mean, value = agg$nmean)

  invisible(NULL)
}


# ---- 3.3 BATCHED RANDOM FOREST PREDICTION ----

#' Predict in chunks to control peak memory on a 16 GB machine.
#'
#' @param model A trained Random Forest model (ranger or randomForest object).
#' @param dt data.table containing all predictor columns.
#' @param feature_cols Character vector of predictor column names.
#' @param batch_size Integer, number of rows per prediction batch.
#' @return Numeric vector of predictions (length = nrow(dt)).

predict_rf_batched <- function(model, dt, feature_cols, batch_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)

  starts <- seq(1L, n, by = batch_size)

  for (k in seq_along(starts)) {
    i_start <- starts[k]
    i_end   <- min(i_start + batch_size - 1L, n)
    idx     <- i_start:i_end

    batch <- dt[idx, ..feature_cols]

    # Support both ranger and randomForest objects
    if (inherits(model, "ranger")) {
      preds[idx] <- predict(model, data = batch)$predictions
    } else {
      # randomForest
      preds[idx] <- predict(model, newdata = batch)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("  Predicted batch %d/%d (rows %dâ€“%d)",
                       k, length(starts), i_start, i_end))
    }
  }

  preds
}


# ---- 3.4 MAIN PIPELINE ----

run_optimized_pipeline <- function(
  cell_data,          # data.frame or data.table with columns: id, year, + predictors
  id_order,           # integer vector of cell IDs matching nb object order
  rook_neighbors_unique,  # spdep::nb object
  rf_model,           # trained Random Forest model (not retrained)
  feature_cols,       # character vector of all ~110 predictor column names
  neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
  prediction_batch_size = 500000L
) {

  # --- Convert to data.table (by reference if already one) ---
  if (!is.data.table(cell_data)) {
    message("Converting cell_data to data.table...")
    cell_data <- as.data.table(cell_data)
  }
  setkey(cell_data, id, year)

  message(sprintf("Data: %s rows x %s cols", format(nrow(cell_data), big.mark=","),
                  ncol(cell_data)))

  # --- Step 1: Build edge table (once) ---
  message("Building neighbor edge table...")
  t0 <- proc.time()
  edges <- build_neighbor_edges(id_order, rook_neighbors_unique)
  setkey(edges, id)
  message(sprintf("  Edge table: %s edges  [%.1f sec]",
                  format(nrow(edges), big.mark=","),
                  (proc.time() - t0)[3]))

  # --- Step 2: Compute neighbor features (vectorized, in-place) ---
  message("Computing neighbor features...")
  for (var_name in neighbor_source_vars) {
    t1 <- proc.time()
    compute_and_add_neighbor_features_dt(cell_data, var_name, edges)
    elapsed <- (proc.time() - t1)[3]
    message(sprintf("  %s: done [%.1f sec]", var_name, elapsed))
  }

  # --- Step 3: Garbage collect before prediction ---
  rm(edges)
  gc(verbose = FALSE)

  # --- Step 4: Batched prediction ---
  message("Running batched Random Forest prediction...")
  t2 <- proc.time()

  # Verify all feature columns exist
  missing_cols <- setdiff(feature_cols, names(cell_data))
  if (length(missing_cols) > 0) {
    stop("Missing predictor columns in cell_data: ",
         paste(missing_cols, collapse = ", "))
  }

  cell_data[, predicted_gdp := predict_rf_batched(
    model        = rf_model,
    dt           = cell_data,
    feature_cols = feature_cols,
    batch_size   = prediction_batch_size
  )]

  elapsed_pred <- (proc.time() - t2)[3]
  message(sprintf("  Prediction complete [%.1f sec]", elapsed_pred))

  # --- Return result ---
  cell_data
}


# =============================================================================
# USAGE EXAMPLE (uncomment and adapt to your file paths / object names)
# =============================================================================
#
# library(data.table)
# library(ranger)  # or library(randomForest)
#
# # Load pre-trained model
# rf_model <- readRDS("trained_rf_model.rds")
#
# # Load data
# cell_data <- fread("cell_panel_data.csv")   # or readRDS(...)
# id_order  <- readRDS("id_order.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors.rds")
#
# # Define your ~110 feature column names
# feature_cols <- readRDS("feature_col_names.rds")
#   # OR: feature_cols <- setdiff(names(cell_data), c("id", "year", "gdp", ...))
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   feature_cols           = feature_cols,
#   neighbor_source_vars   = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   prediction_batch_size  = 500000L
# )
#
# # result is a data.table with a new column `predicted_gdp`
# fwrite(result, "cell_gdp_predictions.csv")
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Requirement | How it's met |
|---|---|
| **Trained RF model preserved** | The model is loaded once via `readRDS` and passed read-only to `predict()`. No retraining occurs. |
| **Original numerical estimand preserved** | The neighbor features are computed with identical semantics (max, min, mean of non-NA neighbor values per cell-year). The same predictor columns are passed to the same model, producing identical predictions. `data.table` uses the same R numeric types â€” no floating-point reinterpretation. |
| **Same neighbor logic** | `build_neighbor_edges` faithfully translates the `spdep::nb` index list using the same `id_order` mapping. The join-based aggregation reproduces the original `vals[idx]` subsetting and `max/min/mean` computation. |

---

## 5. KEY PERFORMANCE GAINS SUMMARIZED

1. **String elimination**: No `paste(id, year)` keys. Joins are on integer `(id, year)` pairs via `data.table` indexed merge â€” orders of magnitude faster.
2. **Vectorized aggregation**: `data.table`'s grouped `max/min/mean` runs in optimized C, replacing 6.46M R-level `lapply` iterations Ã— 5 variables.
3. **Zero-copy column addition**: `set()` and `:=` modify the table in place â€” no 5Ã— multi-GB data.frame copies.
4. **Batched prediction**: 500K-row chunks keep peak memory ~4â€“6 GB, well within 16 GB, preventing OS swap thrashing.
5. **Single edge-table build**: The neighbor structure is materialized once as a keyed `data.table` and reused for all 5 variables, replacing the expensive per-row list construction.
END_RESPONSE>>>
