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
run_id: Anthropic_F03_T10_R002
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 2
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_002.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup` is the single largest bottleneck in pre-prediction code.**

| Problem | Explanation |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and `NA` filtering â€” all inside an interpreted R loop. At ~6.46M iterations this is catastrophically slow. |
| **Repeated `paste` key construction** | `paste(neighbor_cell_ids, data$year[i], sep="_")` builds throwaway character vectors 6.46M times. String operations in R are slow and allocation-heavy. |
| **Named-vector lookup (`idx_lookup[neighbor_keys]`)** | Named-vector subsetting is O(n) per call in R's internal hash; with ~4 neighbors per cell this is ~26M hash lookups done one-at-a-time inside `lapply`. |
| **Character coercion inside the loop** | `as.character(data$id[i])` is called per row instead of being vectorized once. |

**`compute_neighbor_stats` is the secondary bottleneck.**

| Problem | Explanation |
|---|---|
| **Per-row `lapply` with subsetting** | 6.46M iterations, each subsetting a numeric vector and computing `max/min/mean`. |
| **`do.call(rbind, result)` on a 6.46M-element list** | Binds 6.46M 3-element vectors into a matrix. This is slow and memory-intensive. |
| **Called 5 times (once per variable)** | Total: ~32.3M R-level loop iterations just for neighbor stats. |

### B. Random Forest Inference Bottleneck

| Problem | Explanation |
|---|---|
| **Single monolithic `predict()` call on 6.46M rows Ã— 110 features** | `ranger`/`randomForest` `predict()` must allocate a full prediction matrix. With 110 features and 6.46M rows, this is a large memory footprint (~5â€“7 GB for the feature matrix alone in `randomForest`). On a 16 GB laptop this risks swapping. |
| **`randomForest::predict` copies the data internally** | The `randomForest` package converts the input to a matrix internally, doubling memory use. `ranger` is more efficient. |
| **Model object loading** | If the model is a `randomForest` object with 500 trees and 110 variables, the object itself can be 1â€“4 GB. Loading from disk is slow; keeping it in memory competes with data. |

### C. Memory Pressure

With 6.46M rows Ã— 110 columns (double), the feature data alone is ~5.4 GB. Adding the neighbor lookup list (~26M integers + list overhead â‰ˆ 0.5â€“1 GB), model object, and intermediate copies, the pipeline easily exceeds 16 GB, causing disk swapping and the estimated 86+ hour runtime.

---

## 2. Optimization Strategy

### Principles
1. **Vectorize everything** â€” eliminate all per-row `lapply` calls.
2. **Use `data.table`** â€” fastest in-memory tabular operations in R, minimal copying.
3. **Replace named-vector lookups with integer joins** â€” O(1) amortized.
4. **Batch prediction** â€” predict in chunks to stay within RAM.
5. **Use `ranger` for prediction if possible** â€” if the model is `ranger`, predict is C++-backed and memory-lean. If `randomForest`, convert or chunk aggressively.
6. **Pre-allocate output matrices** â€” avoid `do.call(rbind, ...)` on millions of elements.

### Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  1. Load model once, keep in memory                 â”‚
â”‚  2. Build neighbor edge-list (vectorized, data.table)â”‚
â”‚  3. Compute neighbor stats via data.table joins      â”‚
â”‚     (all 5 variables in one pass)                   â”‚
â”‚  4. Predict in chunks of ~500K rows                 â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest)
# Preserves: trained RF model, original numerical estimand
# =============================================================================

library(data.table)

# ---- Configuration ----------------------------------------------------------
CHUNK_SIZE <- 500000L # rows per prediction chunk (tune to RAM)

# =============================================================================
# STEP 1: VECTORIZED NEIGHBOR EDGE-LIST CONSTRUCTION
# =============================================================================
# Replaces build_neighbor_lookup entirely.
# Produces a data.table with columns: focal_row, neighbor_row
# This is the core speedup: fully vectorized, no per-row lapply.

build_neighbor_edgelist_dt <- function(data_dt, id_order, neighbors) {
  # data_dt must be a data.table with columns: id, year, and a .row_idx column
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  # --- Map cell id -> reference index into id_order --------------------------
  id_to_ref <- data.table(
    cell_id = as.integer(id_order),
    ref_idx = seq_along(id_order)
  )

  # --- Explode the nb object into an edge-list of (ref_idx -> neighbor_ref_idx)
  #     then map to cell IDs. This is done ONCE, independent of year. ----------
  n_neighbors <- vapply(neighbors, length, integer(1))
  edge_dt <- data.table(
    focal_ref   = rep(seq_along(neighbors), times = n_neighbors),
    neighbor_ref = unlist(neighbors, use.names = FALSE)
  )
  # Map ref indices back to cell IDs
  edge_dt[, focal_id    := id_order[focal_ref]]
  edge_dt[, neighbor_id := id_order[neighbor_ref]]
  edge_dt[, c("focal_ref", "neighbor_ref") := NULL]

  # --- Cross-join with years to get (focal_id, year) -> (neighbor_id, year) ---
  #     Instead of a full cross-join (which would be huge), we join through
  #     the data rows. ---------------------------------------------------------

  # Add row index to data
  if (!"row_idx" %in% names(data_dt)) {
    data_dt[, row_idx := .I]
  }

  # Focal side: join data rows to edges on focal_id
  # For each data row (id=focal_id, year=y), find all neighbor cell IDs
  focal_keys <- data_dt[, .(focal_row = row_idx, focal_id = id, year)]
  setkey(focal_keys, focal_id)
  setkey(edge_dt, focal_id)

  # merge: for each focal data row, get all neighbor_ids
  merged <- edge_dt[focal_keys, on = "focal_id", allow.cartesian = TRUE,
                    nomatch = NULL]
  # merged now has: focal_row, focal_id, year, neighbor_id

  # Neighbor side: look up the row index of (neighbor_id, year)
  neighbor_keys <- data_dt[, .(neighbor_id = id, year, neighbor_row = row_idx)]
  setkey(neighbor_keys, neighbor_id, year)
  setkey(merged, neighbor_id, year)

  result <- neighbor_keys[merged, on = c("neighbor_id", "year"),
                          nomatch = NA]
  # Keep only matched rows
  result <- result[!is.na(neighbor_row),
                   .(focal_row, neighbor_row)]

  setkey(result, focal_row)
  return(result)
}


# =============================================================================
# STEP 2: VECTORIZED NEIGHBOR STATISTICS (ALL VARIABLES, ONE PASS)
# =============================================================================
# Replaces compute_neighbor_stats + the outer for-loop over 5 variables.
# Uses data.table grouping instead of per-row lapply.

compute_all_neighbor_stats_dt <- function(data_dt, edge_dt, var_names) {
  # edge_dt: data.table with focal_row, neighbor_row (keyed on focal_row)
  # var_names: character vector of column names to compute stats for
  # Returns: data.table with nrow(data_dt) rows and 3 columns per variable

  n_rows <- nrow(data_dt)

  # Extract neighbor values for all variables at once
  # Build a table: focal_row + all variable values at the neighbor row
  neighbor_vals <- data_dt[edge_dt$neighbor_row, ..var_names]
  neighbor_vals[, focal_row := edge_dt$focal_row]

  # Compute grouped stats
  stat_cols <- character(0)
  expr_list <- list()
  for (v in var_names) {
    col_max  <- paste0("n_max_", v)
    col_min  <- paste0("n_min_", v)
    col_mean <- paste0("n_mean_", v)
    stat_cols <- c(stat_cols, col_max, col_min, col_mean)
    expr_list[[col_max]]  <- substitute(max(x, na.rm = TRUE), list(x = as.name(v)))
    expr_list[[col_min]]  <- substitute(min(x, na.rm = TRUE), list(x = as.name(v)))
    expr_list[[col_mean]] <- substitute(mean(x, na.rm = TRUE), list(x = as.name(v)))
  }

  agg <- neighbor_vals[,
    lapply(expr_list, eval, envir = .SD),
    by = focal_row
  ]

  # Ensure all rows are represented (some may have no neighbors)
  all_rows <- data.table(focal_row = seq_len(n_rows))
  agg <- agg[all_rows, on = "focal_row"]

  # Replace -Inf/Inf from max/min of empty sets with NA
  for (col in stat_cols) {
    agg[is.infinite(get(col)), (col) := NA_real_]
  }

  # Order by focal_row and drop the key column
  setorder(agg, focal_row)
  agg[, focal_row := NULL]

  return(agg)
}


# =============================================================================
# STEP 3: CHUNKED RANDOM FOREST PREDICTION
# =============================================================================
# Predicts in chunks to stay within 16 GB RAM.
# Works with both ranger and randomForest model objects.

predict_chunked <- function(model, data_dt, feature_cols, chunk_size = CHUNK_SIZE) {
  n <- nrow(data_dt)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)

  is_ranger <- inherits(model, "ranger")

  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)

    chunk <- data_dt[start_idx:end_idx, ..feature_cols]

    if (is_ranger) {
      pred <- predict(model, data = chunk)$predictions
    } else {
      # randomForest
      pred <- predict(model, newdata = chunk)
    }

    predictions[start_idx:end_idx] <- pred

    # Free chunk memory
    rm(chunk, pred)
    if (i %% 5 == 0) gc(verbose = FALSE)
  }

  return(predictions)
}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_optimized_pipeline <- function(cell_data,
                                   id_order,
                                   rook_neighbors_unique,
                                   rf_model,
                                   feature_cols,
                                   neighbor_source_vars = c("ntl", "ec",
                                     "pop_density", "def", "usd_est_n2"),
                                   chunk_size = CHUNK_SIZE) {

  cat("Converting to data.table...\n")
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, row_idx := .I]

  # ---- Step 1: Build vectorized neighbor edge-list -------------------------
  cat("Building neighbor edge-list (vectorized)...\n")
  t0 <- proc.time()

  edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order,
                                         rook_neighbors_unique)

  cat(sprintf("  Edge-list built: %d edges in %.1f seconds\n",
              nrow(edge_dt), (proc.time() - t0)[3]))

  # ---- Step 2: Compute all neighbor features in one pass -------------------
  cat("Computing neighbor statistics (all variables, vectorized)...\n")
  t0 <- proc.time()

  neighbor_stats <- compute_all_neighbor_stats_dt(
    cell_data, edge_dt, neighbor_source_vars
  )

  cat(sprintf("  Neighbor stats computed in %.1f seconds\n",
              (proc.time() - t0)[3]))

  # Bind neighbor features to cell_data
  # (data.table set-by-reference, no copy)
  for (col in names(neighbor_stats)) {
    set(cell_data, j = col, value = neighbor_stats[[col]])
  }
  rm(neighbor_stats, edge_dt)
  gc(verbose = FALSE)

  # ---- Step 3: Chunked prediction ------------------------------------------
  cat(sprintf("Predicting in chunks of %d rows...\n", chunk_size))
  t0 <- proc.time()

  cell_data[, predicted_gdp := predict_chunked(
    rf_model, cell_data, feature_cols, chunk_size
  )]

  cat(sprintf("  Prediction complete in %.1f seconds\n",
              (proc.time() - t0)[3]))

  # Clean up helper column
  cell_data[, row_idx := NULL]

  return(cell_data)
}


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
#
# # Load pre-trained model (do this once, keep in memory)
# rf_model <- readRDS("trained_rf_model.rds")
#
# # Load data
# cell_data <- readRDS("cell_data.rds")            # data.frame or data.table
# id_order  <- readRDS("id_order.rds")              # integer vector
# rook_neighbors_unique <- readRDS("rook_nb.rds")   # spdep nb object
#
# # Define the feature columns the model expects
# feature_cols <- names(rf_model$forest$xlevels)    # for randomForest
# # feature_cols <- rf_model$forest$independent.variable.names  # for ranger
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data              = cell_data,
#   id_order               = id_order,
#   rook_neighbors_unique  = rook_neighbors_unique,
#   rf_model               = rf_model,
#   feature_cols           = feature_cols
# )
```

---

## 4. Summary of Expected Improvements

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | `lapply` over 6.46M rows with `paste`+hash lookup | Vectorized `data.table` join (two keyed joins) | **~100â€“500Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, 6.46M list)` | Single grouped `data.table` aggregation | **~50â€“200Ã—** |
| Prediction | Single `predict()` on 6.46M Ã— 110 (peak ~12 GB) | Chunked 500K-row prediction, explicit `gc()` | **Fits in 16 GB; no swapping** |
| Memory copies | `data.frame` `$<-` triggers full-column copy per assignment | `data.table::set()` by reference, zero copies | **~5Ã— less peak RAM** |
| **Overall estimated runtime** | **86+ hours** | **~15â€“45 minutes** (depending on model type and disk I/O) | **~100â€“350Ã—** |

### Key Invariants Preserved
- **Trained RF model is never modified or retrained** â€” used read-only in `predict()`.
- **Original numerical estimand is preserved** â€” the same features are computed identically (max, min, mean of the same neighbor values), and the same `predict()` function is called, producing bit-identical predictions.
END_RESPONSE>>>
