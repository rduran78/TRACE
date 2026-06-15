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
run_id: Anthropic_F03_T10_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows using `lapply`, performing per-row string pasting (`paste(id, year)`), hash-table lookups, and NA filtering. With ~6.46M iterations, each involving character coercion, string concatenation, and named-vector lookups, this alone can take hours. Named vector lookups in R are O(n) in pathological cases and carry per-call overhead from character matching.

**`compute_neighbor_stats`:** Called 5 times (once per variable), it iterates over 6.46M rows again via `lapply`, each time subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end builds a 6.46M Ã— 3 matrix from a list of 6.46M length-3 vectors â€” this is extremely slow and memory-intensive.

**Outer loop:** The sequential `for` loop over 5 variables calls `compute_neighbor_stats` five separate times, re-traversing the neighbor lookup each time rather than computing all variables in a single pass.

### 1.2 Random Forest Inference Bottleneck

Calling `predict()` on a single `randomForest` or `ranger` model object with 6.46M rows and ~110 predictors is itself expensive. If the model was trained with the `randomForest` package (not `ranger`), prediction is done in pure R/C with poor memory locality and no parallelism. Additionally:

- If prediction is done in a **row-by-row or chunked loop** rather than a single vectorized call, overhead multiplies dramatically.
- If the prediction data.frame is being **copied** (e.g., via `data.frame()` construction or column-binding inside a loop), each copy of a 6.46M Ã— 110 frame is ~5â€“6 GB, which on a 16 GB machine causes swapping.
- Model objects from `randomForest` can be very large in memory; loading and holding them alongside the data may exceed RAM.

### 1.3 Summary of Root Causes

| Cause | Impact |
|---|---|
| Per-row `lapply` with string ops in `build_neighbor_lookup` | Hours of wall time for 6.46M rows |
| `do.call(rbind, ...)` on millions of small vectors | Massive memory allocation + GC pressure |
| Repeated traversal of neighbor lookup (5Ã—) | 5Ã— redundant iteration |
| Likely row-wise or copy-heavy prediction workflow | Memory thrashing on 16 GB laptop |
| Possible use of `randomForest::predict` instead of `ranger::predict` | Slow single-threaded C inference |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorized with `data.table`

- Replace all character-keyed lookups with integer-keyed joins via `data.table`.
- Build the neighbor lookup as a **two-column integer edge-list** (`row_i`, `neighbor_row_j`), avoiding per-row `lapply` entirely.
- Compute all neighbor stats (max, min, mean) across **all 5 variables simultaneously** in a single grouped `data.table` aggregation on the edge-list â€” one pass, fully vectorized in C.
- This eliminates `do.call(rbind, ...)`, per-row `lapply`, and string operations.

### 2.2 Random Forest Inference

- If the model is a `randomForest` object, convert it to `ranger` format or, if that's not feasible, use `predict()` in a **single vectorized call** on the full matrix, ensuring no per-row loop.
- If `ranger` is usable, re-wrap the predict call with `ranger::predict` which is multi-threaded.
- Prepare the prediction input as a **matrix** (not data.frame) to avoid method-dispatch overhead and column-type checking on each tree traversal.
- Predict in **chunks** (e.g., 500K rows) only if memory is the binding constraint, to stay within 16 GB.

### 2.3 Memory Management

- Use in-place column assignment (`:=` in `data.table`) to avoid copying the full 6.46M-row table.
- Remove intermediate objects and call `gc()` before prediction.
- Ensure only one copy of the data exists at prediction time.

### Expected Speedup

| Stage | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~10â€“20 hrs | ~10â€“30 sec |
| `compute_neighbor_stats` (Ã—5) | ~40â€“60 hrs | ~20â€“60 sec |
| RF prediction | ~5â€“10 hrs | ~5â€“20 min |
| **Total** | **~86+ hrs** | **~10â€“25 min** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (optional, for faster predict)
# =============================================================================

library(data.table)

# ---- 0. Load pre-trained model and data ------------------------------------
# Adjust paths as needed
# rf_model       <- readRDS("trained_rf_model.rds")
# cell_data      <- readRDS("cell_data.rds")           # data.frame or data.table
# id_order       <- readRDS("id_order.rds")             # vector of cell IDs
# rook_neighbors_unique <- readRDS("rook_neighbors.rds") # spdep nb object

# ---- 1. Convert to data.table in place -------------------------------------
if (!is.data.table(cell_data)) setDT(cell_data)

# ---- 2. Build vectorized neighbor edge-list --------------------------------
build_neighbor_edgelist <- function(dt, id_order, neighbors) {
  # Map each cell ID to its position in id_order (1-based)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a row-index keyed by (id, year) â€” integer key for speed
  # We assign each row a sequential row number
  dt[, .row_idx := .I]

  # Create an integer-keyed lookup: for each unique (id, year) -> row index
  # Use data.table keyed join instead of named character vector
  row_key <- dt[, .(id, year, .row_idx)]
  setkey(row_key, id, year)

  # Expand the spdep nb object into a two-column edge-list:
  #   focal_ref (index into id_order) -> neighbor_ref (index into id_order)
  n_refs <- length(neighbors)
  focal_refs <- rep(seq_len(n_refs),
                    times = lengths(neighbors))
  neighbor_refs <- unlist(neighbors, use.names = FALSE)

  # Filter out 0-length (isolated) nodes â€” already handled by rep/unlist
  # Convert ref indices to actual cell IDs
  edge_dt <- data.table(
    focal_id    = id_order[focal_refs],
    neighbor_id = id_order[neighbor_refs]
  )

  # Now cross-join with years: each edge applies to every year
  # Instead of a full cross join (expensive), we join via the data
  # For each row in dt, find its focal_id, then look up neighbor rows

  # Step A: For each focal cell, list its neighbor cell IDs
  # (this is small: ~344K cells, ~1.37M edges)
  # Step B: For each row in dt, get the neighbor cell IDs, then find
  #         the row indices of (neighbor_id, same year)

  # Efficient approach: join edge_dt with row_key on focal side,
  # then join again on neighbor side for the same year.

  # Get all (focal_id, year, neighbor_id) combinations by joining
  # dt's (id, year, row_idx) with edge_dt on focal_id

  setnames(edge_dt, c("focal_id", "neighbor_id"))

  # Join: for each row in dt, get its neighbor IDs
  # focal_rows: (focal_id, year, focal_row_idx)
  focal_rows <- dt[, .(focal_id = id, year, focal_row_idx = .row_idx)]
  setkey(focal_rows, focal_id)
  setkey(edge_dt, focal_id)

  # This is the large join: 6.46M rows Ã— ~4 neighbors each â‰ˆ 25.8M rows
  expanded <- edge_dt[focal_rows,
                      .(focal_row_idx, neighbor_id, year),
                      on = "focal_id",
                      allow.cartesian = TRUE,
                      nomatch = 0L]

  # Now find the row index of each (neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  # row_key is keyed on (id, year)
  expanded[row_key,
           neighbor_row_idx := i..row_idx,
           on = c(neighbor_id = "id", "year")]

  # Drop rows where neighbor was not found (boundary cells in some years)
  expanded <- expanded[!is.na(neighbor_row_idx)]

  # Clean up
  dt[, .row_idx := NULL]

  return(expanded[, .(focal_row_idx, neighbor_row_idx)])
}

cat("Building neighbor edge-list...\n")
system.time({
  edge_list <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~10-30 seconds, ~25M rows, two integer columns (~200 MB)

# ---- 3. Compute all neighbor stats in one vectorized pass ------------------
compute_all_neighbor_stats <- function(dt, edge_list, var_names) {
  # Extract the neighbor values for ALL variables at once
  # edge_list has columns: focal_row_idx, neighbor_row_idx

  # Build a sub-table of neighbor values
  neighbor_vals <- dt[edge_list$neighbor_row_idx, ..var_names]
  neighbor_vals[, focal_row_idx := edge_list$focal_row_idx]

  # Aggregate: for each focal_row_idx, compute max/min/mean of each variable
  # Use data.table's efficient grouped aggregation
  agg_exprs <- list()
  for (v in var_names) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  agg_result <- neighbor_vals[, eval(agg_call), by = focal_row_idx]

  # Replace Inf/-Inf from max/min of all-NA groups with NA
  inf_cols <- grep("^n_max_|^n_min_", names(agg_result), value = TRUE)
  for (col in inf_cols) {
    vals <- agg_result[[col]]
    set(agg_result, which(is.infinite(vals)), col, NA_real_)
  }

  return(agg_result)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics for all variables...\n")
system.time({
  neighbor_stats <- compute_all_neighbor_stats(cell_data, edge_list, neighbor_source_vars)
})
# Expected: ~20-60 seconds

# ---- 4. Join neighbor stats back to the main table -------------------------
cat("Joining neighbor features to main data...\n")

# Ensure row indices align
cell_data[, .row_idx := .I]
setkey(neighbor_stats, focal_row_idx)

# Join in place
stat_cols <- setdiff(names(neighbor_stats), "focal_row_idx")
cell_data[neighbor_stats, (stat_cols) := mget(paste0("i.", stat_cols)),
          on = c(.row_idx = "focal_row_idx")]
cell_data[, .row_idx := NULL]

# Rows without neighbors will have NA â€” this matches original behavior

# Free memory
rm(edge_list, neighbor_stats)
gc()

# ---- 5. Prepare prediction matrix -----------------------------------------
cat("Preparing prediction input...\n")

# Identify the predictor columns (adjust to match your trained model)
# If your model was trained with specific feature names, use those:
predictor_cols <- setdiff(names(cell_data),
                          c("id", "year", "gdp", "gdp_predicted",
                            # add any other non-predictor columns here
                            "geometry"))

# Verify we have the expected ~110 predictors
cat(sprintf("  Number of predictor columns: %d\n", length(predictor_cols)))

# Build a matrix for faster predict (if model supports it)
# For randomForest/ranger, data.frame is required; but we avoid copies
pred_data <- cell_data[, ..predictor_cols]

# ---- 6. Predict using the trained Random Forest ---------------------------
cat("Running Random Forest prediction...\n")

# Detect model type and predict accordingly
if (inherits(rf_model, "ranger")) {
  # ranger: natively multi-threaded, fast
  system.time({
    preds <- predict(rf_model, data = pred_data,
                     num.threads = parallel::detectCores())$predictions
  })
} else if (inherits(rf_model, "randomForest")) {
  # randomForest package: single-threaded, slower
  # Predict in chunks to manage memory (each chunk ~ 500K rows)
  chunk_size <- 500000L
  n_rows     <- nrow(pred_data)
  n_chunks   <- ceiling(n_rows / chunk_size)
  preds      <- numeric(n_rows)

  cat(sprintf("  Predicting in %d chunks of up to %d rows...\n",
              n_chunks, chunk_size))

  system.time({
    for (ch in seq_len(n_chunks)) {
      start_i <- (ch - 1L) * chunk_size + 1L
      end_i   <- min(ch * chunk_size, n_rows)
      preds[start_i:end_i] <- predict(rf_model,
                                       newdata = pred_data[start_i:end_i, ])
      if (ch %% 5 == 0) {
        cat(sprintf("    Chunk %d/%d done\n", ch, n_chunks))
        gc()
      }
    }
  })
} else {
  # Generic fallback
  system.time({
    preds <- predict(rf_model, newdata = pred_data)
  })
}

# ---- 7. Attach predictions ------------------------------------------------
cell_data[, gdp_predicted := preds]

# Free prediction input
rm(pred_data, preds)
gc()

cat("Done. Predictions stored in cell_data$gdp_predicted.\n")

# =============================================================================
# OPTIONAL: If you have a randomForest model and want maximum speed,
# convert it to a ranger-compatible prediction. This preserves the original
# trained model's numerical output exactly IF the conversion is lossless.
# Otherwise, keep the chunked predict() above.
#
# NOTE: There is no automatic lossless conversion from randomForest to ranger.
# If retraining is forbidden, use the chunked approach for randomForest models.
# =============================================================================
```

---

## 4. KEY POINTS AND GUARANTEES

| Requirement | How It Is Met |
|---|---|
| **Preserve trained RF model** | The model object is loaded from disk and used read-only via `predict()`. No retraining occurs. |
| **Preserve original numerical estimand** | `predict()` is called on the same model with the same features; neighbor stats use identical formulae (max, min, mean). Infinite values from all-NA groups are mapped to `NA`, matching the original `if (length(neighbor_vals) == 0) return(c(NA, NA, NA))` logic. |
| **16 GB RAM feasibility** | `data.table` in-place operations avoid copies. Edge-list is ~200 MB. Chunked prediction (for `randomForest`) keeps peak memory well under 16 GB. Intermediate objects are freed with `rm()` + `gc()`. |
| **Speed** | String operations and per-row `lapply` are eliminated. All neighbor computation is a single vectorized `data.table` grouped aggregation. Expected total runtime: **~10â€“25 minutes** vs. 86+ hours. |
END_RESPONSE>>>
