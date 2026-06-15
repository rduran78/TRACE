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
run_id: Anthropic_F03_T07_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup into `idx_lookup`
- NA filtering

With 6.46M iterations of interpreted R code doing string operations, this alone could take **hours**.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length integer vector, subsetting a numeric vector, removing NAs, and computing three summary statistics. The `do.call(rbind, result)` at the end materialises a list of 6.46M 3-element vectors into a matrix â€” this is slow because `rbind` on a long list is O(nÂ²) in memory-copy cost.

**Outer loop over 5 variables:** `compute_and_add_neighbor_features` is called 5 times, each time copying the entire `cell_data` data.frame (6.46M Ã— 110+ columns). In base R, `cell_data$new_col <- ...` triggers a full copy of the data.frame each time (COW semantics notwithstanding, column addition forces a shallow-copy of the column-pointer vector and often a deep copy).

### 1.2 Prediction-Workflow Bottlenecks

Although the prediction code isn't shown, the common problems at this scale are:

| Issue | Impact |
|---|---|
| **Model object size** | `ranger` or `randomForest` models with many trees on 110 predictors can be 1â€“4 GB. Loading from disk and keeping in memory alongside 6.46M Ã— 110 data is tight on 16 GB. |
| **Predicting all 6.46M rows at once** | `predict.randomForest` copies the input to a matrix internally. With 110 numeric columns Ã— 6.46M rows Ã— 8 bytes â‰ˆ 5.3 GB â€” exceeds available RAM. Even `ranger::predict` builds an internal matrix. |
| **Repeated object copies** | If prediction is done inside a loop that subsets or re-binds data frames, each iteration copies data. |
| **Single-threaded prediction** | `randomForest::predict` is single-threaded. `ranger::predict` is multi-threaded by default but must be called correctly. |

### 1.3 Summary of Root Causes

1. **Interpreted R loops over 6.46M rows** (string ops in `build_neighbor_lookup`, per-row stats in `compute_neighbor_stats`).
2. **`do.call(rbind, ...)` on millions of small vectors** â€” quadratic memory allocation.
3. **Repeated full-copy of the data.frame** when adding columns in a loop.
4. **Likely RAM exhaustion** during prediction if all rows are passed at once.
5. **Possible use of single-threaded `randomForest` instead of `ranger`**.

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorised Feature Preparation with `data.table`

- Replace the per-row `lapply` in `build_neighbor_lookup` with a **vectorised merge/join** in `data.table`.
- Build an edge-list representation of the neighbor graph (cell_id â†’ neighbor_cell_id), join it to the panel data by (neighbor_cell_id, year), then compute grouped summary statistics with `data.table`'s `by=` â€” no R-level loop at all.
- Add all 15 neighbor-stat columns (5 vars Ã— 3 stats) in one pass, avoiding repeated data.frame copies.

**Expected speedup:** The entire neighbor-feature step drops from hours to **~1â€“3 minutes**.

### Strategy B: Chunked, Memory-Safe Prediction

- Predict in chunks of ~500Kâ€“1M rows to stay well within 16 GB RAM.
- If the model is `randomForest`, wrap it with `ranger` format or at minimum ensure `predict` is called with `num.threads`.
- Pre-allocate the output vector.

### Strategy C: General Memory Hygiene

- Use `data.table` (modify-in-place, no copy) instead of `data.frame`.
- `gc()` between major phases.
- Load the model once, predict, then `rm()` and `gc()` before the next phase if needed.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMISED PIPELINE â€” Feature Preparation + Prediction
# Dependencies: data.table, ranger (or randomForest), spdep (for nb object)
# =============================================================================

library(data.table)

# ---- 0. LOAD DATA -----------------------------------------------------------
# cell_data        : data.frame / data.table with columns id, year, ntl, ec,
#                    pop_density, def, usd_est_n2, ... (6.46M rows)
# rook_neighbors_unique : spdep nb object (list of integer index vectors)
# id_order         : integer/character vector mapping nb-list position -> cell id
# rf_model         : pre-trained model (ranger or randomForest)
# ------------------------------------------------------------------------------

# Convert to data.table in place (no copy if already a data.table)
setDT(cell_data)

# ---- 1. BUILD VECTORISED EDGE LIST FROM nb OBJECT ---------------------------

build_edge_list_dt <- function(id_order, neighbors) {
  # neighbors is an spdep nb object: a list where element i contains integer
  # indices (into id_order) of the neighbors of id_order[i].
  # We produce a data.table with columns: id (focal cell), neighbor_id.

  # Pre-compute lengths to allocate in one shot
  lens <- lengths(neighbors)
  total_edges <- sum(lens)

  focal_idx <- rep.int(seq_along(neighbors), lens)
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  data.table(
    id          = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building edge list...\n")
edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed edges)

cat("Edge list rows:", nrow(edge_dt), "\n")

# ---- 2. COMPUTE ALL NEIGHBOR FEATURES IN ONE VECTORISED PASS ----------------

compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # cell_dt must have columns: id, year, and all source_vars
  # edge_dt must have columns: id, neighbor_id
  #
  # For each source_var, we compute neighbor_max, neighbor_min, neighbor_mean.

  # Step 1: Join edge list with cell_dt to get (focal id, year) -> neighbor rows
  # We need: for each (id, year), find all (neighbor_id, year) rows and their values.

  # Key the cell data for fast join
  # We only need id, year, and the source variables for the neighbor lookup
  keep_cols <- c("id", "year", source_vars)
  neighbor_values <- cell_dt[, ..keep_cols]
  setnames(neighbor_values, "id", "neighbor_id")

  # Join: edge_dt gives us (id -> neighbor_id), then we join on (neighbor_id, year)
  # to get the neighbor's variable values for the same year as the focal cell.

  # First, add year to edge_dt by cross-joining with the unique years?
  # No â€” more efficient: join edge_dt to cell_dt to get focal (id, year), then
  # look up neighbor values.

  # Approach: create a long table of (id, year, neighbor_id) then join neighbor values.

  # To avoid a massive cross join, we do it in a memory-efficient way:
  # focal_dt has (id, year) â€” 6.46M rows
  # edge_dt has (id, neighbor_id) â€” 1.37M rows
  # Merge on id gives (id, year, neighbor_id) â€” roughly 6.46M * avg_neighbors rows
  # Average neighbors per cell â‰ˆ 1.37M / 344,208 â‰ˆ 4.0
  # So the merged table â‰ˆ 6.46M * 4 â‰ˆ 25.8M rows â€” manageable.

  cat("  Joining edges to panel years...\n")

  # Get (id, year) pairs from cell_dt
  focal_keys <- cell_dt[, .(id, year)]

  # Merge: for each (id, year), attach all neighbor_ids
  # Use data.table merge on 'id'
  setkey(edge_dt, id)
  setkey(focal_keys, id)
  expanded <- edge_dt[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  # ~25.8M rows

  cat("  Expanded edge-year rows:", nrow(expanded), "\n")

  # Now join neighbor values on (neighbor_id, year)
  setkey(neighbor_values, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  cat("  Joining neighbor values...\n")
  expanded <- neighbor_values[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has: neighbor_id, year, id, and all source_var columns
  # (values are from the NEIGHBOR cell for that year)

  # Compute grouped stats: for each (id, year), get max/min/mean of each var
  cat("  Computing grouped statistics...\n")

  # Build aggregation expressions programmatically
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats_dt <- expanded[, lapply(agg_exprs, eval), by = .(id, year)]

  # Fix Inf/-Inf from max/min on all-NA groups (groups with no valid neighbors)
  inf_cols <- grep("^neighbor_(max|min)_", names(stats_dt), value = TRUE)
  for (col in inf_cols) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  cat("  Stats computed. Merging back to cell_data...\n")

  # Merge stats back to cell_dt
  setkey(stats_dt, id, year)
  setkey(cell_dt, id, year)
  cell_dt <- stats_dt[cell_dt, on = .(id, year)]

  return(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
t0 <- proc.time()
cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
cat("Neighbor features done in", (proc.time() - t0)[3], "seconds\n")

# Clean up intermediate objects
rm(edge_dt)
gc()

# ---- 3. PREPARE PREDICTION MATRIX -------------------------------------------

# Identify the predictor columns the model expects.
# If using ranger:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores predictor names differently
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model))
}

# Verify all predictor columns are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns in cell_data: ",
       paste(missing_vars, collapse = ", "))
}

cat("Predictor variables verified:", length(pred_vars), "columns\n")

# ---- 4. CHUNKED PREDICTION --------------------------------------------------

predict_chunked <- function(model, dt, pred_vars, chunk_size = 500000L) {
  n <- nrow(dt)
  predictions <- numeric(n)

  n_chunks <- ceiling(n / chunk_size)
  cat("Predicting in", n_chunks, "chunks of up to", chunk_size, "rows...\n")

  is_ranger <- inherits(model, "ranger")

  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)
    idx       <- start_idx:end_idx

    # Extract chunk as a plain data.frame (required by predict methods)
    chunk_df <- as.data.frame(dt[idx, ..pred_vars])

    if (is_ranger) {
      pred_obj <- predict(model, data = chunk_df, num.threads = parallel::detectCores())
      predictions[idx] <- pred_obj$predictions
    } else {
      # randomForest::predict
      predictions[idx] <- predict(model, newdata = chunk_df)
    }

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %d-%d)\n", i, n_chunks, start_idx, end_idx))
    }

    # Free chunk memory
    rm(chunk_df, pred_obj)
    if (i %% 10 == 0) gc()
  }

  return(predictions)
}

cat("Starting prediction...\n")
t0 <- proc.time()
cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, pred_vars,
                                              chunk_size = 750000L)]
cat("Prediction done in", (proc.time() - t0)[3], "seconds\n")

# ---- 5. VERIFY OUTPUT -------------------------------------------------------
cat("Prediction summary:\n")
print(summary(cell_data$predicted_gdp))
cat("Total rows with predictions:", sum(!is.na(cell_data$predicted_gdp)), "\n")
cat("NA predictions:", sum(is.na(cell_data$predicted_gdp)), "\n")

# The original numerical estimand (GDP predictions from the trained RF) is preserved.
# The trained Random Forest model (rf_model) is never modified or retrained.
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Phase | Original (est.) | Optimised (est.) | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~10â€“20 hours (6.46M string-key lapply) | ~5 seconds (vectorised `rep.int` + `unlist`) | ~10,000Ã— |
| `compute_neighbor_stats` (Ã—5 vars) | ~40â€“60 hours (5 Ã— 6.46M lapply + `do.call(rbind)`) | ~60â€“120 seconds (one `data.table` grouped aggregation) | ~2,000Ã— |
| Data.frame column addition (Ã—15 cols) | ~5â€“10 hours (repeated full-copy) | ~0 (in-place `data.table` join) | eliminated |
| Prediction (6.46M rows) | ~10+ hours (possible OOM, single-thread) | ~15â€“45 min (chunked, multi-threaded ranger) | ~15â€“40Ã— |
| **Total** | **~86+ hours** | **~20â€“50 minutes** | **~100â€“250Ã—** |

### Key Principles Applied

1. **Eliminate interpreted R loops over millions of rows** â€” replaced with vectorised `data.table` joins and grouped aggregations that run in C.
2. **Eliminate `do.call(rbind, list_of_millions)`** â€” replaced with a single `data.table` `[, ..., by=]` that allocates the result matrix once.
3. **Eliminate repeated data.frame copies** â€” `data.table` modifies in place with `:=` and `set()`.
4. **Chunk prediction to fit in 16 GB RAM** â€” 750K rows Ã— 110 columns Ã— 8 bytes â‰ˆ 0.66 GB per chunk, well within budget.
5. **Multi-threaded prediction** â€” `ranger::predict` with `num.threads = detectCores()`.
6. **Model and estimand preserved** â€” the trained RF object is only used for `predict()`; no retraining occurs; the numerical output is identical.
END_RESPONSE>>>
