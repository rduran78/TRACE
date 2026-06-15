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
run_id: Anthropic_F03_T07_R023
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 23
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_023.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookups, and NA filtering â€” all inside an interpreted R loop. |
| **`paste` key construction** | Creating ~6.46M Ã— avg-4-neighbors string keys (â‰ˆ26M strings) is extremely expensive. |
| **Named-vector lookup (`idx_lookup[neighbor_keys]`)** | R's named vector lookup is O(n) hash-probe per call; doing it ~26M times is slow. |
| **Output is a list of 6.46M integer vectors** | Massive memory overhead from list structure (~50â€“100 bytes per list element overhead Ã— 6.46M â‰ˆ 0.5â€“1 GB just for the container). |

**`compute_neighbor_stats`** is the second bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements** | Each call extracts a small integer vector, subsets, removes NAs, and computes three summary stats â€” all in interpreted R. |
| **Called 5 times** | Once per neighbor source variable, so the full loop runs ~32M R-function invocations. |
| **`do.call(rbind, result)` on 6.46M rows** | Binding 6.46M 3-element vectors into a matrix is itself slow and memory-hungry. |

### B. Random Forest Inference Bottleneck

| Problem | Detail |
|---|---|
| **Single monolithic `predict()` call on 6.46M Ã— 110 columns** | `ranger`/`randomForest` predict must build the full data frame in memory. With 110 numeric columns this is ~5.4 GB for the data alone, plus internal copies. On a 16 GB laptop this causes swapping. |
| **Object copying** | If the model is a `randomForest` object, `predict.randomForest` internally coerces to `data.frame` and may copy. Even `ranger` will allocate a prediction workspace. |
| **No chunking** | A single call means peak memory = model + full dataset + prediction workspace simultaneously. |

### C. Estimated Time Breakdown (86+ hours)

| Phase | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~25â€“35% |
| `compute_neighbor_stats` (Ã—5 vars) | ~30â€“40% |
| RF `predict` (memory pressure / swapping) | ~25â€“35% |

---

## 2. Optimization Strategy

### Feature Preparation: Vectorize Everything with `data.table`

1. **Replace the string-key lookup with a `data.table` integer join.** Convert the neighbor list into a flat edge-list `data.table` with columns `(row_i, neighbor_row_j)`. Build this once using vectorized operations.
2. **Replace `lapply`-based `compute_neighbor_stats` with a grouped `data.table` aggregation** on the flat edge-list joined to the value column â€” one vectorized pass per variable.
3. **Eliminate the 6.46M-element list entirely.**

Expected speedup: **50â€“200Ã—** for feature preparation (minutes instead of tens of hours).

### RF Inference: Chunked Prediction

1. **Predict in chunks** (e.g., 500K rows) to keep peak memory well under 16 GB.
2. **Use `ranger` if possible** (faster C++ predict path); if the model is `randomForest`, still chunk.
3. **Pre-allocate the output vector** and fill in-place.

### Memory

1. Use `data.table` in-place `:=` assignment instead of repeated `cell_data <- cbind(...)` copies.
2. Remove intermediate objects and `gc()` between phases.

---

## 3. Working R Code

```r
# ============================================================
# 0. Dependencies
# ============================================================
library(data.table)
# library(ranger)        # if model is ranger
# library(randomForest)  # if model is randomForest

# ============================================================
# 1. Convert cell_data to data.table (in-place, no copy)
# ============================================================
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure a row index exists (used for joins)
cell_data[, .row_idx := .I]

# ============================================================
# 2. Build flat neighbor edge-list (vectorised)
#    Replaces build_neighbor_lookup entirely
# ============================================================
build_neighbor_edgelist <- function(cell_dt, id_order, neighbors) {
  # --- Map each cell id to its position in id_order ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build a lookup: (id, year) -> row index ---
  key_dt <- cell_dt[, .(id, year, .row_idx)]
  setkey(key_dt, id, year)

  # --- Expand neighbor list into flat (ref_idx, neighbor_cell_id) pairs ---
  #     neighbors[[k]] gives the neighbor positions in id_order for id_order[k]
  n_lengths <- lengths(neighbors)
  from_ref  <- rep(seq_along(neighbors), times = n_lengths)
  to_ref    <- unlist(neighbors, use.names = FALSE)

  from_id <- id_order[from_ref]
  to_id   <- id_order[to_ref]

  edge_dt <- data.table(from_id = from_id, to_id = to_id)

  # --- For every (from_id, year) row, find the row index of (to_id, year) ---
  #     Step 1: attach row indices and years for from_id
  from_info <- cell_dt[, .(from_id = id, year, from_row = .row_idx)]
  edge_year <- edge_dt[from_info, on = "from_id", allow.cartesian = TRUE,
                        nomatch = NULL]
  # edge_year now has: from_id, to_id, year, from_row

  # --- Step 2: join to get to_row (row index of the neighbor in same year) ---
  to_info <- cell_dt[, .(to_id = id, year, to_row = .row_idx)]
  setkey(to_info, to_id, year)
  setkey(edge_year, to_id, year)

  edge_year <- to_info[edge_year, nomatch = NA]
  # Keep only matched rows
  edge_year <- edge_year[!is.na(to_row)]

  # Return minimal columns
  edge_year[, .(from_row, to_row)]
}

cat("Building neighbor edge-list â€¦\n")
system.time({
  edge_list <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
  setkey(edge_list, from_row)
})
# edge_list is ~10-30M rows Ã— 2 integer columns â‰ˆ 200-500 MB


# ============================================================
# 3. Compute & attach neighbor features (vectorised)
#    Replaces compute_neighbor_stats + outer loop
# ============================================================
compute_and_add_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  # Attach the neighbor's value to each edge
  edge_dt[, val := cell_dt[[var_name]][to_row]]

  # Grouped aggregation: max, min, mean per from_row
  stats <- edge_dt[!is.na(val),
                   .(nb_max  = max(val),
                     nb_min  = min(val),
                     nb_mean = mean(val)),
                   by = from_row]

  # Column names matching original pipeline convention
  col_max  <- paste0("neighbor_", var_name, "_max")
  col_min  <- paste0("neighbor_", var_name, "_min")
  col_mean <- paste0("neighbor_", var_name, "_mean")

  # In-place assignment (no copy of cell_dt)
  cell_dt[, (col_max)  := NA_real_]
  cell_dt[, (col_min)  := NA_real_]
  cell_dt[, (col_mean) := NA_real_]

  cell_dt[stats$from_row, (col_max)  := stats$nb_max]
  cell_dt[stats$from_row, (col_min)  := stats$nb_min]
  cell_dt[stats$from_row, (col_mean) := stats$nb_mean]

  # Clean up temp column on edge_dt
  edge_dt[, val := NULL]

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features â€¦\n")
system.time({
  for (var_name in neighbor_source_vars) {
    cat("  ", var_name, "\n")
    compute_and_add_neighbor_features_fast(cell_data, edge_list, var_name)
  }
})

# Free the edge list
rm(edge_list); gc()


# ============================================================
# 4. Chunked Random Forest Prediction
#    Preserves trained model; preserves original numerical output
# ============================================================
chunked_predict_rf <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  preds <- numeric(n)           # pre-allocate full output vector

  starts <- seq(1L, n, by = chunk_size)
  n_chunks <- length(starts)

  for (ci in seq_along(starts)) {
    i1 <- starts[ci]
    i2 <- min(i1 + chunk_size - 1L, n)
    cat(sprintf("  Predicting chunk %d / %d  (rows %d â€“ %d)\n",
                ci, n_chunks, i1, i2))

    chunk <- newdata[i1:i2, ]

    # --- Works for both ranger and randomForest objects ---
    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = chunk)$predictions
    } else {
      # randomForest
      preds[i1:i2] <- predict(model, newdata = chunk)
    }

    # Hint GC to release chunk workspace
    rm(chunk)
    if (ci %% 5 == 0) gc()
  }

  preds
}

# --- Prepare prediction data frame (drop non-feature columns) ---
# Adjust feature_cols to match whatever the trained model expects.
# If using ranger, feature_cols can be extracted from the model object.
if (inherits(rf_model, "ranger")) {
  feature_cols <- rf_model$forest$independent.variable.names
} else {
  # randomForest stores variable names used during training
  feature_cols <- rownames(importance(rf_model))
}

# Subset to feature columns only (avoids sending id/year/target to predict)
pred_data <- cell_data[, ..feature_cols]   # data.table column selection

cat("Running chunked RF prediction â€¦\n")
system.time({
  cell_data[, predicted_gdp := chunked_predict_rf(rf_model, pred_data,
                                                   chunk_size = 500000L)]
})

rm(pred_data); gc()

cat("Done.\n")
```

---

## 4. Expected Impact

| Phase | Before | After | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~20â€“30 hrs | ~2â€“5 min | ~300Ã— |
| `compute_neighbor_stats` (Ã—5) | ~30â€“40 hrs | ~3â€“8 min | ~300Ã— |
| RF `predict` | ~15â€“20 hrs (with swap) | ~20â€“60 min (in-RAM) | ~20Ã— |
| **Total** | **86+ hrs** | **~30â€“75 min** | **~70â€“170Ã—** |

### Why Each Change Matters

1. **`build_neighbor_edgelist`**: Replaces 6.46M interpreted R iterations (each with `paste`, character matching) with a single vectorized `data.table` cartesian join. The join uses C-level hash/merge â€” orders of magnitude faster.

2. **`compute_and_add_neighbor_features_fast`**: Replaces 6.46M `lapply` calls per variable with one `data.table` grouped aggregation (`[, .(max, min, mean), by = from_row]`), which runs in optimized C with radix-sort grouping. In-place `:=` assignment avoids copying the 6.46M-row table.

3. **Chunked prediction**: Keeps peak memory at roughly `model_size + chunk_size Ã— 110 columns` (~400 MB per chunk) instead of `model_size + 6.46M Ã— 110` (~5.4 GB), eliminating OS swap thrashing on a 16 GB laptop. The numerical predictions are identical â€” chunking only controls how many rows enter `predict()` at once.
END_RESPONSE>>>
