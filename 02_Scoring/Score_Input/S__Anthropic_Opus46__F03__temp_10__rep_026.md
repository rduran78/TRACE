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
run_id: Anthropic_F03_T10_R026
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 26
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_026.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The 86+ hour runtime on ~6.46 million rows with ~110 features has **two major bottleneck zones**:

### A. Feature Preparation (Neighbor Lookup & Stats)

| Bottleneck | Root Cause |
|---|---|
| **`build_neighbor_lookup`**: character key creation & named-vector lookup for every row | `paste()` creates ~6.46M strings; `setNames` + named-vector indexing (`idx_lookup[neighbor_keys]`) is O(n) hash-table construction and then millions of repeated hash lookups inside an `lapply` over 6.46M rows. |
| **`compute_neighbor_stats`**: `lapply` over 6.46M rows, each extracting a variable-length neighbor vector, removing NAs, computing `max/min/mean` | Pure-R per-row loop with repeated subsetting, allocation, and NA removal. Called **5 times** (once per source variable), so ~32.3M R-level iterations. |
| **Object copying via `cell_data <- compute_and_add_neighbor_features(...)`** | Each call likely copies or grows the full data.frame (~6.46M Ã— 110+ columns), triggering R's copy-on-modify. With 5 variables Ã— 3 stats = 15 new columns, this means up to 15 full-data-frame copies. |

### B. Random Forest Inference

| Bottleneck | Root Cause |
|---|---|
| **Predicting 6.46M rows at once** with `predict.randomForest` or `predict.ranger` | `randomForest::predict` is notoriously slow on large data â€” it passes every row through every tree in pure R/C with a single-threaded loop. Even `ranger::predict` can be memory-bound if the full prediction matrix is materialised as a dense `data.frame`. |
| **Memory pressure** | 6.46M rows Ã— 110 features Ã— 8 bytes â‰ˆ 5.7 GB just for the numeric matrix. Combined with the model object, tree traversal workspace, and R overhead, 16 GB RAM is tight â€” likely causes swapping. |
| **Model loading** | If the model is re-loaded from disk on every run (or worse, inside a loop), deserialization of a large RF object is expensive. |

### Summary: Estimated Time Split (approximate)

| Phase | Estimated Share |
|---|---|
| `build_neighbor_lookup` | ~15â€“25% |
| `compute_neighbor_stats` (Ã—5 vars) | ~25â€“35% |
| Data.frame copying / column binding | ~10â€“15% |
| RF prediction (single-threaded, memory-bound) | ~25â€“40% |

---

## 2. Optimization Strategy

### Feature Preparation

1. **Replace `data.frame` with `data.table`** â€” eliminates copy-on-modify, enables in-place column addition via `:=`.
2. **Replace string-key lookup with integer arithmetic** â€” encode `(id, year)` as a direct integer index using a pre-built matrix or a fast hash via `data.table`.
3. **Vectorize neighbor stats computation** â€” replace per-row `lapply` with a single grouped `data.table` operation using an edge-list representation (flat two-column table of `row â†’ neighbor_row`), then group-by aggregation.
4. **Compute all 5 variables' stats in one pass** over the edge list rather than 5 separate passes.

### Random Forest Inference

1. **Ensure the model is `ranger`, not `randomForest`** â€” `ranger::predict` is multi-threaded and much faster. If the existing model is `randomForest`, wrap it once to extract predictions with a chunked approach.
2. **Chunk prediction** â€” split the 6.46M rows into chunks of ~500K to stay within RAM, predict each chunk, and concatenate.
3. **Load model once** and keep in memory.
4. **Convert prediction input to a `matrix`** (not `data.frame`) â€” `ranger` and `randomForest` both benefit from contiguous numeric storage.

### Memory

1. `data.table` in-place ops keep peak memory lower.
2. Chunked prediction avoids duplicating the full feature matrix.
3. Remove intermediate objects with `rm()` + `gc()` at strategic points.

**Expected speedup: ~50â€“200Ã—** (from 86+ hours to roughly 30â€“90 minutes).

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- 0. Load model once ------------------------------------------------------
# Assumes rf_model is already loaded or load it once here:
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# Keep rf_model in memory for the entire session.

# ---- 1. Convert to data.table (in place, no copy) ----------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# ---- 2. Build fast neighbor edge-list ----------------------------------------
# This replaces build_neighbor_lookup entirely.
# Produces a two-column data.table: (row_i, neighbor_row_j)

build_neighbor_edgelist <- function(dt, id_order, neighbors_nb) {
  # --- Map cell id -> position in id_order (reference index) ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Map (id, year) -> row number using data.table keyed join ---
  #     Much faster than paste + named vector lookup
  dt[, .row_idx := .I]
  key_dt <- dt[, .(id, year, .row_idx)]
  setkey(key_dt, id, year)

  # --- Expand nb object to an edge-list of (ref_idx, neighbor_ref_idx) ---
  #     spdep::nb objects are a list of integer vectors
  edge_ref <- rbindlist(lapply(seq_along(neighbors_nb), function(i) {
    nb <- neighbors_nb[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(NULL)
    }
    data.table(ref_i = i, ref_j = nb)
  }))

  # Map ref index -> cell id
  edge_ref[, id_i := id_order[ref_i]]
  edge_ref[, id_j := id_order[ref_j]]

  # --- Cross-join with years: for each (id_i, id_j) pair,
  #     create rows for every year present in the data ---
  years_present <- sort(unique(dt$year))

  # Expand: every edge Ã— every year
  edge_year <- edge_ref[, .(id_i, id_j)][
    , CJ_dt := TRUE  # placeholder
  ]
  # More memory-efficient: use a cross join
  edge_year <- CJ(edge_idx = seq_len(nrow(edge_ref)),
                   year = years_present)
  edge_year[, id_i := edge_ref$id_i[edge_idx]]
  edge_year[, id_j := edge_ref$id_j[edge_idx]]

  # --- Map to actual row indices via keyed join ---
  # Row of the focal cell (i)
  edge_year[key_dt, row_i := i..row_idx, on = .(id_i = id, year)]
  # Row of the neighbor cell (j)
  edge_year[key_dt, row_j := i..row_idx, on = .(id_j = id, year)]

  # Drop edges where either focal or neighbor row doesn't exist
  edge_year <- edge_year[!is.na(row_i) & !is.na(row_j),
                          .(row_i, row_j)]

  # Clean up temporary column
  dt[, .row_idx := NULL]

  return(edge_year)
}

cat("Building neighbor edge-list...\n")
system.time({
  edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
})
# edge_dt has columns: row_i (focal row), row_j (neighbor row)
cat(sprintf("Edge-list rows: %s\n", format(nrow(edge_dt), big.mark = ",")))

# ---- 3. Compute all neighbor features in one vectorised pass -----------------
# For each (source_var), compute max, min, mean of neighbor values,
# then join back to cell_data by row_i.

compute_all_neighbor_features <- function(dt, edge, var_names) {
  # Extract neighbor values for ALL variables at once
  # edge has (row_i, row_j); we need vals[row_j] for each variable

  # Build a sub-table of neighbor values
  neighbor_vals <- edge[, .(row_i, row_j)]

  # Add all variable values at the neighbor row
  for (v in var_names) {
    set(neighbor_vals, j = v, value = dt[[v]][neighbor_vals$row_j])
  }

  # Group by row_i and compute stats for each variable
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))

  # Build the aggregation call dynamically
  # Faster approach: compute in a single data.table grouped operation
  stats <- neighbor_vals[,
    {
      out <- list()
      for (v in var_names) {
        vals <- get(v)
        vals <- vals[!is.na(vals)]
        if (length(vals) == 0L) {
          out[[paste0("n_", v, "_max")]]  <- NA_real_
          out[[paste0("n_", v, "_min")]]  <- NA_real_
          out[[paste0("n_", v, "_mean")]] <- NA_real_
        } else {
          out[[paste0("n_", v, "_max")]]  <- max(vals)
          out[[paste0("n_", v, "_min")]]  <- min(vals)
          out[[paste0("n_", v, "_mean")]] <- mean(vals)
        }
      }
      out
    },
    by = row_i
  ]

  # Join back to dt by row index
  setkey(stats, row_i)

  # Assign columns in place
  feat_cols <- setdiff(names(stats), "row_i")
  for (fc in feat_cols) {
    set(dt, i = stats$row_i, j = fc, value = stats[[fc]])
  }
  # Rows not present in stats (no neighbors) remain NA â€” initialise them
  missing_rows <- setdiff(seq_len(nrow(dt)), stats$row_i)
  if (length(missing_rows) > 0L) {
    for (fc in feat_cols) {
      set(dt, i = missing_rows, j = fc, value = NA_real_)
    }
  }

  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorised, single pass)...\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# Free the large edge table
rm(edge_dt); gc()

# ---- 4. Chunked Random Forest Prediction -------------------------------------

predict_chunked <- function(model, dt, feature_cols, chunk_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)
  n_chunks <- ceiling(n / chunk_size)

  # Determine predict function based on model class
  is_ranger <- inherits(model, "ranger")

  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))

  for (k in seq_len(n_chunks)) {
    start_i <- (k - 1L) * chunk_size + 1L
    end_i   <- min(k * chunk_size, n)
    idx     <- start_i:end_i

    # Extract chunk as a plain matrix for speed
    chunk_mat <- as.matrix(dt[idx, ..feature_cols])

    if (is_ranger) {
      # ranger::predict is multi-threaded
      preds[idx] <- predict(model, data = chunk_mat)$predictions
    } else {
      # randomForest::predict â€” single-threaded but works on matrix
      preds[idx] <- predict(model, newdata = chunk_mat)
    }

    if (k %% 5 == 0 || k == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
                  k, n_chunks,
                  format(start_i, big.mark = ","),
                  format(end_i, big.mark = ",")))
    }
    gc()  # free chunk memory
  }

  return(preds)
}

# Identify feature columns (all predictors used by the model)
# Adjust this to match your model's expected feature names:
if (inherits(rf_model, "ranger")) {
  feature_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores row names of importance matrix
  feature_cols <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

cat("Starting chunked prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, feature_cols,
                                                chunk_size = 500000L)]
})

cat("Done.\n")
```

---

## 4. Why This Works â€” Performance Comparison

| Operation | Original | Optimised | Speedup |
|---|---|---|---|
| **Neighbor lookup construction** | `paste` + named vector over 6.46M rows â‰ˆ 20â€“40 min | `data.table` keyed join, built once â‰ˆ 1â€“3 min | ~10â€“20Ã— |
| **Neighbor stats (5 vars Ã— 6.46M rows)** | 5 Ã— `lapply` over 6.46M rows â‰ˆ several hours | Single grouped `data.table` aggregation on flat edge-list â‰ˆ 3â€“10 min | ~30â€“60Ã— |
| **Column binding / object copying** | 15 `data.frame` copies of ~6.46M rows | `data.table` `:=` / `set()` in place â€” **zero copies** | âˆž (eliminated) |
| **RF prediction** | One monolithic `predict()` call on 6.46M Ã— 110 â†’ ~5.7 GB matrix, possible swap | Chunked 500K-row matrix slices, fits in RAM | ~2â€“5Ã— (avoids swap) |
| **Total estimated** | **86+ hours** | **~30â€“90 minutes** | **~60â€“170Ã—** |

### Key Guarantees

- **Trained RF model is never modified** â€” only `predict()` is called.
- **Numerical estimand is preserved** â€” identical `max`, `min`, `mean` computations; identical `predict()` output; no rounding, approximation, or model alteration.
- **No retraining** â€” the model object is used as-is from disk.
END_RESPONSE>>>
