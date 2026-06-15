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
run_id: Anthropic_F03_T00_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - Character coercion (`as.character`), named-vector lookups (`id_to_ref[...]`), `paste` to build keys, and named-vector indexing (`idx_lookup[neighbor_keys]`).
  - Named-vector lookup in R is **O(n)** hash-table probing per call. Doing this 6.46M times with string keys is extremely slow.
- The output is a **list of 6.46M integer vectors** â€” a large, fragmented memory structure.

**`compute_neighbor_stats`** is the second bottleneck:

- Another `lapply` over 6.46M elements, each extracting a subset of a numeric vector, removing NAs, and computing `max/min/mean`.
- Called **5 times** (once per neighbor source variable), so ~32.3M R-level function invocations.
- `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is itself very slow (repeated memory allocation).

**Overall feature-prep cost**: The nested string operations, millions of R-level function calls, and repeated list-to-matrix conversions dominate. This alone likely accounts for many hours.

### B. Random Forest Inference Bottleneck

- Predicting ~6.46M rows Ã— ~110 features through a Random Forest (even a pre-trained one) is inherently expensive.
- If `predict.randomForest` is called **once per row** or in small batches, overhead is catastrophic. Even a single vectorized call on 6.46M rows can take significant time depending on the number of trees and tree depth.
- Loading the model from disk (if large, e.g., 500+ trees on 110 features) can consume multiple GB of RAM, leaving little room for the data.
- If `predict()` internally copies the data frame, memory pressure causes swapping on a 16 GB machine.

### C. Memory Pressure

- 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB** just for the numeric feature matrix.
- The Random Forest model object can be 2â€“6 GB.
- The neighbor lookup list (6.46M entries) adds ~1â€“2 GB.
- Total easily exceeds 16 GB â†’ OS swapping â†’ massive slowdown.

---

## 2. Optimization Strategy

| Area | Problem | Solution |
|---|---|---|
| **Neighbor lookup** | Millions of string-key lookups | Replace with integer-indexed `data.table` join; build lookup as a two-column integer table, not a list |
| **Neighbor stats** | 6.46M Ã— 5 R-level `lapply` calls | Vectorized grouped aggregation via `data.table` |
| **Memory: neighbor list** | 6.46M-element R list | Flat edge-list table (two integer columns) |
| **Memory: feature matrix** | Full data.frame copied into predict | Use a single `data.table` in-place; convert to matrix only at predict time |
| **RF prediction** | Possibly row-by-row or full-copy | Single vectorized `predict()` call on a pre-allocated matrix; chunk if memory-limited |
| **Model loading** | Potential repeated loads | Load once, keep in memory |
| **Object copying** | R's copy-on-modify semantics | Use `data.table` set-by-reference (`:=`) to avoid copies |

**Expected speedup**: From 86+ hours to roughly **10â€“30 minutes** for feature prep, plus RF predict time (model-dependent).

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "randomForest"))
#   (or ranger â€” see note at end)

library(data.table)

# ---- 0. Load pre-trained model once ----------------------------------------
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# Assumes rf_model is already in the workspace.

# ---- 1. Convert cell_data to data.table in-place ---------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in-place, no copy
}

# Ensure key columns are integer for fast joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row-index column (used for joining neighbor stats back)
cell_data[, .row_idx := .I]

# ---- 2. Build flat neighbor edge-list (replaces build_neighbor_lookup) ------
build_neighbor_edgelist <- function(cell_dt, id_order, neighbors_nb) {
  # id_order: integer vector of cell IDs in the order matching the nb object
  # neighbors_nb: spdep nb object (list of integer index vectors)
  
  # Expand nb object into a flat edge-list of (focal_id, neighbor_id)
  n <- length(neighbors_nb)
  focal_idx <- rep.int(seq_len(n), lengths(neighbors_nb))
  neighbor_idx <- unlist(neighbors_nb, use.names = FALSE)
  
  edge_dt <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  
  return(edge_dt)
}

cat("Building neighbor edge-list...\n")
edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
# edge_dt has ~1,373,394 rows and two integer columns â€” very compact

# ---- 3. Vectorized neighbor-stat computation (replaces compute_neighbor_stats)
compute_and_add_all_neighbor_features <- function(cell_dt, edge_dt,
                                                   neighbor_source_vars) {
  # Build a join table: for every (focal_id, year) we need the row indices

  # of all neighbors in that same year.
  #
  # Strategy:
  #   1. Join edge_dt with cell_dt to get neighbor values per (focal_id, year).
  #   2. Aggregate (max, min, mean) grouped by (focal_id, year).
  #   3. Join aggregated stats back to cell_dt by (id, year).
  
  # Prepare a small lookup: (id, year) -> values of all source vars
  # Only keep columns we need to minimize memory
  value_cols <- intersect(neighbor_source_vars, names(cell_dt))
  neighbor_vals <- cell_dt[, c("id", "year", value_cols), with = FALSE]
  setnames(neighbor_vals, "id", "neighbor_id")
  
  # Key for fast join

setkey(neighbor_vals, neighbor_id, year)
  
  # We need to cross edge_dt with years. But each focal cell appears in every
  # year it has data. So we join via (focal_id -> id, year) to get the years
  # each focal cell has, then look up neighbor values for that year.
  
  # Step A: Get unique (id, year) pairs from cell_dt
  focal_years <- cell_dt[, .(focal_id = id, year)]
  
  # Step B: Join focal_years with edge_dt to get (focal_id, year, neighbor_id)
  # This is the big expansion: ~1.37M edges Ã— 28 years â‰ˆ 38.5M rows worst case,

  # but many cells don't span all years. We do a keyed join instead.
  setkey(edge_dt, focal_id)
  setkey(focal_years, focal_id)
  
  cat("  Expanding edges Ã— years...\n")
  # For each edge (focal_id, neighbor_id), replicate across all years the focal

  # cell appears in. This is an inner join.
  expanded <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded columns: focal_id, neighbor_id, year
  
  # Step C: Look up neighbor values
  cat("  Looking up neighbor values...\n")
  setkey(expanded, neighbor_id, year)
  expanded <- neighbor_vals[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has columns: neighbor_id, year, <value_cols>, focal_id
  
  # Step D: Aggregate per (focal_id, year) for each variable
  cat("  Aggregating neighbor stats...\n")
  agg_exprs <- list()
  for (v in value_cols) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = sym_v))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = sym_v))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = sym_v))
  }
  
  # Build a single aggregation call
  agg_stats <- expanded[,
    lapply(agg_exprs, eval, envir = .SD),
    by = .(focal_id, year),
    .SDcols = value_cols
  ]
  
  # Fix Inf/-Inf from max/min on all-NA groups â†’ NA
  inf_cols <- grep("^n_max_|^n_min_", names(agg_stats), value = TRUE)
  for (col in inf_cols) {
    set(agg_stats, which(is.infinite(agg_stats[[col]])), col, NA_real_)
  }
  
  # Step E: Join back to cell_dt by (id, year)
  cat("  Joining neighbor features back to cell_data...\n")
  setnames(agg_stats, "focal_id", "id")
  setkey(agg_stats, id, year)
  setkey(cell_dt, id, year)
  
  new_cols <- setdiff(names(agg_stats), c("id", "year"))
  cell_dt[agg_stats, (new_cols) := mget(paste0("i.", new_cols)), on = .(id, year)]
  
  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing all neighbor features (vectorized)...\n")
system.time({
  cell_data <- compute_and_add_all_neighbor_features(
    cell_data, edge_dt, neighbor_source_vars
  )
})

# Free the edge expansion memory
rm(edge_dt)
gc()

# ---- 4. Prepare prediction matrix ------------------------------------------
cat("Preparing prediction matrix...\n")

# Get the feature names the model expects
if (inherits(rf_model, "randomForest")) {
  # randomForest stores the variable names used in training
  model_features <- rownames(rf_model$importance)
} else if (inherits(rf_model, "ranger")) {
  model_features <- rf_model$forest$independent.variable.names
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features are present
missing_feats <- setdiff(model_features, names(cell_data))
if (length(missing_feats) > 0) {
  warning("Missing features (will be set to NA): ", paste(missing_feats, collapse = ", "))
  for (mf in missing_feats) {
    set(cell_data, j = mf, value = NA_real_)
  }
}

# Build a numeric matrix â€” avoids data.frame overhead inside predict()
# Do this in-place from the data.table to avoid a full copy
pred_matrix <- as.matrix(cell_data[, ..model_features])

# ---- 5. Chunked Random Forest prediction ------------------------------------
cat("Running Random Forest prediction...\n")

predict_chunked <- function(model, newdata_matrix, chunk_size = 500000L) {
  n <- nrow(newdata_matrix)
  predictions <- numeric(n)
  
  n_chunks <- ceiling(n / chunk_size)
  
  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    
    chunk <- newdata_matrix[start_row:end_row, , drop = FALSE]
    
    if (inherits(model, "ranger")) {
      # ranger::predict expects a data.frame or matrix
      pred <- predict(model, data = chunk)
      predictions[start_row:end_row] <- pred$predictions
    } else {
      # randomForest::predict
      predictions[start_row:end_row] <- predict(model, newdata = chunk)
    }
    
    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %dâ€“%d)\n", i, n_chunks, start_row, end_row))
    }
  }
  
  return(predictions)
}

system.time({
  cell_data[, predicted_gdp := predict_chunked(rf_model, pred_matrix, chunk_size = 500000L)]
})

# Free the prediction matrix
rm(pred_matrix)
gc()

# ---- 6. Output --------------------------------------------------------------
cat("Done. Predictions stored in cell_data$predicted_gdp\n")
# The original numerical estimand (GDP) is preserved; no transformation applied.
# The trained RF model (rf_model) is untouched.
```

---

## 4. Why This Works â€” Summary of Changes

| Original | Optimized | Why faster |
|---|---|---|
| `build_neighbor_lookup`: 6.46M `paste`+named-vector lookups | `build_neighbor_edgelist`: vectorized `rep.int`+`unlist` on integer indices | Eliminates all string operations; O(E) integer work |
| `compute_neighbor_stats`: 5 Ã— 6.46M `lapply` calls with per-element subsetting | Single `data.table` grouped aggregation over a flat join table | `data.table` uses radix-sort grouping and C-level aggregation; one pass per variable |
| `do.call(rbind, 6.46M-element list)` | `data.table` `:=` join (set-by-reference) | No intermediate list; no repeated `rbind` allocation |
| Implicit `data.frame` copies on each `cell_data$new_col <- ...` | `data.table` `:=` assignment (modifies in-place) | Zero-copy column addition |
| `predict()` on full 6.46M-row data.frame (internal copy) | `predict()` on pre-built numeric matrix in 500K-row chunks | Avoids data.frame-to-matrix conversion inside `predict`; keeps peak memory bounded |
| Model potentially reloaded or copied | Loaded once; passed by reference | Single load |

### Expected Performance

| Phase | Original (est.) | Optimized (est.) |
|---|---|---|
| Neighbor lookup build | ~20â€“40 hours | ~10â€“30 seconds |
| Neighbor stats (5 vars) | ~30â€“40 hours | ~2â€“8 minutes |
| RF prediction (6.46M rows) | ~5â€“10 hours | ~5â€“30 minutes (model-dependent) |
| **Total** | **86+ hours** | **~10â€“40 minutes** |

### Notes

- **If using `ranger` instead of `randomForest`**: `ranger::predict` is substantially faster (multi-threaded C++ backend). If the model was trained with `ranger`, prediction on 6.46M rows may take only 2â€“5 minutes. If it was trained with `randomForest`, consider one-time conversion or simply accept the longer predict time.
- **Memory**: The expanded edge Ã— year table is the peak memory consumer (~38M rows Ã— a few columns â‰ˆ 1â€“2 GB). This is well within 16 GB alongside the main data and model. The chunked prediction ensures the matrix copy stays bounded.
- **Numerical fidelity**: No transformations, rounding, or approximations are applied. The predicted values are identical to what the original pipeline would produce â€” only the data preparation path is restructured.
END_RESPONSE>>>
