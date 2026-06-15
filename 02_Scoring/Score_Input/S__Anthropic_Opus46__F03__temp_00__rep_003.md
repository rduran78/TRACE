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
run_id: Anthropic_F03_T00_R003
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 3
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_003.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup (`idx_lookup[neighbor_keys]`)
- NA filtering

Named-vector lookups in R are **O(n)** hash probes per call, but the sheer volume (~6.46M iterations Ã— ~4 neighbors on average) and repeated `paste`/`as.character` allocations make this extremely slow. The result is a **list of 6.46M integer vectors** consuming substantial memory.

**`compute_neighbor_stats`:** Called 5 times (once per variable). Each call iterates over the 6.46M-element neighbor lookup list, subsets a numeric vector, removes NAs, and computes max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix â€” this alone is a notorious R anti-pattern (millions of small allocations + a single massive `rbind`).

**Outer loop:** Calls `compute_and_add_neighbor_features` 5 times, each presumably calling `compute_neighbor_stats` and column-binding results into `cell_data`. If `cell_data` is a `data.frame`, each column addition triggers a full copy (~6.46M Ã— 110+ columns).

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, calling `predict()` on a Random Forest in one shot can:
- Require the entire prediction matrix in memory simultaneously alongside the model object (which for ranger/randomForest can be hundreds of MB to several GB).
- On a 16 GB laptop, this risks swapping or OOM.
- If using `randomForest::predict.randomForest`, the implementation copies the data internally and is single-threaded.

### 1.3 Memory Bottleneck

- `cell_data` at 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ **5.7 GB** as a numeric matrix.
- The neighbor lookup list (6.46M elements, each a small integer vector) adds ~1â€“2 GB with R's per-object overhead.
- The RF model + prediction workspace can add 2â€“4 GB.
- Total easily exceeds 16 GB.

### 1.4 Root Cause Summary

| Component | Problem | Severity |
|---|---|---|
| `build_neighbor_lookup` | Per-row `paste`/named-vector lookup over 6.46M rows | High |
| `compute_neighbor_stats` | `lapply` + `do.call(rbind, ...)` over 6.46M elements, called 5Ã— | **Critical** |
| Column binding to data.frame | Full-copy semantics on each assignment | High |
| Prediction | Possibly single-threaded, single-batch, memory-heavy | High |
| Overall memory | ~12â€“16 GB working set on 16 GB machine | High |

---

## 2. OPTIMIZATION STRATEGY

### Strategy A: Vectorized Neighbor Stats via Sparse Matrix Multiplication

Instead of looping over 6.46M rows, represent the neighbor graph as a **sparse row-normalized matrix** and compute neighbor means via matrix-vector multiplication. Max and min can be computed via grouped operations on a long-format edge table using `data.table`.

This replaces 6.46M R-level iterations with a handful of vectorized C-level operations.

### Strategy B: Use `data.table` Throughout

- Convert `cell_data` to `data.table` for in-place column addition (no copy).
- Build the neighbor lookup as a two-column `data.table` (row_idx, neighbor_row_idx) â€” a flat edge list â€” and use grouped aggregation.

### Strategy C: Chunked Prediction with `ranger`

- If the model is a `ranger` object, `predict()` is already multi-threaded. Predict in chunks of ~500K rows to control peak memory.
- If the model is a `randomForest` object, convert the prediction matrix to a plain `matrix` (not data.frame) and predict in chunks.

### Strategy D: Eliminate the Lookup List Entirely

Replace the 6.46M-element R list with a flat edge `data.table` that maps each `(row_idx) â†’ (neighbor_row_idx)`. All neighbor stats become simple grouped aggregations.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, Matrix, ranger (or randomForest)
# Preserves: trained RF model, original numerical estimand
# =============================================================================

library(data.table)
library(Matrix)

# ---- STEP 0: Convert cell_data to data.table (if not already) ---------------
# Assume cell_data is a data.frame/data.table with columns: id, year, + predictors
# Assume id_order is a vector of unique cell IDs in the order matching rook_neighbors_unique
# Assume rook_neighbors_unique is an nb object (list of integer index vectors)

optimize_pipeline <- function(cell_data, id_order, rook_neighbors_unique, rf_model,
                              neighbor_source_vars = c("ntl", "ec", "pop_density",
                                                       "def", "usd_est_n2"),
                              predict_chunk_size = 500000L) {

  cat("Converting to data.table...\n")
  if (!is.data.table(cell_data)) {
    setDT(cell_data)
  }

  n_rows <- nrow(cell_data)
  n_cells <- length(id_order)
  cat(sprintf("Rows: %s | Cells: %s | Years: %s\n",
              format(n_rows, big.mark = ","),
              format(n_cells, big.mark = ","),
              cell_data[, uniqueN(year)]))

  # ---- STEP 1: Build flat edge list (row_idx -> neighbor_row_idx) ----------
  cat("Building row-level edge list...\n")

  # Map cell id -> position in id_order (1-indexed)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Map (id, year) -> row index in cell_data
  cell_data[, row_idx := .I]
  key_dt <- cell_data[, .(id, year, row_idx)]
  setkey(key_dt, id, year)

  # Build cell-level edge list from nb object
  # Each element rook_neighbors_unique[[i]] contains neighbor indices into id_order
  from_cell <- rep(seq_along(rook_neighbors_unique),
                   lengths(rook_neighbors_unique))
  to_cell   <- unlist(rook_neighbors_unique, use.names = FALSE)

  # Remove 0-neighbor entries (nb objects use 0L for no-neighbor cells)
  valid <- to_cell > 0L
  from_cell <- from_cell[valid]
  to_cell   <- to_cell[valid]

  cell_edges <- data.table(
    from_id = id_order[from_cell],
    to_id   = id_order[to_cell]
  )
  rm(from_cell, to_cell, valid)

  cat(sprintf("Cell-level edges: %s\n", format(nrow(cell_edges), big.mark = ",")))

  # Expand to row-level edges: for each year, map (from_id, year) and (to_id, year)
  # to their respective row indices
  # This is the key step: join cell_edges Ã— years to get row-level edges

  years <- sort(unique(cell_data$year))

  cat("Expanding cell edges to row-level edges across years...\n")

  # Cross join cell_edges with years
  # To avoid a massive cross join in memory, process in year chunks
  edge_list <- rbindlist(lapply(years, function(yr) {
    # Get row indices for this year
    yr_rows <- key_dt[year == yr]
    setkey(yr_rows, id)

    # Map from_id -> from_row_idx
    from_map <- yr_rows[.(cell_edges$from_id), .(from_row_idx = row_idx), nomatch = 0L]
    to_map   <- yr_rows[.(cell_edges$to_id),   .(to_row_idx   = row_idx), nomatch = 0L]

    # We need to keep alignment: only edges where both from and to exist in this year
    # Rebuild with matched indices
    matched <- yr_rows[.(cell_edges$from_id), .(from_row_idx = row_idx, to_id = cell_edges$to_id), nomatch = NA]
    matched <- matched[!is.na(from_row_idx)]

    # Now join to_id -> to_row_idx
    setkey(yr_rows, id)
    matched[, to_row_idx := yr_rows[.(to_id), row_idx, nomatch = NA]]
    matched <- matched[!is.na(to_row_idx), .(from_row_idx, to_row_idx)]

    matched
  }), use.names = TRUE)

  cat(sprintf("Row-level edges: %s\n", format(nrow(edge_list), big.mark = ",")))

  rm(cell_edges, key_dt)
  gc()

  # ---- STEP 2: Compute neighbor stats using grouped aggregation -------------
  cat("Computing neighbor features...\n")

  for (var_name in neighbor_source_vars) {
    cat(sprintf("  Processing: %s\n", var_name))

    vals <- cell_data[[var_name]]

    # Attach the neighbor's value to each edge
    edge_list[, nbr_val := vals[to_row_idx]]

    # Grouped aggregation: for each from_row_idx, compute max, min, mean
    # Exclude NAs
    stats <- edge_list[!is.na(nbr_val),
                       .(nbr_max  = max(nbr_val),
                         nbr_min  = min(nbr_val),
                         nbr_mean = mean(nbr_val)),
                       by = from_row_idx]

    # Assign back to cell_data by row index
    # Initialize with NA
    max_col  <- paste0(var_name, "_nbr_max")
    min_col  <- paste0(var_name, "_nbr_min")
    mean_col <- paste0(var_name, "_nbr_mean")

    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    set(cell_data, i = stats$from_row_idx, j = max_col,  value = stats$nbr_max)
    set(cell_data, i = stats$from_row_idx, j = min_col,  value = stats$nbr_min)
    set(cell_data, i = stats$from_row_idx, j = mean_col, value = stats$nbr_mean)

    rm(stats)
  }

  # Clean up temporary column from edge_list
  edge_list[, nbr_val := NULL]
  rm(edge_list)
  gc()

  # Remove helper column
  cell_data[, row_idx := NULL]

  cat("Neighbor features complete.\n")

  # ---- STEP 3: Chunked Random Forest Prediction -----------------------------
  cat("Starting chunked prediction...\n")

  # Identify predictor columns (exclude id, year, and the target if present)
  # The user should adjust this to match their model's expected features
  model_class <- class(rf_model)[1]

  if (model_class == "ranger") {
    # ranger stores variable names
    pred_vars <- rf_model$forest$independent.variable.names
  } else if (model_class == "randomForest") {
    # randomForest: use the variable names from the forest
    if (!is.null(rf_model$forest$xlevels)) {
      pred_vars <- names(rf_model$forest$xlevels)
    } else {
      # Fallback: try to get from terms or use all numeric columns minus id/year
      pred_vars <- attr(rf_model$terms, "term.labels")
      if (is.null(pred_vars)) {
        # Last resort: use rownames of importance
        pred_vars <- rownames(rf_model$importance)
      }
    }
  } else {
    stop("Unsupported model class: ", model_class)
  }

  # Verify all predictor variables exist
  missing_vars <- setdiff(pred_vars, names(cell_data))
  if (length(missing_vars) > 0) {
    warning("Missing predictor variables: ", paste(missing_vars, collapse = ", "))
  }

  # Pre-allocate prediction vector
  predictions <- numeric(n_rows)

  # Determine chunks
  n_chunks <- ceiling(n_rows / predict_chunk_size)
  cat(sprintf("Predicting in %d chunks of up to %s rows...\n",
              n_chunks, format(predict_chunk_size, big.mark = ",")))

  for (chunk_i in seq_len(n_chunks)) {
    start_idx <- (chunk_i - 1L) * predict_chunk_size + 1L
    end_idx   <- min(chunk_i * predict_chunk_size, n_rows)

    # Extract chunk as matrix for efficiency (avoids data.frame overhead)
    chunk_data <- cell_data[start_idx:end_idx, ..pred_vars]

    if (model_class == "ranger") {
      pred_result <- predict(rf_model, data = chunk_data,
                             num.threads = parallel::detectCores() - 1L)
      predictions[start_idx:end_idx] <- pred_result$predictions
    } else {
      # randomForest
      # Convert to matrix if all numeric (faster predict)
      chunk_mat <- as.matrix(chunk_data)
      predictions[start_idx:end_idx] <- predict(rf_model, newdata = chunk_mat)
    }

    if (chunk_i %% 5 == 0 || chunk_i == n_chunks) {
      cat(sprintf("  Chunk %d/%d complete (rows %s-%s)\n",
                  chunk_i, n_chunks,
                  format(start_idx, big.mark = ","),
                  format(end_idx, big.mark = ",")))
    }

    rm(chunk_data)
    if (exists("chunk_mat")) rm(chunk_mat)
    if (exists("pred_result")) rm(pred_result)
    if (chunk_i %% 10 == 0) gc()
  }

  # Attach predictions
  cell_data[, predicted_gdp := predictions]

  cat("Pipeline complete.\n")
  return(cell_data)
}


# =============================================================================
# USAGE
# =============================================================================
# # Load your objects
# cell_data              <- readRDS("cell_data.rds")
# id_order               <- readRDS("id_order.rds")
# rook_neighbors_unique  <- readRDS("rook_neighbors_unique.rds")
# rf_model               <- readRDS("rf_model.rds")
#
# result <- optimize_pipeline(
#   cell_data             = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model              = rf_model,
#   neighbor_source_vars  = c("ntl", "ec", "pop_density", "def", "usd_est_n2"),
#   predict_chunk_size    = 500000L
# )
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| **Neighbor lookup construction** | `lapply` over 6.46M rows with `paste`/named-vector lookups | `data.table` keyed joins, vectorized per-year expansion | **~50â€“100Ã—** |
| **Neighbor stats (per variable)** | `lapply` + `do.call(rbind, ...)` over 6.46M elements | `data.table` grouped aggregation (`by = from_row_idx`) | **~100â€“200Ã—** |
| **Column assignment** | `data.frame` copy-on-modify (copies ~5.7 GB per assignment) | `data.table::set()` â€” in-place, zero-copy | **eliminates ~15 full copies** |
| **Prediction** | Single batch (potential OOM / swap) | Chunked + multi-threaded (ranger) | **2â€“8Ã— + avoids OOM** |
| **Peak memory** | ~14â€“18 GB (exceeds 16 GB) | ~8â€“10 GB (fits in 16 GB) | **~40â€“50% reduction** |
| **Total estimated time** | 86+ hours | **~15â€“45 minutes** | **~100â€“300Ã—** |

### Key Principles Applied

1. **Replace R-level loops with vectorized `data.table` grouped operations** â€” the single largest win.
2. **Flat edge list instead of nested list** â€” eliminates R's per-object overhead for 6.46M list elements.
3. **`set()` for in-place column assignment** â€” eliminates copy-on-modify.
4. **Chunked prediction** â€” keeps peak memory within 16 GB budget.
5. **Multi-threaded prediction** (ranger) â€” uses all available cores.
6. **The trained RF model is never modified or retrained** â€” only `predict()` is called.
7. **The numerical estimand is preserved** â€” max, min, mean computations are identical; RF predictions are identical (same model, same features, same values).
END_RESPONSE>>>
