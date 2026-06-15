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
run_id: Anthropic_F03_T10_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

The pipeline has **two major bottlenecks**: (1) feature preparation (neighbor lookup + neighbor statistics), and (2) Random Forest inference over ~6.46 million rows Ã— ~110 features. Let's break them down:

### 1. `build_neighbor_lookup` â€” O(n) `paste`/hash lookups, but with costly R-level `lapply`

- **6.46 million iterations** of an R-level `lapply`, each performing `paste()`, character key lookups in a named vector (`idx_lookup`), and `as.integer` coercion.
- Named-vector lookups in R use **linear hashing** but the overhead per call is high when repeated millions of times.
- The `paste(id, year)` key construction is repeated redundantly â€” once to build `idx_lookup`, then again inside every iteration for each neighbor.

### 2. `compute_neighbor_stats` â€” repeated subsetting and allocation

- Another 6.46M-iteration `lapply`, each extracting a variable-length subset, removing NAs, computing max/min/mean.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is a known **performance disaster** â€” it incrementally allocates and copies a growing matrix.
- This is called **5 times** (once per neighbor source variable), so the total is ~32.3 million R-level function calls.

### 3. Random Forest Inference

- `predict.randomForest` (or `predict.ranger`) on 6.46M rows is memory-intensive. If using the `randomForest` package, prediction is **single-threaded** and slow; the internal C code loops over trees sequentially.
- If the model is a `randomForest` object, `predict()` copies the entire dataset into a matrix internally â€” for 6.46M Ã— 110, that's ~5.7 GB in double precision **alone**, likely exceeding 16 GB RAM with overhead.
- Garbage collection pressure from repeated copies is enormous.

### 4. Memory pressure

- Base data: 6.46M Ã— 110 doubles â‰ˆ 5.7 GB.
- Neighbor lookup list: 6.46M elements, each an integer vector â†’ ~1â€“2 GB.
- Intermediate copies from `data.frame` column additions (`cell_data <- compute_and_add_neighbor_features(...)`) trigger full-frame copy-on-modify each time.
- RF prediction internal copies can push total well past 16 GB â†’ swapping to disk â†’ **86+ hours**.

---

## Optimization Strategy

| Bottleneck | Solution | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` via `paste`+named vector | Replace with `data.table` integer join; pre-build a (id, year) â†’ row mapping | ~50â€“100Ã— |
| `compute_neighbor_stats` via R-level `lapply` + `do.call(rbind,...)` | Vectorized computation using `data.table` unnest + grouped aggregation, or C++ via `Rcpp` | ~100â€“500Ã— |
| Column-addition triggers copy-on-modify | Use `data.table` set-by-reference (`set()` or `:=`) | eliminates copies |
| RF prediction on 6.46M rows at once | **Chunk-wise prediction** (e.g., 500K rows) to stay in RAM; convert `randomForest` model to `ranger` format if feasible, or use `predict` in chunks | ~2â€“10Ã— (avoids swap) |
| Single-threaded `randomForest::predict` | If model is `ranger`, use `num.threads`; if `randomForest`, chunk + `future.apply` parallel | ~4â€“8Ã— on 4+ cores |

### Key Principles

1. **Replace R-level loops with vectorized `data.table` grouped operations.**
2. **Never copy the main data frame; use `:=` assignment throughout.**
3. **Chunk RF prediction to control peak memory.**
4. **Parallelize where possible.**

---

## Working R Code

```r
# ============================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# ============================================================
# Requirements: data.table, ranger (or randomForest), future.apply

library(data.table)
library(future.apply)

# --------------------------------------------------
# 0. Setup parallel backend (use all but 1 core)
# --------------------------------------------------
plan(multisession, workers = max(1L, parallel::detectCores() - 1L))

# --------------------------------------------------
# 1. Convert main data to data.table (in-place if possible)
# --------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in-place, no copy
}

# Ensure key columns are proper types
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create an integer row-index column (will be used for joining)
cell_data[, .row_idx := .I]

# Set key for fast joins
setkey(cell_data, id, year)


# --------------------------------------------------
# 2. Build neighbor edge-list as a data.table (replaces build_neighbor_lookup)
# --------------------------------------------------
build_neighbor_edgelist_dt <- function(cell_data, id_order, neighbors) {
  # cell_data must be a keyed data.table with columns id, year, .row_idx
  # id_order: integer vector of cell IDs in the order matching `neighbors`
  # neighbors: spdep nb object (list of integer index vectors into id_order)

  id_order <- as.integer(id_order)

  # Map each cell ID to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Unique cell IDs present in data
  unique_ids_in_data <- unique(cell_data$id)

  # Build directed edge list: focal_id -> neighbor_id
  # Only for IDs actually present in the data
  edge_list <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    focal_id <- id_order[ref_idx]
    nb_indices <- neighbors[[ref_idx]]
    if (length(nb_indices) == 0L) return(NULL)
    nb_ids <- id_order[nb_indices]
    data.table(focal_id = focal_id, neighbor_id = nb_ids)
  }))

  # Keep only edges where focal_id is actually in our data
  edge_list <- edge_list[focal_id %in% unique_ids_in_data]

  return(edge_list)
}

cat("Building neighbor edge list...\n")
system.time({
  edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)
})
# edge_dt has columns: focal_id, neighbor_id
cat(sprintf("Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))


# --------------------------------------------------
# 3. Vectorized neighbor statistics (replaces compute_neighbor_stats)
# --------------------------------------------------
compute_and_add_all_neighbor_features_dt <- function(cell_data, edge_dt,
                                                      neighbor_source_vars) {
  # Strategy:
  #   1. Build a long table: for every (focal_id, year) row, join in neighbor rows.
  #   2. Compute grouped max/min/mean per (focal_row, variable).
  #   3. Join results back to cell_data by reference.
  #
  # To manage memory, we process one variable at a time but do the join once.

  # Minimal lookup table: (id, year) -> .row_idx + variable values
  cols_needed <- c("id", "year", ".row_idx", neighbor_source_vars)
  lookup <- cell_data[, ..cols_needed]
  setnames(lookup, "id", "neighbor_id")
  setkey(lookup, neighbor_id, year)

  # Focal side: for each row, get its id and year
  focal_info <- cell_data[, .(focal_row = .row_idx, focal_id = id, year = year)]

  # Join focal -> edge list to get (focal_row, year, neighbor_id) triples
  # This is the "expand" step
  cat("  Joining focal rows to edge list...\n")
  expanded <- merge(focal_info, edge_dt, by = "focal_id", allow.cartesian = TRUE)
  # expanded has: focal_row, focal_id, year, neighbor_id

  # Now join in neighbor values
  cat("  Joining neighbor values...\n")
  setkey(expanded, neighbor_id, year)
  expanded <- lookup[expanded, on = .(neighbor_id, year), nomatch = NA]
  # Now expanded has: neighbor_id, year, .row_idx (neighbor's row), 
  #                   <var columns>, focal_row, focal_id

  # Compute stats per variable, grouped by focal_row
  cat("  Computing grouped statistics...\n")
  for (var_name in neighbor_source_vars) {
    cat(sprintf("    Processing: %s\n", var_name))
    
    # Extract just what we need to reduce memory
    sub <- expanded[, .(focal_row, val = get(var_name))]
    
    # Remove NA values before aggregation
    sub <- sub[!is.na(val)]
    
    # Grouped aggregation â€” highly optimized in data.table
    stats <- sub[, .(
      nb_max  = max(val),
      nb_min  = min(val),
      nb_mean = mean(val)
    ), by = focal_row]
    
    # Define output column names (preserve original naming convention)
    max_col  <- paste0("nb_max_", var_name)
    min_col  <- paste0("nb_min_", var_name)
    mean_col <- paste0("nb_mean_", var_name)
    
    # Initialize columns with NA
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
    
    # Assign by reference using row indices
    set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)
    
    rm(sub, stats)
  }
  
  rm(expanded, lookup, focal_info)
  gc()
  
  invisible(cell_data)
}

cat("Computing all neighbor features (vectorized)...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

system.time({
  compute_and_add_all_neighbor_features_dt(
    cell_data, edge_dt, neighbor_source_vars
  )
})

# Clean up edge list
rm(edge_dt)
gc()


# --------------------------------------------------
# 4. Prepare prediction matrix ONCE (avoid repeated conversion)
# --------------------------------------------------
cat("Preparing prediction matrix...\n")

# Identify the exact feature columns the model expects
# Adapt this to your model object:
if (inherits(rf_model, "ranger")) {
  # ranger stores feature names in the model object
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest: use the names from the training data
  # These are typically stored or known; adjust as needed
  feature_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features are present
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# Build the prediction matrix ONCE as a plain matrix (most memory-efficient)
# data.table's as.matrix on subset columns is efficient
pred_matrix <- as.matrix(cell_data[, ..feature_names])
cat(sprintf("Prediction matrix: %s rows Ã— %s cols (%.1f GB)\n",
            format(nrow(pred_matrix), big.mark = ","),
            ncol(pred_matrix),
            object.size(pred_matrix) / 1e9))


# --------------------------------------------------
# 5. Chunked + parallelized Random Forest prediction
# --------------------------------------------------
predict_rf_chunked <- function(model, newdata_matrix, chunk_size = 500000L,
                                parallel = TRUE) {
  # newdata_matrix: numeric matrix with named columns matching model features
  # Returns: numeric vector of predictions (same length as nrow(newdata_matrix))
  
  n <- nrow(newdata_matrix)
  n_chunks <- ceiling(n / chunk_size)
  chunk_starts <- seq(1L, n, by = chunk_size)
  chunk_ends   <- pmin(chunk_starts + chunk_size - 1L, n)
  
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))
  
  is_ranger <- inherits(model, "ranger")
  
  predict_one_chunk <- function(idx) {
    start_i <- chunk_starts[idx]
    end_i   <- chunk_ends[idx]
    chunk   <- newdata_matrix[start_i:end_i, , drop = FALSE]
    
    if (is_ranger) {
      # ranger: supports num.threads for parallel tree evaluation
      pred <- predict(model, data = chunk, num.threads = 1L)
      return(pred$predictions)
    } else {
      # randomForest
      chunk_df <- as.data.frame(chunk)
      pred <- predict(model, newdata = chunk_df)
      return(as.numeric(pred))
    }
  }
  
  if (parallel && n_chunks > 1L) {
    cat("  Using parallel chunk prediction...\n")
    results <- future_lapply(seq_len(n_chunks), predict_one_chunk,
                             future.seed = NULL,
                             future.chunk.size = 1L)
  } else {
    results <- lapply(seq_len(n_chunks), function(idx) {
      if (idx %% 5 == 0 || idx == n_chunks) {
        cat(sprintf("  Chunk %d / %d\n", idx, n_chunks))
      }
      predict_one_chunk(idx)
    })
  }
  
  # Combine
  predictions <- unlist(results, use.names = FALSE)
  stopifnot(length(predictions) == n)
  
  return(predictions)
}

# --- Choose chunk size based on available RAM ---
# Each chunk of 500K rows Ã— 110 cols â‰ˆ 420 MB as doubles
# With 16 GB RAM and ~6 GB for base data, we can afford ~500K comfortably
CHUNK_SIZE <- 500000L

cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model        = rf_model,
    newdata_matrix = pred_matrix,
    chunk_size   = CHUNK_SIZE,
    parallel     = TRUE
  )]
})

# Free the prediction matrix
rm(pred_matrix)
gc()


# --------------------------------------------------
# 6. (Optional) If model is randomForest, convert to ranger for speed
# --------------------------------------------------
# If you can do a one-time format conversion (not retraining), ranger
# is dramatically faster at prediction. However, there is no lossless
# automatic conversion. If the model is randomForest and prediction is
# still too slow, the chunked + parallel approach above is the best
# option without retraining.
#
# If the model IS already ranger, ensure you pass:
#   predict(model, data = ..., num.threads = parallel::detectCores() - 1)
# inside each chunk for maximum speed. Adjust predict_one_chunk above:
#   num.threads = parallel::detectCores() - 1L   (instead of 1L)
# when NOT using future_lapply parallelism (to avoid oversubscription).


# --------------------------------------------------
# 7. Verification: confirm numerical identity of estimand
# --------------------------------------------------
# Spot-check: predict a small sample with original method and compare
if (FALSE) {  # set to TRUE for verification
  set.seed(42)
  sample_idx <- sample(nrow(cell_data), 100)
  sample_df  <- as.data.frame(cell_data[sample_idx, ..feature_names])
  
  if (inherits(rf_model, "ranger")) {
    ref_preds <- predict(rf_model, data = sample_df)$predictions
  } else {
    ref_preds <- as.numeric(predict(rf_model, newdata = sample_df))
  }
  
  opt_preds <- cell_data$predicted_gdp[sample_idx]
  
  cat(sprintf("Max absolute difference: %e\n", max(abs(ref_preds - opt_preds))))
  cat(sprintf("All equal (tolerance 1e-10): %s\n",
              all(abs(ref_preds - opt_preds) < 1e-10)))
}

cat("Pipeline complete.\n")
```

---

## Summary of Expected Performance Gains

| Stage | Original | Optimized | Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~30â€“60 min (R-level lapply, 6.46M iters, paste/hash) | ~10â€“30 sec (vectorized `data.table` merge) | **~100Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | ~40â€“60+ hours (lapply + `do.call(rbind,...)`) | ~2â€“5 min (`data.table` grouped aggregation) | **~500Ã—** |
| Column addition / copy-on-modify | Repeated full-frame copies (~5+ GB each) | `set()` by reference â€” zero copies | **eliminates OOM** |
| RF Prediction (6.46M rows) | Single-shot, likely OOM / swap thrashing | Chunked (500K), parallel, pre-built matrix | **~5â€“10Ã—** |
| **Total** | **86+ hours** | **~15â€“45 minutes** | **~100â€“300Ã—** |

The trained Random Forest model is never modified. The numerical predictions are **bit-identical** to the original `predict()` output â€” we've only changed how data is prepared and how prediction is batched, not what is computed.
END_RESPONSE>>>
