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
run_id: Anthropic_F03_T10_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** â€” O(n) `lapply` over ~6.46 million rows, each performing:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`) â€” hash lookup per row.
- `paste()` to build neighbor keys â€” string allocation per neighbor per row.
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M) â€” this is **O(k)** per neighbor via partial-match hashing but with enormous constant overhead because R's named vector lookup on a 6.46M-element vector is essentially a repeated hash-table probe with string allocation.
- Net effect: ~6.46M iterations Ã— ~4 neighbors avg = ~25.8M string constructions + hash lookups. Estimated wall time: **30â€“90 minutes**.

**`compute_neighbor_stats`** â€” For each of 5 variables:
- `lapply` over 6.46M rows, subsetting a numeric vector by integer index, removing NAs, computing max/min/mean.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors â€” this alone is an **O(nÂ²)** memory-copy disaster because `rbind` on a list of vectors builds the matrix incrementally.
- Net effect per variable: **minutes to tens of minutes**, dominated by `do.call(rbind, ...)`.

**Outer loop** â€” Runs `compute_neighbor_stats` (and presumably column-binds results) 5 times â†’ the `do.call(rbind, ...)` penalty is paid 5 times.

### B. Prediction Bottlenecks

With a trained Random Forest and 6.46M rows Ã— 110 features:
- **Model loading**: `readRDS()` on a large `ranger`/`randomForest` object can take minutes and consume multiple GB.
- **Single `predict()` call on 6.46M rows**: If using `randomForest`, prediction is single-threaded and stores an internal copy of the data as a matrix â€” peak RAM â‰ˆ 6.46M Ã— 110 Ã— 8 bytes Ã— 2 copies â‰ˆ **11.4 GB** just for the data, likely exceeding 16 GB with model overhead.
- **If looped row-by-row or in small batches**: R function-call overhead dominates; millions of iterations is fatal.
- **Object copying**: R's copy-on-modify semantics mean any `data$new_col <- ...` inside a loop triggers a full data.frame copy (~5.7 GB).

### C. Summary of Root Causes

| Rank | Bottleneck | Severity |
|------|-----------|----------|
| 1 | `do.call(rbind, list_of_vectors)` in `compute_neighbor_stats` | Critical â€” quasi-quadratic |
| 2 | String-key lookup in `build_neighbor_lookup` (6.46M Ã— k) | High |
| 3 | `predict()` on full 6.46M rows at once (RAM) or row-by-row (overhead) | High |
| 4 | Repeated `data.frame` column assignment triggering copies | Moderate |
| 5 | Single-threaded prediction (`randomForest` package) | Moderate |

---

## 2. OPTIMIZATION STRATEGY

### Feature Preparation
1. **Replace `build_neighbor_lookup`** with a pure integer-index approach using `data.table` â€” build a `(cell_id, year) â†’ row_index` hash table via `data.table` keyed join; vectorize neighbor expansion.
2. **Replace `compute_neighbor_stats`** â€” pre-allocate a matrix, use vectorized grouped operations via `data.table` instead of `lapply` + `do.call(rbind, ...)`.
3. **Avoid column-by-column data.frame mutation** â€” work entirely in `data.table` (in-place `:=` assignment, no copies).

### Prediction
4. **Chunk-based prediction** â€” split 6.46M rows into ~500K-row chunks to control peak RAM while avoiding per-row overhead.
5. **Use `ranger` for prediction if possible** â€” `ranger::predict` is multithreaded C++ and 5â€“20Ã— faster than `randomForest::predict`. If the model is a `randomForest` object, we can still chunk it. If it's `ranger`, we enable `num.threads`.
6. **Load model once, predict in chunks, `gc()` between chunks**.

### Expected Speedup
- Feature preparation: from hours â†’ **2â€“10 minutes**.
- Prediction: from hours â†’ **10â€“45 minutes** (depending on model type and tree count).
- Total: from **86+ hours â†’ under 1 hour**.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Dependencies
library(data.table)

# ---- Step 0: Convert to data.table (once, in-place) -------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist as expected
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Add a row index column (used for neighbor mapping)
cell_data[, .row_idx := .I]

# =============================================================================
# STEP 1: BUILD NEIGHBOR LOOKUP (vectorized, integer-only)
# =============================================================================
build_neighbor_lookup_fast <- function(dt, id_order, neighbors_nb) {
  # dt         : data.table with columns 'id', 'year', '.row_idx'
  # id_order   : integer vector of cell IDs in the order used by the nb object
  # neighbors_nb: spdep nb object (list of integer index vectors)
  
  # --- 1a. Build directed edge list from nb object ---------------------------
  #     Each element neighbors_nb[[i]] is an integer vector of neighbor indices
  #     into id_order. We expand into a two-column edge table of cell IDs.
  n_cells <- length(id_order)
  from_idx <- rep(seq_len(n_cells), lengths(neighbors_nb))
  to_idx   <- unlist(neighbors_nb, use.names = FALSE)
  
  # Remove the 0-valued entries that spdep uses for cells with no neighbors
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]
  
  edges <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx, valid)
  
  # --- 1b. Build (id, year) -> row_idx lookup via keyed join -----------------
  idx_table <- dt[, .(id, year, .row_idx)]
  setkey(idx_table, id, year)
  
  # --- 1c. For each row, find its neighbors' row indices ---------------------
  #     Strategy: join cell_data rows with edges on id = from_id,
  #     then join back to idx_table to get neighbor row indices.
  
  # Get (from_id, year, source_row_idx)
  source <- dt[, .(from_id = id, year, src_row = .row_idx)]
  
  # Join: for each source row, expand to all neighbor cell IDs
  # source Ã— edges on from_id
  setkey(source, from_id)
  setkey(edges, from_id)
  expanded <- edges[source, on = "from_id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded now has columns: from_id, to_id, year, src_row
  
  # Join: map (to_id, year) -> neighbor row index
  setnames(idx_table, c("id", "year", ".row_idx"), c("to_id", "year", "nbr_row"))
  setkey(idx_table, to_id, year)
  setkey(expanded, to_id, year)
  expanded <- idx_table[expanded, on = c("to_id", "year"), nomatch = NA_integer_]
  # expanded now has: to_id, year, nbr_row, from_id, src_row
  
  # Drop rows where the neighbor wasn't found in the data
  expanded <- expanded[!is.na(nbr_row)]
  
  # Sort by src_row for efficient grouped operations later
  setkey(expanded, src_row)
  
  return(expanded)
  # Result columns: src_row (row in dt), nbr_row (neighbor's row in dt), year, to_id, from_id
}

cat("Building neighbor lookup...\n")
system.time({
  neighbor_edges <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})
# Expected: ~20-60 seconds

# =============================================================================
# STEP 2: COMPUTE NEIGHBOR STATS (vectorized via data.table grouped ops)
# =============================================================================
compute_and_add_all_neighbor_features <- function(dt, neighbor_edges, var_names) {
  # dt             : main data.table (with .row_idx)
  # neighbor_edges : data.table with (src_row, nbr_row) from Step 1
  # var_names      : character vector of variable names to compute neighbor stats for
  
  n_rows <- nrow(dt)
  
  for (vname in var_names) {
    cat("  Processing neighbor stats for:", vname, "\n")
    
    # Extract the variable values for all neighbor rows
    # (vectorized indexing â€” one shot)
    neighbor_edges[, nbr_val := dt[[vname]][nbr_row]]
    
    # Grouped aggregation: max, min, mean per source row, excluding NAs
    stats <- neighbor_edges[!is.na(nbr_val),
                            .(nbr_max  = max(nbr_val),
                              nbr_min  = min(nbr_val),
                              nbr_mean = mean(nbr_val)),
                            keyby = src_row]
    
    # Create full-length result vectors (default NA for rows with no valid neighbors)
    col_max  <- rep(NA_real_, n_rows)
    col_min  <- rep(NA_real_, n_rows)
    col_mean <- rep(NA_real_, n_rows)
    
    col_max[stats$src_row]  <- stats$nbr_max
    col_min[stats$src_row]  <- stats$nbr_min
    col_mean[stats$src_row] <- stats$nbr_mean
    
    # In-place assignment (no copy triggered)
    max_name  <- paste0(vname, "_max")
    min_name  <- paste0(vname, "_min")
    mean_name <- paste0(vname, "_mean")
    
    set(dt, j = max_name,  value = col_max)
    set(dt, j = min_name,  value = col_min)
    set(dt, j = mean_name, value = col_mean)
    
    rm(stats, col_max, col_min, col_mean)
  }
  
  # Clean up temporary column
  neighbor_edges[, nbr_val := NULL]
  
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, neighbor_edges, neighbor_source_vars)
})
# Expected: ~1-5 minutes total for all 5 variables

# Free the edge table
rm(neighbor_edges)
gc()

# =============================================================================
# STEP 3: RANDOM FOREST PREDICTION (chunked, memory-safe)
# =============================================================================
predict_chunked <- function(model, dt, feature_names, chunk_size = 500000L) {
  # model         : pre-trained RF model (randomForest or ranger object)
  # dt            : data.table containing all feature columns
  # feature_names : character vector of the ~110 predictor column names
  # chunk_size    : rows per prediction chunk (tune to RAM; 500K â‰ˆ 440 MB per chunk)
  
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)
  
  # Detect model type
  is_ranger <- inherits(model, "ranger")
  
  cat(sprintf("Predicting %d rows in %d chunks of up to %d...\n", n, n_chunks, chunk_size))
  
  for (chunk_i in seq_len(n_chunks)) {
    start_row <- (chunk_i - 1L) * chunk_size + 1L
    end_row   <- min(chunk_i * chunk_size, n)
    
    # Extract chunk as a plain data.frame (most predict methods expect this)
    chunk_dt <- dt[start_row:end_row, ..feature_names]
    
    if (is_ranger) {
      # ranger: multithreaded prediction
      pred <- predict(model, data = chunk_dt, num.threads = parallel::detectCores())$predictions
    } else {
      # randomForest or other
      pred <- predict(model, newdata = chunk_dt)
    }
    
    predictions[start_row:end_row] <- pred
    
    rm(chunk_dt, pred)
    if (chunk_i %% 3 == 0) gc()  # periodic gc every 3 chunks
    
    if (chunk_i %% 5 == 0 || chunk_i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %d-%d)\n", chunk_i, n_chunks, start_row, end_row))
    }
  }
  
  return(predictions)
}

# ---- Load model once --------------------------------------------------------
cat("Loading trained RF model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")  # <-- adjust path

# ---- Get feature names (must match training) --------------------------------
# Option A: if stored with the model
if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores the names used at training time
  feature_names <- rownames(rf_model$importance)
} else {
  # Fallback: specify manually
  # feature_names <- c("ntl", "ec", ..., "pop_density_mean")
  stop("Unknown model class: ", class(rf_model), ". Please specify feature_names manually.")
}

# Verify all features exist in cell_data
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# ---- Predict ----------------------------------------------------------------
cat("Running predictions...\n")
system.time({
  cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, feature_names, chunk_size = 500000L)]
})

# ---- Cleanup ----------------------------------------------------------------
cell_data[, .row_idx := NULL]  # remove helper column
gc()

cat("Done. Predictions stored in cell_data$predicted_gdp\n")
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with string key hashing | Vectorized `data.table` keyed join on edge table | ~50â€“200Ã— |
| **Neighbor stats** | `lapply` + `do.call(rbind, 6.46M-element list)` Ã— 5 vars | `data.table` grouped aggregation (`:=`, `set()`) | ~100â€“500Ã— |
| **Column assignment** | `data.frame$col <- ...` (copy-on-modify, ~5.7 GB copies) | `data.table::set()` (in-place, zero-copy) | Eliminates ~5 full copies |
| **Prediction** | Single call on 6.46M rows (OOM risk) or row-by-row loop | 500K-row chunks; auto-detects `ranger` multithreading | Fits in 16 GB RAM; ~2â€“10Ã— if `ranger` |
| **Overall estimated time** | 86+ hours | **30â€“60 minutes** | ~100Ã— |

The trained Random Forest model is loaded read-only and never modified. The numerical predictions are identical because the same model, same features, and same computation (max, min, mean of neighbors) are preserved â€” only the implementation mechanics change.
END_RESPONSE>>>
