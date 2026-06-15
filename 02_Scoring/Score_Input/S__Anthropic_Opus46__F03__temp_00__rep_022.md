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
run_id: Anthropic_F03_T00_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

1. **Row-level `lapply` over 6.46M rows.** Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. With ~6.46 million iterations, the overhead of millions of small character-vector allocations and named-vector hash lookups is enormous.

2. **Redundant key construction.** Every cell-year row independently builds `neighbor_keys` by pasting neighbor IDs with its year. Because the neighbor topology is fixed across years, the same neighbor-ID sets are reconstructed 28 times (once per year per cell). This multiplies work by 28Ã—.

3. **Output is a list of 6.46M integer vectors.** This list alone consumes substantial memory and is expensive to build element-by-element.

**`compute_neighbor_stats`** is the second bottleneck:

1. **Row-level `lapply` over 6.46M rows**, each extracting a variable-length subset of a numeric vector, filtering NAs, and computing three summary statistics. Called 5 times (once per neighbor source variable) = ~32.3 million R-level function calls.

2. **`do.call(rbind, result)`** on a 6.46M-element list of length-3 vectors is a well-known R anti-pattern that creates massive intermediate copies.

### B. Random Forest Inference Bottleneck

With ~6.46 million rows and ~110 predictors, a single call to `predict(rf_model, newdata)` on the full dataset will:

1. **Construct a full model matrix** internally â€” copying and reformatting all 110 columns for 6.46M rows.
2. **Traverse every tree** for every row. Depending on `ntree` (commonly 500â€“2000), this is billions of tree-node comparisons.
3. **Memory pressure**: the `newdata` data.frame, the internal model matrix, and the prediction vector can easily exceed available RAM on a 16 GB laptop, causing swapping.

### C. Summary of Time Sinks (estimated share of 86+ hours)

| Component | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~25â€“35% | 6.46M string-paste + named-vector lookups |
| `compute_neighbor_stats` (Ã—5) | ~25â€“35% | 6.46M Ã— 5 row-level lapply + `do.call(rbind)` |
| `predict()` on full data | ~20â€“30% | Full model-matrix construction + tree traversal on 6.46M Ã— 110 |
| Object copying / memory pressure | ~10â€“15% | Repeated `cell_data <- ...` triggers copy-on-modify |

---

## 2. OPTIMIZATION STRATEGY

### Feature Preparation

| Problem | Solution |
|---|---|
| Per-row string pasting and named-vector lookup | Exploit the fact that neighbor topology is **year-invariant**. Build a cell-level neighbor lookup (344K entries) once, then use vectorized integer indexing to expand to cell-year rows via `data.table`. |
| Row-level `lapply` in `compute_neighbor_stats` | Replace with a **sparse adjacency matrix** (cells Ã— cells). For each year, matrix-multiply (or use sparse column operations) to compute neighbor sums, counts, max, min in fully vectorized operations. Alternatively, use `data.table` grouped joins. |
| `do.call(rbind, ...)` on millions of elements | Preallocate a matrix or use `data.table::set()` to write columns in-place. |
| Repeated `cell_data <- ...` copy-on-modify | Use `data.table` and `:=` for in-place column addition. |

### Random Forest Inference

| Problem | Solution |
|---|---|
| Single monolithic `predict()` call | **Chunk prediction** into batches of ~500Kâ€“1M rows to control peak memory. |
| Full data.frame copy inside `predict()` | Pass a **matrix** (not data.frame) to `predict()` when the model supports it (ranger, randomForest). |
| Model loading overhead | Load model **once**, keep in memory, reuse across chunks. |
| Potential for parallelism | Use `ranger` (if the model is ranger) which supports multi-threaded prediction natively; otherwise parallelize chunks. |

### Memory Management

- Convert `cell_data` to `data.table` once at the start.
- Add all new columns via `:=` (zero-copy in-place modification).
- Use `gc()` between major stages.
- Chunk prediction to keep peak memory well under 16 GB.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# =============================================================================
# Requirements: data.table, Matrix, ranger (or randomForest)
# Preserves: trained RF model (loaded from disk, never retrained)
# Preserves: original numerical estimand (identical predictions)
# =============================================================================

library(data.table)
library(Matrix)

# ---- 0. Convert to data.table (once) ----------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure integer types for join keys
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row-index column for fast reference
cell_data[, .row_idx := .I]

# ---- 1. OPTIMIZED NEIGHBOR LOOKUP (year-invariant, cell-level) ---------------
# rook_neighbors_unique: spdep nb object, indexed by position in id_order
# id_order: integer vector of cell IDs in the order matching the nb object

build_neighbor_dt <- function(id_order, neighbors) {
  # Build a data.table of directed edges: (focal_id, neighbor_id)
  # This is done once for 344K cells, not 6.46M cell-years.
  n_cells <- length(id_order)
  
  # Preallocate vectors
  focal_ids <- integer(0)
  neighbor_ids <- integer(0)
  
  # Estimate total edges for preallocation
  total_edges <- sum(lengths(neighbors))
  focal_ids <- integer(total_edges)
  neighbor_ids <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_len(n_cells)) {
    nb_idx <- neighbors[[i]]
    n_nb <- length(nb_idx)
    if (n_nb > 0L && !(n_nb == 1L && nb_idx[1L] == 0L)) {
      # spdep nb objects use 0 to indicate no neighbors
      nb_idx <- nb_idx[nb_idx != 0L]
      n_nb <- length(nb_idx)
      if (n_nb > 0L) {
        idx_range <- pos:(pos + n_nb - 1L)
        focal_ids[idx_range] <- id_order[i]
        neighbor_ids[idx_range] <- id_order[nb_idx]
        pos <- pos + n_nb
      }
    }
  }
  
  # Trim to actual size
  focal_ids <- focal_ids[seq_len(pos - 1L)]
  neighbor_ids <- neighbor_ids[seq_len(pos - 1L)]
  
  data.table(focal_id = focal_ids, neighbor_id = neighbor_ids)
}

cat("Building cell-level neighbor edge list...\n")
edge_dt <- build_neighbor_dt(id_order, rook_neighbors_unique)
cat(sprintf("  %d directed neighbor edges built.\n", nrow(edge_dt)))

# ---- 2. OPTIMIZED NEIGHBOR STATS (vectorized via data.table joins) -----------

compute_and_add_all_neighbor_features <- function(cell_data, edge_dt, 
                                                   neighbor_source_vars) {
  # Strategy:
  # 1. For each year, join focal->neighbor via edge_dt to get neighbor values.
  # 2. Compute max, min, mean per focal cell in one grouped aggregation.
  # 3. This avoids 6.46M row-level lapply calls entirely.
  
  # We process all years at once using a single large join.
  # Key insight: edges are year-invariant, so we cross-join edges with years.
  
  # Step 1: Create a keyed lookup of (id, year) -> variable values
  # We only need the neighbor_source_vars columns plus id and year.
  
  lookup_cols <- c("id", "year", neighbor_source_vars)
  val_dt <- cell_data[, ..lookup_cols]
  setnames(val_dt, "id", "neighbor_id")
  setkeyv(val_dt, c("neighbor_id", "year"))
  
  # Step 2: Expand edges by year â€” but instead of a full cross join (which would
  # be 1.37M edges Ã— 28 years = 38.5M rows), we do a join:
  # For each row in cell_data, find its neighbors and their values.
  
  # More memory-efficient approach: process year by year
  years <- sort(unique(cell_data$year))
  
  # Preallocate result columns
  for (var_name in neighbor_source_vars) {
    max_col <- paste0("n_max_", var_name)
    min_col <- paste0("n_min_", var_name)
    mean_col <- paste0("n_mean_", var_name)
    cell_data[, (max_col) := NA_real_]
    cell_data[, (min_col) := NA_real_]
    cell_data[, (mean_col) := NA_real_]
  }
  
  # Key cell_data for fast row matching
  setkey(cell_data, id, year)
  
  cat("Computing neighbor statistics for all variables...\n")
  
  for (yr in years) {
    cat(sprintf("  Year %d...\n", yr))
    
    # Get values for this year
    yr_vals <- cell_data[year == yr, c("id", neighbor_source_vars), with = FALSE]
    setnames(yr_vals, "id", "neighbor_id")
    setkey(yr_vals, neighbor_id)
    
    # Join edges with neighbor values for this year
    # edge_dt: (focal_id, neighbor_id)
    # yr_vals: (neighbor_id, ntl, ec, pop_density, def, usd_est_n2)
    edge_with_vals <- edge_dt[yr_vals, on = "neighbor_id", nomatch = 0L, 
                               allow.cartesian = TRUE]
    
    # Now aggregate by focal_id to get max, min, mean for each variable
    agg_exprs <- list()
    for (var_name in neighbor_source_vars) {
      max_col <- paste0("n_max_", var_name)
      min_col <- paste0("n_min_", var_name)
      mean_col <- paste0("n_mean_", var_name)
      agg_exprs[[max_col]] <- substitute(
        max(v, na.rm = TRUE), list(v = as.name(var_name)))
      agg_exprs[[min_col]] <- substitute(
        min(v, na.rm = TRUE), list(v = as.name(var_name)))
      agg_exprs[[mean_col]] <- substitute(
        mean(v, na.rm = TRUE), list(v = as.name(var_name)))
    }
    
    # Build the aggregation call
    agg_call <- as.call(c(as.name("list"), agg_exprs))
    stats_yr <- edge_with_vals[, eval(agg_call), by = focal_id]
    
    # Replace -Inf/Inf from max/min of empty sets with NA
    for (var_name in neighbor_source_vars) {
      max_col <- paste0("n_max_", var_name)
      min_col <- paste0("n_min_", var_name)
      mean_col <- paste0("n_mean_", var_name)
      stats_yr[is.infinite(get(max_col)), (max_col) := NA_real_]
      stats_yr[is.infinite(get(min_col)), (min_col) := NA_real_]
      stats_yr[is.nan(get(mean_col)), (mean_col) := NA_real_]
    }
    
    # Write results back into cell_data for this year
    # Match by id where year == yr
    setkey(stats_yr, focal_id)
    
    target_rows <- cell_data[year == yr, which = TRUE]
    target_ids <- cell_data[target_rows, id]
    
    # Create a lookup from focal_id to stats
    match_idx <- stats_yr[.(target_ids), which = TRUE]
    
    # More robust: merge approach
    # We'll use a direct indexed update
    result_cols <- names(stats_yr)[names(stats_yr) != "focal_id"]
    
    # Build a keyed version for matching
    matched <- stats_yr[.(target_ids), on = "focal_id"]
    
    for (col in result_cols) {
      set(cell_data, i = target_rows, j = col, value = matched[[col]])
    }
  }
  
  invisible(cell_data)
}

compute_and_add_all_neighbor_features(
  cell_data, edge_dt, 
  c("ntl", "ec", "pop_density", "def", "usd_est_n2")
)

gc()

# ---- 3. OPTIMIZED RANDOM FOREST PREDICTION (chunked, matrix-based) -----------

predict_rf_chunked <- function(model, data, feature_cols, 
                                chunk_size = 500000L) {
  # Allocate output vector
  n <- nrow(data)
  predictions <- numeric(n)
  
  # Determine if model is ranger or randomForest
  is_ranger <- inherits(model, "ranger")
  
  # Number of chunks
  n_chunks <- ceiling(n / chunk_size)
  
  cat(sprintf("Predicting %d rows in %d chunks of up to %d...\n", 
              n, n_chunks, chunk_size))
  
  for (ch in seq_len(n_chunks)) {
    start_idx <- (ch - 1L) * chunk_size + 1L
    end_idx <- min(ch * chunk_size, n)
    
    cat(sprintf("  Chunk %d/%d (rows %d-%d)...\n", ch, n_chunks, 
                start_idx, end_idx))
    
    # Extract chunk as a matrix for maximum predict() efficiency
    # data.table subsetting is fast and avoids full copy
    chunk_dt <- data[start_idx:end_idx, ..feature_cols]
    
    # Convert to matrix (RF predict on matrix avoids model.frame overhead)
    chunk_mat <- as.matrix(chunk_dt)
    
    if (is_ranger) {
      # ranger::predict returns a list with $predictions
      pred <- predict(model, data = chunk_mat)$predictions
    } else {
      # randomForest::predict
      pred <- predict(model, newdata = chunk_mat)
    }
    
    predictions[start_idx:end_idx] <- pred
    
    # Free chunk memory
    rm(chunk_dt, chunk_mat, pred)
    if (ch %% 5 == 0) gc()
  }
  
  predictions
}

# ---- 4. LOAD MODEL AND RUN PREDICTION ---------------------------------------

cat("Loading trained Random Forest model...\n")
# rf_model <- readRDS("path/to/trained_rf_model.rds")  # load once, reuse

# Identify the feature columns the model expects
# (Adjust this to match your actual model's expected features)
if (inherits(rf_model, "ranger")) {
  feature_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores feature names in the model
  feature_cols <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model))
}

# Verify all features exist in cell_data
missing_cols <- setdiff(feature_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop("Missing feature columns in cell_data: ", 
       paste(missing_cols, collapse = ", "))
}

cat("Running chunked prediction...\n")
cell_data[, predicted_gdp := predict_rf_chunked(
  model = rf_model,
  data = cell_data,
  feature_cols = feature_cols,
  chunk_size = 500000L
)]

cat("Done. Predictions stored in cell_data$predicted_gdp.\n")

# ---- 5. CLEANUP -------------------------------------------------------------
# Remove temporary row index if not needed
cell_data[, .row_idx := NULL]
gc()
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Stage | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup** | ~6.46M row-level `lapply` with string ops | 344K-cell edge table built once; no per-row string work | **~50â€“100Ã—** |
| **Neighbor stats** (Ã—5 vars) | 5 Ã— 6.46M `lapply` + `do.call(rbind)` | 28 year-wise vectorized `data.table` grouped aggregations | **~30â€“80Ã—** |
| **RF prediction** | Single `predict()` on 6.46M-row data.frame (memory thrashing) | Chunked matrix-based prediction, controlled memory | **~2â€“5Ã—** (+ eliminates OOM/swap) |
| **Column additions** | `cell_data <- cbind(...)` triggers full copy each time | `data.table` `:=` and `set()` â€” zero-copy in-place | **~5â€“10Ã—** |
| **Overall estimated wall time** | 86+ hours | **~1â€“3 hours** | **~30â€“80Ã—** |

### Why Numerical Results Are Preserved

- The same trained RF model object is loaded and used without modification.
- `predict()` receives exactly the same feature values (max, min, mean of the same neighbor sets) â€” only the *computation path* changes, not the *computed values*.
- `data.table` grouped `max`, `min`, `mean` are numerically identical to the base R equivalents applied to the same subsets.
- Chunked prediction produces identical results to a single `predict()` call â€” Random Forest prediction is row-independent (no cross-row state).
END_RESPONSE>>>
