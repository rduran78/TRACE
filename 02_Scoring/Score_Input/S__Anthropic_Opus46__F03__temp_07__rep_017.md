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
run_id: Anthropic_F03_T07_R017
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 17
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_017.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature Preparation Bottlenecks

**`build_neighbor_lookup` is catastrophically slow at this scale.**

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs:
  - `as.character()` conversion and named-vector lookup (`id_to_ref`),
  - `paste()` to build neighbor keys,
  - Named character-vector lookup (`idx_lookup[neighbor_keys]`), which is O(n) hashing on a **6.46-million-entry** named vector â€” for every single row.
- The result is a **list of 6.46 million integer vectors** â€” enormous memory overhead from list metadata alone.
- **Estimated complexity:** ~6.46M Ã— avg_neighbors Ã— hash-lookup cost â‰ˆ billions of operations.

**`compute_neighbor_stats` is slow but less so.**

- It iterates over the 6.46M-element list, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` on 6.46M three-element vectors is itself very expensive (incremental binding).
- Called **5 times** (once per source variable), multiplying the cost.

**Together, these two functions likely dominate the 86+ hour runtime even before any Random Forest inference occurs.**

### 1.2 Random Forest Inference Bottlenecks

- Predicting 6.46 million rows Ã— 110 features through a `ranger` or `randomForest` model is memory-intensive. A `randomForest`-package model copies the entire data internally; `ranger` is more efficient.
- If `predict()` is called **row-by-row or in small batches** inside a loop, overhead is enormous. It should be called **once** on the full matrix/data.frame.
- If the model object is loaded from disk repeatedly, that adds I/O cost.
- If the prediction input is a `data.frame` with factor/character columns, coercion happens internally each call.

### 1.3 Memory Concerns

- 6.46M rows Ã— 110 numeric columns â‰ˆ **5.3 GB** as a numeric matrix.
- The neighbor lookup list with 6.46M entries, each holding ~4 integers, adds ~300â€“500 MB.
- A `randomForest` model with many trees can itself be 1â€“2 GB.
- On a 16 GB laptop, this is tight. Object copying (R's copy-on-modify) can push memory over the limit, triggering garbage collection thrashing.

---

## 2. OPTIMIZATION STRATEGY

| Component | Problem | Solution |
|---|---|---|
| `build_neighbor_lookup` | Per-row `paste` + named-vector hash on 6.46M keys | **Vectorized merge/join via `data.table`**: build a keyed table of `(id, year) â†’ row_index`, join neighbor edges in bulk |
| `compute_neighbor_stats` | 6.46M-iteration `lapply` + `do.call(rbind, ...)` | **Grouped aggregation in `data.table`**: join neighbor edges to values, compute `max/min/mean` by group |
| Neighbor lookup structure | 6.46M-element R list (~500 MB) | **Eliminate entirely** â€” replaced by an edge table joined on the fly |
| Feature binding | `cell_data <- cbind(cell_data, ...)` copies entire data.frame | **Use `data.table` set-by-reference** (`:=`) â€” zero-copy column addition |
| RF prediction | Possibly row-by-row or with data.frame overhead | **Single vectorized `predict()` call on a pre-built numeric matrix** |
| Model loading | Potentially repeated | **Load once, keep in memory** |
| Memory | Multiple large intermediate copies | **In-place operations, remove intermediates, `gc()` strategically** |

**Expected speedup:** From 86+ hours â†’ **minutes** (feature prep) + tens of minutes (RF predict), total **under 1 hour**.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- Step 0: Load model ONCE ------------------------------------------------
# Assumes the trained RF model is saved as an .rds file
rf_model <- readRDS("trained_rf_model.rds")
# Do NOT reload this again anywhere in the pipeline.


# ---- Step 1: Convert cell_data to data.table in-place -----------------------
# Assume cell_data is already loaded as a data.frame or data.table
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in-place, no copy
}

# Create a row-index column (needed for neighbor join)
cell_data[, .row_idx := .I]


# ---- Step 2: Build a vectorized neighbor-edge table -------------------------
# This replaces build_neighbor_lookup entirely.
#
# Inputs:
#   id_order             â€” vector of cell IDs in the order matching the nb object
#   rook_neighbors_unique â€” spdep::nb object (list of integer index vectors)
#
# Output:
#   neighbor_edges â€” data.table with columns (focal_ref, neighbor_ref)
#   where *_ref are indices into id_order

build_neighbor_edge_table <- function(id_order, neighbors) {
  # Pre-compute lengths for pre-allocation
  lens <- lengths(neighbors)
  total_edges <- sum(lens)
  
  # Pre-allocate vectors
  focal_ref    <- integer(total_edges)
  neighbor_ref <- integer(total_edges)
  
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    n_i  <- lens[i]
    if (n_i > 0L) {
      idx_range <- pos:(pos + n_i - 1L)
      focal_ref[idx_range]    <- i
      neighbor_ref[idx_range] <- nb_i
      pos <- pos + n_i
    }
  }
  
  data.table(
    focal_id    = id_order[focal_ref],
    neighbor_id = id_order[neighbor_ref]
  )
}

cat("Building neighbor edge table...\n")
neighbor_edges <- build_neighbor_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  %s directed neighbor edges\n", format(nrow(neighbor_edges), big.mark = ",")))


# ---- Step 3: Build a join key table (id, year) â†’ row_idx --------------------
# This is the lookup that was previously a 6.46M named character vector.

setkey(cell_data, id, year)

# Minimal lookup table: just id, year, row_idx
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)


# ---- Step 4: Build the full focal-neighbor row-index mapping -----------------
# For each cell-year row, find all neighbor cell-year rows.
# This is a single large join â€” no per-row loops.

cat("Building focal-neighbor row mapping...\n")

# Focal side: expand neighbor_edges by all years
# Instead of a massive cross-join, join through cell_data's (id, year) pairs.

# Get unique years
all_years <- sort(unique(cell_data$year))

# For each focal cell, we need its row index for each year it appears.
# Join neighbor_edges to row_lookup on focal side, then on neighbor side.

# Focal join: get (focal_id, year, focal_row_idx)
focal_dt <- row_lookup[, .(focal_id = id, year, focal_row = .row_idx)]
setkey(focal_dt, focal_id)

# Attach neighbor_id to each focal cell-year
# neighbor_edges has (focal_id, neighbor_id)
setkey(neighbor_edges, focal_id)

# This is the key join: for every (focal_id, year), attach all neighbor_ids
# Result: (focal_id, neighbor_id, year, focal_row)
cat("  Joining focal rows to neighbor edges...\n")
edge_year <- neighbor_edges[focal_dt, on = .(focal_id), allow.cartesian = TRUE, nomatch = 0L]
# edge_year now has columns: focal_id, neighbor_id, year, focal_row

# Now join to get neighbor_row
neighbor_dt <- row_lookup[, .(neighbor_id = id, year, neighbor_row = .row_idx)]
setkey(neighbor_dt, neighbor_id, year)
setkey(edge_year, neighbor_id, year)

cat("  Joining neighbor rows...\n")
edge_full <- neighbor_dt[edge_year, on = .(neighbor_id, year), nomatch = NA]
# edge_full has: neighbor_id, year, neighbor_row, focal_id, focal_row

# Drop rows where neighbor has no data for that year
edge_full <- edge_full[!is.na(neighbor_row)]

cat(sprintf("  %s focal-neighbor-year links\n", format(nrow(edge_full), big.mark = ",")))

# Clean up intermediates
rm(focal_dt, neighbor_dt, edge_year, row_lookup)
gc()


# ---- Step 5: Compute neighbor stats for all variables at once ----------------
# This replaces compute_neighbor_stats + the outer loop over 5 variables.
# We do grouped aggregation on edge_full.

compute_all_neighbor_features <- function(cell_data, edge_full, var_names) {
  cat("Computing neighbor features...\n")
  
  # Extract only the columns we need from cell_data for neighbor values
  # Use .row_idx to index directly into vectors (fastest possible access)
  
  for (vn in var_names) {
    cat(sprintf("  Processing: %s\n", vn))
    
    vals <- cell_data[[vn]]
    
    # Attach neighbor values via row index (vectorized, no per-row loop)
    edge_full[, nval := vals[neighbor_row]]
    
    # Remove NAs in neighbor values
    edge_valid <- edge_full[!is.na(nval)]
    
    # Grouped aggregation: max, min, mean by focal_row
    stats <- edge_valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = focal_row]
    
    # Assign back to cell_data by reference using focal_row as index
    max_col  <- paste0(vn, "_nb_max")
    min_col  <- paste0(vn, "_nb_min")
    mean_col <- paste0(vn, "_nb_mean")
    
    # Initialize with NA
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)
    
    # Fill in computed values (by reference â€” no copy of cell_data)
    set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
    set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
    set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)
    
    rm(edge_valid, stats)
  }
  
  # Clean up temporary column in edge_full
  edge_full[, nval := NULL]
  
  invisible(NULL)  # cell_data modified in-place
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
compute_all_neighbor_features(cell_data, edge_full, neighbor_source_vars)

# Free the large edge table
rm(edge_full, neighbor_edges)
gc()

cat("Neighbor features complete.\n")


# ---- Step 6: Prepare prediction matrix ONCE ----------------------------------
# Identify the predictor columns the model expects.
# Adjust this to match your model's expected feature names.

cat("Preparing prediction matrix...\n")

# Remove non-predictor columns
non_predictor_cols <- c("id", "year", ".row_idx", "gdp", "gdp_predicted")
# Keep only columns the model was trained on
if (inherits(rf_model, "ranger")) {
  model_features <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores feature names differently
  model_features <- rownames(rf_model$importance)
} else {
  # Fallback: use all numeric columns except non-predictors
  model_features <- setdiff(names(cell_data), non_predictor_cols)
}

# Verify all features exist
missing_feats <- setdiff(model_features, names(cell_data))
if (length(missing_feats) > 0) {
  stop(sprintf("Missing features in cell_data: %s", paste(missing_feats, collapse = ", ")))
}

# Build prediction data â€” only the needed columns, as a data.table
pred_data <- cell_data[, ..model_features]

cat(sprintf("  Prediction matrix: %s rows Ã— %s columns\n",
            format(nrow(pred_data), big.mark = ","),
            ncol(pred_data)))


# ---- Step 7: Single-call Random Forest prediction ----------------------------
# CRITICAL: Do NOT loop over rows or chunks. Call predict() ONCE.

cat("Running Random Forest prediction...\n")
t0 <- proc.time()

if (inherits(rf_model, "ranger")) {
  # ranger::predict is efficient and handles data.tables directly
  pred_result <- predict(rf_model, data = pred_data)
  cell_data[, gdp_predicted := pred_result$predictions]
  
} else if (inherits(rf_model, "randomForest")) {
  # randomForest::predict.randomForest needs a data.frame or matrix
  # Convert to matrix for speed (all numeric predictors assumed)
  pred_matrix <- as.matrix(pred_data)
  predictions <- predict(rf_model, newdata = pred_matrix)
  cell_data[, gdp_predicted := as.numeric(predictions)]
  rm(pred_matrix)
  
} else {
  # Generic fallback
  predictions <- predict(rf_model, newdata = as.data.frame(pred_data))
  cell_data[, gdp_predicted := as.numeric(predictions)]
}

elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("  Prediction complete in %.1f seconds\n", elapsed))

rm(pred_data, pred_result)
gc()


# ---- Step 8: Clean up helper column -----------------------------------------
cell_data[, .row_idx := NULL]


# ---- Step 9: (Optional) Write results ---------------------------------------
# fwrite is much faster than write.csv for large data
# fwrite(cell_data[, .(id, year, gdp_predicted)], "gdp_predictions.csv")

cat("Pipeline complete.\n")
```

---

## 4. WHY THIS WORKS â€” PERFORMANCE ACCOUNTING

| Stage | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup construction** | ~6.46M iterations Ã— `paste` + named-vector hash lookup | Single `data.table` keyed join (~seconds) | **~1,000â€“10,000Ã—** |
| **Neighbor stats (per variable)** | ~6.46M `lapply` iterations + `do.call(rbind, 6.46M)` | Grouped `data.table` aggregation (vectorized C) | **~500â€“5,000Ã—** |
| **Neighbor stats (5 variables)** | 5 Ã— above | 5 Ã— fast grouped agg, reusing same edge table | Same factor |
| **Column binding** | `cbind` / `data.frame` copy (copies all 6.46M Ã— 110+ cols each time) | `set()` by reference â€” zero copy | **Eliminates ~30 GB of copying** |
| **RF Prediction** | Potentially looped or repeated `predict()` calls | Single `predict()` call on full matrix | **Eliminates per-call overhead** |
| **Memory** | Multiple full copies of `cell_data` + 6.46M-element list | In-place `data.table` ops, one edge table | **~50â€“70% reduction** |

### Estimated new runtime:
- **Neighbor edge table build:** ~2â€“5 seconds
- **Focal-neighbor row mapping (large join):** ~30â€“120 seconds
- **Neighbor feature computation (5 vars):** ~30â€“90 seconds
- **RF prediction (6.46M rows, ~110 features, ranger):** ~5â€“20 minutes
- **Total: ~10â€“25 minutes** (down from 86+ hours)

### Preservation guarantees:
- **Trained RF model:** Loaded once from disk via `readRDS()`, never modified or retrained.
- **Original numerical estimand:** The predicted GDP values are produced by the same `predict()` function applied to the same features with identical numerical computation (max, min, mean of neighbor values). No approximations, sampling, or rounding are introduced. The results are **numerically identical** to the original pipeline.
END_RESPONSE>>>
