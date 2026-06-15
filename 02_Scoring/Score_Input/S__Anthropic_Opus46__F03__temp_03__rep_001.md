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
run_id: Anthropic_F03_T03_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows with `lapply`, performing per-row string pasting (`paste(id, year)`), character-key lookups into a named vector (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are O(n) hash lookups but the sheer volume (6.46M iterations Ã— ~4 neighbors each) and repeated `paste`/string allocation is extremely expensive. The entire structure is a list of 6.46M integer vectors â€” heavy on memory and GC pressure.

**`compute_neighbor_stats`:** Called 5 times (once per source variable), each call iterates over the 6.46M-element neighbor lookup list, subsetting a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end materialises a 6.46M Ã— 3 matrix from a list of 6.46M length-3 vectors â€” this alone is a massive allocation and copy operation.

**`compute_and_add_neighbor_features` (outer loop):** Each call likely column-binds new columns onto `cell_data`. If `cell_data` is a `data.frame`, each column addition triggers a full copy of the entire frame (~6.46M Ã— 110+ columns). Over 5 variables Ã— 3 stats = 15 new columns, this means ~15 full-frame copies.

### 1.2 Prediction Bottleneck

**Model loading:** If the serialized Random Forest is large (e.g., 500+ trees Ã— 110 predictors Ã— deep trees), `readRDS` and deserialization can take minutes and consume multiple GB of RAM.

**Prediction loop:** If `predict()` is called row-by-row or in small chunks rather than in a single vectorized call, overhead is catastrophic. Even a single `predict(model, newdata)` call on 6.46M rows can be slow if the model object is from `randomForest` (pure R tree traversal). The `ranger` package is 10â€“50Ã— faster for prediction.

**Memory:** A `data.frame` with 6.46M rows Ã— 125 columns of doubles â‰ˆ 6.0 GB. Combined with the RF model (potentially 2â€“4 GB) and intermediate copies, 16 GB RAM is easily exhausted, causing swapping and the 86+ hour runtime.

**Object copying:** R's copy-on-modify semantics mean that any modification to `cell_data` (adding columns, modifying values) triggers a full copy if there is more than one reference to the object.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| Data structure | `data.frame` copies on modification | Use `data.table` (modify in-place by reference) |
| Neighbor lookup | Per-row `paste`/string lookup in `lapply` | Vectorized integer-key join via `data.table` |
| Neighbor stats | 5 Ã— `lapply` over 6.46M list elements | Sparse matrix multiplication or `data.table` grouped aggregation |
| Column addition | 15 frame copies | `:=` assignment in `data.table` (zero-copy) |
| RF prediction | Possibly `randomForest::predict` (slow R traversal) | Convert model to `ranger` format or use single vectorized call; chunk if memory-bound |
| Memory | ~6 GB data + model + intermediates > 16 GB | `data.table` in-place ops; predict in chunks; `gc()` strategically |

### Key Insight: Replace the List-Based Neighbor Lookup with a Long-Format Join

Instead of building a 6.46M-element list and iterating over it, we create a two-column `data.table` of `(row_index, neighbor_row_index)` pairs (~1.37M Ã— 28 years â‰ˆ up to ~38M rows but typically less after NA filtering). Then computing neighbor max/min/mean is a single grouped `data.table` aggregation â€” fully vectorized in C.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (optional but recommended for predict speed)
# Preserves: trained Random Forest model, original numerical estimand

library(data.table)

# ---- 0. Convert cell_data to data.table (once, in-place) --------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in-place, no copy
}

# Ensure id and year are the types we expect
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# ---- 1. Build vectorized neighbor edge table --------------------------------
# This replaces build_neighbor_lookup entirely.
# rook_neighbors_unique is an nb object: a list of integer index vectors
# id_order is the vector mapping nb-index -> cell id

build_neighbor_edges <- function(id_order, neighbors) {
  # neighbors[[i]] gives the nb-indices of neighbors of cell id_order[i]
  # We need: (focal_cell_id, neighbor_cell_id) pairs
  n <- length(neighbors)
  
  # Pre-compute lengths for pre-allocation
  lens <- vapply(neighbors, length, integer(1))
  total <- sum(lens)
  
  focal_id    <- rep.int(as.integer(id_order), lens)
  neighbor_id <- as.integer(id_order[unlist(neighbors, use.names = FALSE)])
  
  data.table(focal_id = focal_id, neighbor_id = neighbor_id)
}

cat("Building neighbor edge table...\n")
edge_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# ---- 2. Build row-index mapping: (id, year) -> row_idx ---------------------
# We add a row index column to cell_data
cell_data[, .row_idx := .I]

# Create a keyed lookup table for (id, year) -> row_idx
id_year_key <- cell_data[, .(id, year, .row_idx)]
setkey(id_year_key, id, year)

# ---- 3. Expand edges across years and map to row indices --------------------
# For each year, every edge (focal_id, neighbor_id) becomes a
# (focal_row_idx, neighbor_row_idx) pair.

cat("Expanding edges across years...\n")
years <- sort(unique(cell_data$year))

# Cross join edges with years
edge_year <- CJ_dt_edges(edge_dt, years)

# Helper: efficient cross join of edges Ã— years
CJ_dt_edges <- function(edges, years) {
  # Repeat each edge for every year
  n_edges <- nrow(edges)
  n_years <- length(years)
  
  dt <- data.table(
    focal_id    = rep.int(edges$focal_id, n_years),
    neighbor_id = rep.int(edges$neighbor_id, n_years),
    year        = rep(years, each = n_edges)
  )
  dt
}

edge_year <- CJ_dt_edges(edge_dt, years)
cat(sprintf("  Edge-year table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

# Map focal (id, year) -> focal row index
setnames(id_year_key, c("id", "year", ".row_idx"), c("focal_id", "year", "focal_row"))
setkey(id_year_key, focal_id, year)
edge_year <- id_year_key[edge_year, on = .(focal_id, year), nomatch = 0L]

# Map neighbor (id, year) -> neighbor row index
setnames(id_year_key, c("focal_id", "year", "focal_row"), c("neighbor_id", "year", "neighbor_row"))
setkey(id_year_key, neighbor_id, year)
edge_year <- id_year_key[edge_year, on = .(neighbor_id, year), nomatch = 0L]

# Rename back for clarity
setnames(id_year_key, c("neighbor_id", "year", "neighbor_row"), c("id", "year", ".row_idx"))

# edge_year now has columns: focal_row, neighbor_row (and focal_id, neighbor_id, year)
# Keep only what we need
edge_year <- edge_year[, .(focal_row, neighbor_row)]
setkey(edge_year, focal_row)

cat(sprintf("  Mapped edge-year table: %s rows\n", format(nrow(edge_year), big.mark = ",")))

# Clean up
rm(edge_dt)
gc()

# ---- 4. Compute neighbor stats: vectorized grouped aggregation ---------------
# For each variable, look up neighbor values via edge_year, group by focal_row,
# compute max/min/mean. All in data.table C code â€” no R-level loops.

compute_and_add_neighbor_features_fast <- function(cell_data, var_name, edge_year) {
  cat(sprintf("  Computing neighbor stats for '%s'...\n", var_name))
  
  # Extract the variable values indexed by row
  vals <- cell_data[[var_name]]
  
  # Attach neighbor values to edge table
  edge_year[, nval := vals[neighbor_row]]
  
  # Remove edges where neighbor value is NA
  valid <- edge_year[!is.na(nval)]
  
  # Grouped aggregation (fully vectorized in C)
  stats <- valid[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = focal_row]
  
  # Prepare output columns (NA for rows with no valid neighbors)
  n <- nrow(cell_data)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)
  
  col_max[stats$focal_row]  <- stats$nb_max
  col_min[stats$focal_row]  <- stats$nb_min
  col_mean[stats$focal_row] <- stats$nb_mean
  
  # Assign by reference â€” no copy of cell_data
  max_name  <- paste0("neighbor_max_", var_name)
  min_name  <- paste0("neighbor_min_", var_name)
  mean_name <- paste0("neighbor_mean_", var_name)
  
  cell_data[, (max_name)  := col_max]
  cell_data[, (min_name)  := col_min]
  cell_data[, (mean_name) := col_mean]
  
  # Clean temp column from edge_year
  edge_year[, nval := NULL]
  
  invisible(NULL)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  compute_and_add_neighbor_features_fast(cell_data, var_name, edge_year)
}
cat("Neighbor features complete.\n")

# Free the edge table
rm(edge_year)
gc()

# Remove helper column
cell_data[, .row_idx := NULL]

# ---- 5. Random Forest Prediction (optimized) --------------------------------

cat("Loading trained Random Forest model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify predictor columns (exclude id, year, and the response variable)
# Adjust 'response_var' to your actual response column name
response_var <- "gdp"  # or whatever your target is
meta_cols    <- c("id", "year", response_var)
predictor_cols <- setdiff(names(cell_data), meta_cols)

# --- Option A: If model is a 'ranger' object (fastest) -----------------------
if (inherits(rf_model, "ranger")) {
  cat("Predicting with ranger (vectorized C++ backend)...\n")
  
  # Predict in chunks to stay within RAM
  chunk_size <- 500000L
  n <- nrow(cell_data)
  preds <- numeric(n)
  
  n_chunks <- ceiling(n / chunk_size)
  for (ch in seq_len(n_chunks)) {
    idx_start <- (ch - 1L) * chunk_size + 1L
    idx_end   <- min(ch * chunk_size, n)
    idx       <- idx_start:idx_end
    
    chunk_df <- as.data.frame(cell_data[idx, ..predictor_cols])
    pred_obj <- predict(rf_model, data = chunk_df)
    preds[idx] <- pred_obj$predictions
    
    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("    Chunk %d/%d complete\n", ch, n_chunks))
    }
  }
  
  cell_data[, predicted_gdp := preds]

# --- Option B: If model is a 'randomForest' object ---------------------------
} else if (inherits(rf_model, "randomForest")) {
  cat("Model is 'randomForest' class. Predicting in chunks...\n")
  cat("  TIP: For 10-50x faster prediction, convert to ranger or use\n")
  cat("        the 'predict' method with num.threads if available.\n")
  
  chunk_size <- 200000L
  n <- nrow(cell_data)
  preds <- numeric(n)
  
  n_chunks <- ceiling(n / chunk_size)
  for (ch in seq_len(n_chunks)) {
    idx_start <- (ch - 1L) * chunk_size + 1L
    idx_end   <- min(ch * chunk_size, n)
    idx       <- idx_start:idx_end
    
    chunk_df <- as.data.frame(cell_data[idx, ..predictor_cols])
    preds[idx] <- predict(rf_model, newdata = chunk_df)
    
    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("    Chunk %d/%d complete\n", ch, n_chunks))
    }
    
    # Periodic garbage collection to manage memory
    if (ch %% 10 == 0) gc()
  }
  
  cell_data[, predicted_gdp := preds]
  
} else {
  # Generic fallback
  cat("Unknown model class: ", class(rf_model), ". Attempting generic predict...\n")
  chunk_size <- 200000L
  n <- nrow(cell_data)
  preds <- numeric(n)
  
  n_chunks <- ceiling(n / chunk_size)
  for (ch in seq_len(n_chunks)) {
    idx_start <- (ch - 1L) * chunk_size + 1L
    idx_end   <- min(ch * chunk_size, n)
    idx       <- idx_start:idx_end
    
    chunk_df <- as.data.frame(cell_data[idx, ..predictor_cols])
    preds[idx] <- predict(rf_model, newdata = chunk_df)
    
    if (ch %% 5 == 0 || ch == n_chunks) {
      cat(sprintf("    Chunk %d/%d complete\n", ch, n_chunks))
    }
  }
  
  cell_data[, predicted_gdp := preds]
}

cat("Prediction complete.\n")

# ---- 6. (Optional) Convert model from randomForest -> ranger for speed ------
# If you have the original training data and formula, you can re-wrap:
#
#   library(ranger)
#   # This does NOT retrain â€” but ranger has no direct import from randomForest.
#   # Instead, at prediction time you can use the 'ranger' predict C++ backend
#   # by saving your model as ranger from the start.
#
# Alternatively, use the 'trimTrees' or serialization tricks to reduce model
# object size for faster readRDS.

# ---- 7. Summary of expected improvements ------------------------------------
cat("\n=== OPTIMIZATION SUMMARY ===\n")
cat("Feature preparation:\n")
cat("  - build_neighbor_lookup: eliminated (replaced by vectorized edge table)\n")
cat("  - compute_neighbor_stats: replaced by data.table grouped aggregation\n")
cat("  - Column addition: in-place via := (zero-copy)\n")
cat("  - Expected speedup: ~50-200x for feature preparation\n")
cat("Prediction:\n")
cat("  - Chunked prediction avoids OOM on 16 GB RAM\n")
cat("  - ranger backend (if available): 10-50x faster than randomForest\n")
cat("  - Expected total runtime: minutes to low single-digit hours\n")
cat("    (vs. 86+ hours originally)\n")
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Expected Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | 6.46M `lapply` iterations with `paste` + named-vector lookup | Single vectorized `data.table` join to build edge table | **~100Ã—** |
| `compute_neighbor_stats` | 5 Ã— 6.46M `lapply` iterations with per-element subsetting | 5 Ã— single `data.table` grouped aggregation (C backend) | **~50â€“200Ã—** |
| Column binding | 15 full `data.frame` copies (~6 GB each) | `data.table` `:=` (zero-copy, in-place) | **~15Ã— memory, ~10Ã— time** |
| RF prediction | Possibly single giant call or row-level loop | Chunked vectorized `predict()`, optionally `ranger` | **~2â€“50Ã—** |
| **Total estimated runtime** | **86+ hours** | **~15 min â€“ 2 hours** | **~40â€“300Ã—** |

The trained Random Forest model is loaded as-is from disk and used only for `predict()` â€” it is never retrained or modified. The numerical predictions are identical to the original pipeline (same model, same features, same computation of max/min/mean), preserving the original estimand.
END_RESPONSE>>>
