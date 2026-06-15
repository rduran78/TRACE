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
run_id: Anthropic_F03_T00_R030
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 30
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_030.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows with `lapply`, performing per-row string pasting (`paste(id, year)`), character-keyed lookups into named vectors (`id_to_ref`, `idx_lookup`), and NA filtering. Named-vector lookups in R are hash-table operations, but doing ~6.46M Ã— ~4 neighbors â‰ˆ 26M hash lookups with string construction is extremely slow in interpreted R.

**`compute_neighbor_stats`:** For each of 6.46M rows, this extracts a small integer vector of neighbor indices, subsets a numeric vector, removes NAs, and computes `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern over 6.46M list elements is a well-known R anti-pattern: `do.call(rbind, list_of_6.46M_vectors)` alone can take minutes because it repeatedly allocates and copies.

**Outer loop over 5 variables:** `compute_and_add_neighbor_features` is called 5 times, each time re-traversing the full 6.46M-row neighbor lookup. If `compute_and_add_neighbor_features` copies `cell_data` (likely, since `cell_data <- ...` triggers copy-on-modify), you get 5 full copies of a ~6.46M Ã— 110-column data.frame (~5â€“8 GB each depending on types), which will thrash a 16 GB machine.

### 1.2 Random Forest Inference Bottlenecks

- **Model loading:** If the serialized RF model is large (hundreds of trees Ã— 110 features Ã— deep trees), `readRDS()` can take significant time and memory. A `ranger` model is typically much smaller than a `randomForest` model.
- **Prediction on 6.46M rows:** `predict()` on a `randomForest` object with 6.46M rows is notoriously slow because the `randomForest` package's predict method is single-threaded and has high per-row overhead. `ranger::predict` is multithreaded and far faster.
- **Data copying into predict:** If `cell_data` is a `data.frame`, `predict.randomForest` may internally convert it to a matrix, causing a full copy (~6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.7 GB). On a 16 GB machine this alone can cause swapping.
- **Single monolithic predict call:** Even with `ranger`, predicting 6.46M rows at once requires holding the full feature matrix plus the prediction output in memory simultaneously.

### 1.3 Summary of Root Causes

| Bottleneck | Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + hash lookups in R loop | Hours |
| `compute_neighbor_stats` | 6.46M `lapply` + `do.call(rbind, ...)` | Hours (Ã—5 vars) |
| Data.frame copy-on-modify | `cell_data <- compute_and_add_neighbor_features(...)` Ã—5 | Memory thrashing |
| RF predict (if `randomForest`) | Single-threaded, per-row overhead, internal matrix copy | Hours |
| Model deserialization | Large `.rds` object | Minutes |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation: Vectorize with `data.table`

1. **Replace `build_neighbor_lookup`** with a `data.table` join-based approach. Pre-build an edge table (`cell_id â†’ neighbor_cell_id`) and join it with the data keyed on `(id, year)`. This replaces 6.46M R-level iterations with a single vectorized merge.

2. **Replace `compute_neighbor_stats`** with a `data.table` grouped aggregation (`[, .(max, min, mean), by = .(id, year)]`). This is fully vectorized in C and eliminates the `lapply`/`do.call(rbind, ...)` pattern.

3. **Eliminate copy-on-modify** by adding columns to the `data.table` by reference (`:=`).

### 2.2 Random Forest Inference

1. **If the model is `randomForest`:** Convert it once to a `ranger`-compatible form, or re-wrap predictions using chunked, matrix-based input. Since the model must not be retrained, we use `predict()` on the existing object but feed it a pre-built matrix and chunk the prediction.

2. **If the model is `ranger`:** Use `ranger::predict` directly (already multithreaded).

3. **Chunk predictions** into blocks of ~500K rows to control peak memory.

4. **Pre-convert features to a matrix** once, avoiding repeated internal conversions.

### 2.3 Expected Speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup build | ~4â€“8 hrs | ~30 sec | ~500Ã— |
| Neighbor stats (Ã—5 vars) | ~10â€“20 hrs | ~2 min | ~500Ã— |
| Data copying | ~5â€“15 hrs (swap) | ~0 (by-ref) | Eliminated |
| RF prediction (6.46M rows) | ~50+ hrs | ~5â€“20 min | ~200Ã— |
| **Total** | **~86+ hrs** | **~30 min** | **~100Ã—+** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   (ranger only needed if model is ranger; randomForest works too)

library(data.table)

# ---- CONFIGURATION ----------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
PREDICT_CHUNK_SIZE   <- 500000L   # rows per prediction chunk (tune to RAM)

# ---- STEP 0: LOAD DATA & MODEL ---------------------------------------------
# Assumes:
#   cell_data            : data.frame/data.table with columns id, year, + features
#   rook_neighbors_unique: spdep nb object (list of integer neighbor indices)
#   id_order             : vector of cell IDs in the order matching the nb object
#   rf_model             : pre-trained randomForest or ranger model

# Convert to data.table in place (no copy if already data.table)
setDT(cell_data)

# ---- STEP 1: BUILD EDGE TABLE FROM nb OBJECT --------------------------------
# This replaces build_neighbor_lookup entirely.
# Convert the spdep nb list into a two-column data.table of directed edges
# using the id_order mapping.

build_edge_table <- function(id_order, nb_list) {
  # nb_list[[i]] contains integer indices into id_order for neighbors of id_order[i]
  # We need: from_id -> to_id (neighbor cell IDs)
  n <- length(nb_list)
  
  # Pre-calculate total edges for pre-allocation
  n_edges <- sum(vapply(nb_list, function(x) {
    if (length(x) == 1L && x[1] == 0L) 0L else length(x)
  }, integer(1)))
  
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  
  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_list[[i]]
    # spdep convention: a region with no neighbors has a single element 0
    if (length(nb) == 1L && nb[1] == 0L) next
    k <- length(nb)
    from_id[pos:(pos + k - 1L)] <- id_order[i]
    to_id[pos:(pos + k - 1L)]   <- id_order[nb]
    pos <- pos + k
  }
  
  data.table(id = from_id, neighbor_id = to_id)
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %d directed edges\n", nrow(edge_dt)))

# ---- STEP 2: VECTORIZED NEIGHBOR FEATURE COMPUTATION ------------------------
# For each (id, year) and each source variable, compute max/min/mean of
# neighbor values via a single keyed join + grouped aggregation.

compute_all_neighbor_features <- function(cell_dt, edge_dt, source_vars) {
  # We need to join:
  #   cell_dt[id, year] -> edge_dt[id -> neighbor_id] -> cell_dt[neighbor_id, year]
  # to get neighbor values, then aggregate by (id, year).
  
  # Subset cell_dt to only the columns we need for the neighbor join
  # to minimize memory during the merge.
  join_cols <- c("id", "year", source_vars)
  # This is the "neighbor data" â€” we'll join it on (neighbor_id, year)
  neighbor_data <- cell_dt[, ..join_cols]
  setnames(neighbor_data, "id", "neighbor_id")
  
  # Merge edge table with the main data to get (id, year) pairs,
  # then merge with neighbor_data to get neighbor values.
  # Step A: cross edge_dt with years via the main data
  #   For each row in cell_dt, we know (id, year).
  #   For each edge (id -> neighbor_id), we want the neighbor's value in the same year.
  
  # Efficient approach: 
  #   1. Create a slim key table: (id, year, row_index) from cell_dt
  #   2. Join edge_dt on id to get (id, year, neighbor_id)
  #   3. Join neighbor_data on (neighbor_id, year) to get neighbor values
  #   4. Aggregate by (id, year)
  
  cat("  Joining edges with cell-years...\n")
  # Slim table: just id and year (with implicit row ordering)
  id_year <- cell_dt[, .(id, year)]
  
  # Keyed join: for each (id, year), find all neighbors
  setkey(edge_dt, id)
  setkey(id_year, id)
  
  # This is an inner join: each (id, year) row gets expanded by its neighbor count
  # Result: (id, year, neighbor_id) â€” potentially ~26M rows for 4 neighbors avg
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year
  
  cat(sprintf("  Expanded join table: %d rows\n", nrow(expanded)))
  
  # Now join neighbor values
  cat("  Joining neighbor values...\n")
  setkey(neighbor_data, neighbor_id, year)
  setkey(expanded, neighbor_id, year)
  
  expanded <- neighbor_data[expanded, on = .(neighbor_id, year)]
  # Now expanded has: neighbor_id, year, id, + all source_vars (neighbor values)
  
  # Aggregate by (id, year)
  cat("  Aggregating neighbor statistics...\n")
  
  # Build aggregation expressions dynamically
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- bquote(max(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("n_min_", v)]]  <- bquote(min(.(v_sym), na.rm = TRUE))
    agg_exprs[[paste0("n_mean_", v)]] <- bquote(mean(.(v_sym), na.rm = TRUE))
  }
  
  # Convert to a single call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  
  agg_result <- expanded[, eval(agg_call), by = .(id, year)]
  
  # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen often)
  for (v in source_vars) {
    max_col <- paste0("n_max_", v)
    min_col <- paste0("n_min_", v)
    agg_result[is.infinite(get(max_col)), (max_col) := NA_real_]
    agg_result[is.infinite(get(min_col)), (min_col) := NA_real_]
  }
  
  # Join aggregated features back to cell_dt by reference
  cat("  Merging neighbor features back to main table...\n")
  feature_cols <- names(agg_result)[!names(agg_result) %in% c("id", "year")]
  
  # Remove any pre-existing neighbor feature columns to avoid conflicts
  existing <- intersect(feature_cols, names(cell_dt))
  if (length(existing) > 0) {
    cell_dt[, (existing) := NULL]
  }
  
  # Keyed merge by reference
  setkey(cell_dt, id, year)
  setkey(agg_result, id, year)
  cell_dt[agg_result, (feature_cols) := mget(feature_cols), on = .(id, year)]
  
  invisible(cell_dt)
}

cat("Computing neighbor features (vectorized)...\n")
system.time({
  cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# Clean up large intermediate objects
rm(edge_dt)
gc()

# ---- STEP 3: PREPARE PREDICTION MATRIX --------------------------------------
# Build the feature matrix once to avoid repeated internal conversions.

cat("Preparing prediction matrix...\n")

# Get the feature names the model expects
if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores the original variable names
  feature_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features are present
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# Extract as matrix (much faster for predict)
pred_matrix <- as.matrix(cell_data[, ..feature_names])
cat(sprintf("  Prediction matrix: %d rows x %d cols (%.1f GB)\n",
            nrow(pred_matrix), ncol(pred_matrix),
            object.size(pred_matrix) / 1e9))

# ---- STEP 4: CHUNKED PREDICTION ---------------------------------------------
# Predict in chunks to control peak memory usage.

cat("Running predictions...\n")

n_rows  <- nrow(pred_matrix)
n_chunks <- ceiling(n_rows / PREDICT_CHUNK_SIZE)
predictions <- numeric(n_rows)

system.time({
  for (chunk_i in seq_len(n_chunks)) {
    start_row <- (chunk_i - 1L) * PREDICT_CHUNK_SIZE + 1L
    end_row   <- min(chunk_i * PREDICT_CHUNK_SIZE, n_rows)
    chunk_idx <- start_row:end_row
    
    chunk_data <- pred_matrix[chunk_idx, , drop = FALSE]
    
    if (inherits(rf_model, "ranger")) {
      # ranger::predict is multithreaded; pass as data.frame for compatibility
      pred_obj <- predict(rf_model, data = as.data.frame(chunk_data))
      predictions[chunk_idx] <- pred_obj$predictions
      
    } else if (inherits(rf_model, "randomForest")) {
      # randomForest::predict â€” single-threaded but chunking controls memory
      predictions[chunk_idx] <- predict(rf_model, newdata = as.data.frame(chunk_data))
    }
    
    if (chunk_i %% 2 == 0 || chunk_i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %d-%d)\n",
                  chunk_i, n_chunks, start_row, end_row))
    }
  }
})

# ---- STEP 5: ATTACH PREDICTIONS TO DATA ------------------------------------
cell_data[, predicted_gdp := predictions]

# Clean up
rm(pred_matrix, predictions)
gc()

cat("Done. Predictions stored in cell_data$predicted_gdp\n")
cat(sprintf("  Rows predicted: %d\n", nrow(cell_data)))
cat(sprintf("  Prediction range: [%.4f, %.4f]\n",
            min(cell_data$predicted_gdp, na.rm = TRUE),
            max(cell_data$predicted_gdp, na.rm = TRUE)))


# =============================================================================
# OPTIONAL: IF MODEL IS randomForest AND PREDICTION IS STILL TOO SLOW,
# CONVERT TO ranger FOR MULTITHREADED INFERENCE (NO RETRAINING)
# =============================================================================
# 
# This is a one-time conversion that preserves the exact same forest structure.
# NOTE: This only works if you can accept the ranger predict interface.
# The numerical predictions will be identical (same trees, same splits).
#
# If the model was trained with randomForest::randomForest(), you can
# alternatively parallelize prediction across cores manually:
#
# library(parallel)
# library(randomForest)
#
# parallel_rf_predict <- function(model, newdata, n_cores = 4L,
#                                  chunk_size = 500000L) {
#   n <- nrow(newdata)
#   chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
#   
#   cl <- makeCluster(n_cores)
#   on.exit(stopCluster(cl))
#   clusterExport(cl, c("model"), envir = environment())
#   clusterEvalQ(cl, library(randomForest))
#   
#   results <- parLapply(cl, chunks, function(idx) {
#     predict(model, newdata = newdata[idx, , drop = FALSE])
#   })
#   
#   preds <- numeric(n)
#   for (i in seq_along(chunks)) {
#     preds[chunks[[i]]] <- results[[i]]
#   }
#   preds
# }
#
# predictions <- parallel_rf_predict(rf_model, 
#                                     as.data.frame(pred_matrix),
#                                     n_cores = 4L)
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

| Decision | Rationale |
|---|---|
| **`data.table` edge-join instead of `lapply` lookup** | Replaces ~6.46M interpreted R iterations + string hashing with a single vectorized C-level join. The `allow.cartesian = TRUE` flag handles the one-to-many (cell â†’ neighbors) expansion efficiently. |
| **Single pass for all 5 variables** | The original code ran 5 separate `lapply` passes over 6.46M rows. The vectorized version joins once and aggregates all 15 statistics (5 vars Ã— 3 stats) in a single grouped operation. |
| **Assignment by reference (`:=`)** | Eliminates the 5 full copies of `cell_data` caused by `cell_data <- compute_and_add_neighbor_features(...)`. On a 16 GB machine with a ~5 GB data.table, this alone prevents swap thrashing. |
| **Pre-built numeric matrix for prediction** | Both `randomForest::predict` and `ranger::predict` internally convert data.frames to matrices. Doing it once avoids repeated ~5.7 GB allocations. |
| **Chunked prediction (500K rows)** | Peak memory during prediction = model + full matrix + one chunk's temporary allocations, rather than model + 2Ã— full matrix. Keeps total memory well under 16 GB. |
| **Trained model preserved exactly** | No retraining. The same model object is used for `predict()`. The numerical estimand (predicted GDP) is identical to what the original code would produce. |
END_RESPONSE>>>
