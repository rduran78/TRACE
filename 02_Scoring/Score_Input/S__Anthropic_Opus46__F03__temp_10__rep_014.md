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
run_id: Anthropic_F03_T10_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows with `lapply`, performing per-row string pasting (`paste(id, year)`), character-keyed lookups into `idx_lookup`, and `as.character` coercions. For each of 6.46M rows it:
- Converts `data$id[i]` to character.
- Looks up the cell's reference index in a named character vector.
- Retrieves the neighbor cell IDs from the `nb` object.
- Pastes each neighbor ID with the current year to form keys.
- Performs named-vector lookups (which are sequential hash probes on a ~6.46M-element named vector).
- Filters NAs.

**Cost:** ~6.46M Ã— ~4 neighbors Ã— (string paste + named-vector lookup) â‰ˆ billions of character operations. Named-vector lookup in R is O(n) linear scan internally per call when the vector is large. This alone can take many hours.

**`compute_neighbor_stats`:** For each of the 5 source variables, it iterates over 6.46M rows again, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M list elements is extremely expensive (each `rbind` copies the growing matrix).

**Outer loop:** The neighbor lookup is rebuilt only once (good), but `compute_and_add_neighbor_features` is called 5 times, each adding columns to `cell_data`. If `cell_data` is a `data.frame`, each column addition copies the entire ~6.46M Ã— 110+ column frame (copy-on-modify semantics).

### 1.2 Random Forest Inference Bottlenecks

Predicting ~6.46M rows Ã— 110 features with `predict.randomForest` or `predict.ranger`:
- **`randomForest::predict`** is slow on large data: it converts to a dense matrix internally, and tree traversal is done in interpreted R/C with per-tree overhead.
- If the model is a `randomForest` object, each call to `predict()` copies the input data to a matrix. With 110 columns Ã— 6.46M rows Ã— 8 bytes â‰ˆ 5.7 GB just for the feature matrixâ€”already near or beyond 16 GB RAM when combined with the model, `cell_data`, and intermediate objects.
- If prediction is done row-by-row or in unnecessarily small chunks, the per-call overhead dominates.
- Loading a large serialized model from disk (potentially 1â€“4 GB for a Random Forest with many trees) is a one-time cost but still significant.

### 1.3 Memory Pressure

With 6.46M rows Ã— 110 columns Ã— 8 bytes â‰ˆ 5.7 GB for the feature matrix alone, plus the neighbor lookup list (~6.46M list elements, each an integer vector), plus the model in memory, plus intermediate copies, a 16 GB laptop will be under severe memory pressure, causing swapping that can slow everything by 10â€“100Ã—.

---

## 2. Optimization Strategy

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` string ops | Replace string-keyed lookup with integer arithmetic: encode `(id, year)` as a direct integer index using `data.table` keyed joins or a 2D integer matrix | 50â€“200Ã— |
| `compute_neighbor_stats` row-loop | Vectorize via `data.table` unnest + grouped aggregation: explode neighbor pairs, join values, group-aggregate | 20â€“100Ã— |
| `do.call(rbind, 6.46M-element list)` | Eliminate entirely; use pre-allocated matrix or `data.table` aggregation | 10â€“50Ã— |
| `data.frame` column addition (copy-on-modify) | Use `data.table` with `:=` (in-place column addition) | 5â€“20Ã— |
| RF prediction on 6.46M rows at once | Batch prediction in chunks (~500K rows) to control peak memory; use `ranger` if model permits, or convert model | 2â€“10Ã— |
| Memory pressure / swapping | Reduce peak memory by dropping intermediates, using `data.table`, chunked prediction | Prevents 10â€“100Ã— slowdown from swap |

### Key Principle: Vectorize Everything, Eliminate String Operations, Use `data.table`

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE: Feature preparation + Random Forest inference
# Preserves the trained RF model and original numerical estimand
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# STEP 0: Convert cell_data to data.table (in-place, no copy)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure id and year are integer for fast operations
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# --------------------------------------------------------------------------
# STEP 1: Build neighbor edge list (fully vectorized, replaces build_neighbor_lookup)
#
# rook_neighbors_unique is an nb object: a list of length = number of cells,
# where element i contains integer indices of neighbors of cell i in id_order.
# id_order is the vector mapping position -> cell_id.
# --------------------------------------------------------------------------
build_neighbor_edgelist <- function(id_order, neighbors) {
  # neighbors[[i]] gives indices (into id_order) of neighbors of cell id_order[i]
  n_cells <- length(id_order)

  # Number of neighbors per cell
  n_neighbors <- vapply(neighbors, length, integer(1))
  total_edges <- sum(n_neighbors)

  # Pre-allocate vectors for the edge list
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n_cells)) {
    nn <- n_neighbors[i]
    if (nn > 0L) {
      idx_range <- pos:(pos + nn - 1L)
      from_id[idx_range] <- id_order[i]
      to_id[idx_range]   <- id_order[neighbors[[i]]]
      pos <- pos + nn
    }
  }

  data.table(from_id = from_id, to_id = to_id)
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id
# Each row means: cell from_id has neighbor cell to_id

cat(sprintf("Edge list: %d directed edges\n", nrow(edge_dt)))

# --------------------------------------------------------------------------
# STEP 2: Compute all neighbor features at once (vectorized via data.table)
#
# For each (cell, year) and each source variable, we need:
#   neighbor_max, neighbor_min, neighbor_mean
# over that cell's rook neighbors in the same year.
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(cell_data, edge_dt, source_vars) {
  # Create a minimal lookup table: (id, year) -> values for source vars
  # Plus a row index so we can assign back
  lookup_cols <- c("id", "year", source_vars)
  lookup <- cell_data[, ..lookup_cols]

  # Merge edges with cell_data to get (from_id, year, to_id)
  # Then join to_id+year to get neighbor values
  # We need to cross edges with years: for each edge (from_id, to_id),
  # the neighbor relationship holds for ALL years.

  # Strategy: join cell_data with edge_dt on id == from_id,
  # then join the neighbor's values on (to_id, year).

  cat("  Joining edges with cell-year data...\n")

  # Step A: Get (from_id, year) pairs with their row indices
  # We add a row-index column for later assignment
  cell_data[, .row_idx := .I]

  # Step B: For each row in cell_data, find its neighbors via edge_dt
  # This is: cell_data[from_id == id] joined to edge_dt
  setkey(edge_dt, from_id)
  setkey(cell_data, id)

  # Expand: for each cell-year row, attach all neighbor cell IDs
  # Use a keyed join: cell_data's id -> edge_dt's from_id
  expanded <- edge_dt[cell_data[, .(id, year, .row_idx)],
                      on = .(from_id = id),
                      allow.cartesian = TRUE,
                      nomatch = NA]
  # expanded has columns: from_id, to_id, year, .row_idx
  # Rows where to_id is NA mean the cell has no neighbors -> will be handled

  # Remove rows with no neighbors
  expanded <- expanded[!is.na(to_id)]

  cat(sprintf("  Expanded table: %s rows\n", format(nrow(expanded), big.mark = ",")))

  # Step C: Join neighbor values
  # We need lookup keyed by (id, year) to get variable values for (to_id, year)
  setkey(lookup, id, year)

  neighbor_vals <- lookup[expanded, on = .(id = to_id, year = year), nomatch = NA]
  # neighbor_vals now has: from_id, to_id (= id from lookup), year, .row_idx,
  # plus all source_vars columns (these are the NEIGHBOR's values)

  cat("  Computing grouped aggregates...\n")

  # Step D: Group by .row_idx (original cell-year row) and compute stats
  # We compute max, min, mean for each source variable
  agg_exprs <- list()
  for (var in source_vars) {
    sym_var <- as.name(var)
    agg_exprs[[paste0("neighbor_max_", var)]]  <- substitute(
      suppressWarnings(max(V, na.rm = TRUE)), list(V = sym_var))
    agg_exprs[[paste0("neighbor_min_", var)]]  <- substitute(
      suppressWarnings(min(V, na.rm = TRUE)), list(V = sym_var))
    agg_exprs[[paste0("neighbor_mean_", var)]] <- substitute(
      mean(V, na.rm = TRUE), list(V = sym_var))
  }

  # Build the j expression for data.table
  j_expr <- as.call(c(as.name("list"),
                       setNames(agg_exprs, names(agg_exprs))))

  agg <- neighbor_vals[, eval(j_expr), by = .row_idx]

  # Fix Inf/-Inf from max/min on all-NA groups -> NA
  inf_cols <- grep("neighbor_max_|neighbor_min_", names(agg), value = TRUE)
  for (col in inf_cols) {
    set(agg, which(is.infinite(agg[[col]])), col, NA_real_)
  }
  # Fix NaN from mean on all-NA groups -> NA
  mean_cols <- grep("neighbor_mean_", names(agg), value = TRUE)
  for (col in mean_cols) {
    set(agg, which(is.nan(agg[[col]])), col, NA_real_)
  }

  cat("  Assigning features back to cell_data...\n")

  # Step E: Assign back to cell_data by .row_idx
  feature_cols <- setdiff(names(agg), ".row_idx")

  # Pre-allocate NA columns

  for (col in feature_cols) {
    set(cell_data, j = col, value = NA_real_)
  }

  # Assign via row index
  setkey(agg, .row_idx)
  for (col in feature_cols) {
    set(cell_data, i = agg$.row_idx, j = col, value = agg[[col]])
  }

  # Clean up temporary column
  cell_data[, .row_idx := NULL]

  cat("  Done.\n")
  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
cell_data <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Clean up large intermediate
rm(edge_dt)
gc()

# --------------------------------------------------------------------------
# STEP 3: Random Forest Prediction (chunked, memory-efficient)
# --------------------------------------------------------------------------
# The trained model is assumed to be loaded already as `rf_model`.
# If it's on disk:
#   rf_model <- readRDS("path/to/rf_model.rds")

cat("Preparing prediction...\n")

# Identify the feature columns the model expects
# For ranger models:
if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores predictor names differently
  feature_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model))
}

# Verify all required features are present
missing_features <- setdiff(feature_names, names(cell_data))
if (length(missing_features) > 0) {
  stop("Missing features in cell_data: ", paste(missing_features, collapse = ", "))
}

# --------------------------------------------------------------------------
# Chunked prediction to control peak memory
# With 110 features Ã— 500K rows Ã— 8 bytes â‰ˆ 440 MB per chunk (comfortable)
# --------------------------------------------------------------------------
predict_chunked <- function(model, data, feature_names, chunk_size = 500000L) {
  n <- nrow(data)
  n_chunks <- ceiling(n / chunk_size)
  predictions <- numeric(n)

  is_ranger <- inherits(model, "ranger")

  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","),
              n_chunks,
              format(chunk_size, big.mark = ",")))

  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)

    # Extract chunk as a plain data.frame (required by predict methods)
    chunk <- as.data.frame(data[start_idx:end_idx, ..feature_names])

    if (is_ranger) {
      pred <- predict(model, data = chunk)$predictions
    } else {
      pred <- predict(model, newdata = chunk)
    }

    predictions[start_idx:end_idx] <- pred

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %s-%s)\n",
                  i, n_chunks,
                  format(start_idx, big.mark = ","),
                  format(end_idx, big.mark = ",")))
    }

    # Release chunk memory
    rm(chunk, pred)
    if (i %% 10 == 0) gc()
  }

  predictions
}

cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, feature_names)]

cat("Pipeline complete.\n")

# --------------------------------------------------------------------------
# OPTIONAL: If the original model is randomForest (not ranger) and prediction
# is still too slow, convert it to ranger for faster predict(). This does NOT
# retrain; it restructures the existing trees.
# --------------------------------------------------------------------------
# If you have a `randomForest` object and want ranger-speed prediction,
# one option is to extract predictions tree-by-tree and average manually.
# However, the simplest and most reliable approach is:
#   1. Save the original randomForest predictions as ground truth.
#   2. Use chunked prediction as above (which already helps a lot).
#
# The chunking + data.table approach should reduce the full pipeline from
# 86+ hours to approximately 15â€“45 minutes depending on the RF model size
# and number of trees.
```

---

## 4. Summary of Changes and Expected Impact

| Component | Original | Optimized | Expected Time |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows, string paste + named-vector lookup | Pre-built integer edge list `data.table` (one loop over 344K cells) | ~5 seconds (was hours) |
| **Neighbor stats** | 5 Ã— `lapply` over 6.46M rows + `do.call(rbind, 6.46M-list)` | Single vectorized `data.table` grouped aggregation for all 5 vars simultaneously | ~2â€“5 minutes (was many hours) |
| **Column assignment** | `data.frame` `$<-` triggers full copy each time (15 new columns Ã— 5.7 GB) | `data.table::set()` in-place, zero copies | Eliminates ~85 GB of hidden copying |
| **RF prediction** | Presumably single call on 6.46M rows (peak ~12+ GB) causing swap | Chunked (500K rows), controlled peak memory (~1.5 GB per chunk) | ~10â€“30 minutes depending on model |
| **Total estimated** | 86+ hours | **~15â€“45 minutes** | **~100â€“300Ã— faster** |

### What Is Preserved
- **The trained Random Forest model** is loaded and used as-isâ€”no retraining.
- **The original numerical estimand** (predicted GDP per cell-year) is identical: the same features are computed with the same formulas (max, min, mean of neighbor values), and the same `predict()` function is called on the same model. The results are numerically equivalent.
END_RESPONSE>>>
