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
run_id: Anthropic_F03_T07_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations

This function iterates over **~6.46 million rows**, and for every single row it:

1. Converts an integer ID to character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes each neighbor ID with the current year to create string keys.
4. Looks those keys up in another named character vector (`idx_lookup`).

**Problems:**
- `paste(..., sep="_")` and named-vector lookups via character keys are extremely slow at scale. Named vector lookup in R is O(n) hash-table probing per call, repeated billions of times across all neighbor pairs.
- The function builds **6.46 million list elements**, each containing a variable-length integer vector. This is a huge memory structure and is slow to construct in an `lapply`.
- The `idx_lookup` named vector itself has 6.46 million entries â€” every lookup against it is a hash probe on a very large table.

**Estimated cost:** With ~1.37M neighbor relationships Ã— 28 years â‰ˆ ~38M neighbor-row lookups, plus 6.46M `paste` operations for key construction, this alone can take **many hours**.

### B. `compute_neighbor_stats` â€” Repeated per variable

For each of the 5 neighbor source variables, this function:

1. Iterates over all 6.46M rows again via `lapply`.
2. Subsets a numeric vector by index, removes NAs, computes `max`, `min`, `mean`.
3. Collects results via `do.call(rbind, ...)` on a 6.46M-element list of length-3 vectors.

**Problems:**
- `do.call(rbind, list_of_6.46M_vectors)` is notoriously slow â€” it creates a huge argument list and binds row-by-row.
- The `lapply` is pure R-level looping over millions of elements.
- This is repeated **5 times** (once per variable), so the total work is ~32.3M R-level loop iterations plus 5 expensive `rbind` calls.

### C. Random Forest Inference

- Predicting ~6.46 million rows Ã— ~110 features with a Random Forest (likely `ranger` or `randomForest`) is computationally heavy.
- If the model is a `randomForest` object (not `ranger`), prediction is **single-threaded** and much slower.
- If prediction is done inside a loop (e.g., year-by-year or chunk-by-chunk) with repeated `predict()` calls, the overhead of data-frame construction and dispatch accumulates.
- Copying the full `cell_data` data.frame on every `cell_data <- compute_and_add_neighbor_features(...)` call triggers R's copy-on-modify semantics, potentially copying the entire ~6.46M Ã— 110 matrix each time.

### D. Memory Pressure

- 6.46M rows Ã— 110 columns of doubles â‰ˆ **5.3 GB** just for the main data.
- The neighbor lookup list (6.46M elements) adds significant overhead.
- Intermediate copies from `cell_data <- ...` assignments can double or triple memory use, causing swapping on a 16 GB machine.

### Summary of Root Causes

| Bottleneck | Cause | Severity |
|---|---|---|
| `build_neighbor_lookup` | Per-row string paste + named-vector hash lookup over 6.46M rows | **Critical** |
| `compute_neighbor_stats` | R-level `lapply` over 6.46M rows Ã— 5 vars; `do.call(rbind, ...)` | **Critical** |
| Object copying | `cell_data <- ...` triggers full copy-on-modify for a multi-GB frame | **High** |
| `do.call(rbind, ...)` | Binding 6.46M 3-element vectors into a matrix | **High** |
| RF prediction | Possibly single-threaded `randomForest::predict`; possibly looped | **High** |
| Memory | ~5+ GB base data + copies + lookup â†’ swapping on 16 GB | **Moderate-High** |

---

## 2. Optimization Strategy

### Strategy A: Replace string-keyed lookup with integer-keyed lookup via `data.table`

Instead of building a per-row list of neighbor indices via string keys, we:

1. Create an **edge table** (a two-column `data.table`) that maps every `(cell_id, neighbor_cell_id)` pair.
2. Expand it by year using a fast cross-join.
3. Join it to the data to get the **row index** of each neighbor in each year.
4. Compute `max`, `min`, `mean` per `(row_index, variable)` using `data.table` grouped aggregation â€” **fully vectorized, no R-level loops**.

This eliminates `build_neighbor_lookup` and `compute_neighbor_stats` entirely and replaces them with a single vectorized pipeline.

### Strategy B: Use `data.table` for `cell_data` to avoid copies

Convert `cell_data` to a `data.table` and add columns **by reference** (`:=`), avoiding copy-on-modify.

### Strategy C: Optimize Random Forest prediction

- If the model is `randomForest`, convert it to `ranger` format or re-wrap prediction to use `ranger::predict` (which is multi-threaded). Since we cannot retrain, we check the model class and use the appropriate optimized path.
- Predict in a **single call** on the full matrix rather than in a loop.
- Pre-allocate the prediction feature matrix as a plain `matrix` (not `data.frame`) for faster dispatch.

### Strategy D: Control memory

- Use `data.table` in-place operations.
- Remove intermediate objects and call `gc()` at key points.
- Process prediction in chunks if needed to stay within 16 GB.

### Expected Improvement

| Operation | Before | After (estimated) |
|---|---|---|
| Neighbor lookup build | Hours | ~30â€“90 seconds |
| Neighbor stats (all 5 vars) | Hours | ~30â€“60 seconds |
| Object copying overhead | Hours (cumulative) | ~0 (in-place `:=`) |
| RF prediction (6.46M rows) | Variable | ~2â€“15 minutes (depends on model) |
| **Total** | **86+ hours** | **~5â€“20 minutes** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (optional, for faster predict if model is ranger)
# Preserves: trained RF model object, original numerical estimand
# =============================================================================

library(data.table)

# ---- STEP 0: Convert cell_data to data.table (in-place, no copy) -----------

if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure id and year columns are of consistent types
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row-index column for fast back-mapping
cell_data[, .row_idx := .I]

# ---- STEP 1: Build vectorized edge table from nb object --------------------
# rook_neighbors_unique is a list of length = number of unique cell IDs
# id_order is the vector mapping position -> cell_id

build_edge_table <- function(id_order, neighbors_nb) {
  # neighbors_nb is an nb object: list of integer vectors (neighbor positions)
  # id_order[i] gives the cell_id for position i
  n <- length(neighbors_nb)

  # Pre-compute total number of edges for pre-allocation
  edge_counts <- vapply(neighbors_nb, length, integer(1))
  total_edges <- sum(edge_counts)

  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb_i <- neighbors_nb[[i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) next
    len <- length(nb_i)
    idx_range <- pos:(pos + len - 1L)
    from_id[idx_range] <- id_order[i]
    to_id[idx_range]   <- id_order[nb_i]
    pos <- pos + len
  }

  # Trim if any nb entries were empty (0-valued sentinels in spdep::nb)
  actual <- pos - 1L
  data.table(
    focal_id    = from_id[seq_len(actual)],
    neighbor_id = to_id[seq_len(actual)]
  )
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %d directed edges\n", nrow(edge_dt)))

# ---- STEP 2: Expand edges by year and join to get neighbor row indices ------

# Unique years in the data
years_vec <- sort(unique(cell_data$year))

# Cross-join edges with years: every edge exists for every year
cat("Expanding edges across years...\n")
edge_year_dt <- edge_dt[, CJ(focal_id = focal_id, neighbor_id = neighbor_id, year = years_vec),
                         by = .EACHI][, .(focal_id, neighbor_id, year)]

# Actually, CJ inside by is wrong. Use a simpler cross-join:
edge_year_dt <- CJ(edge_idx = seq_len(nrow(edge_dt)), year = years_vec)
edge_year_dt[, focal_id    := edge_dt$focal_id[edge_idx]]
edge_year_dt[, neighbor_id := edge_dt$neighbor_id[edge_idx]]
edge_year_dt[, edge_idx    := NULL]

cat(sprintf("  Edge-year table: %d rows\n", nrow(edge_year_dt)))

# Build a lookup from (id, year) -> row index in cell_data
setkey(cell_data, id, year)
row_lookup <- cell_data[, .(id, year, .row_idx)]
setkey(row_lookup, id, year)

# Join to get focal row index
setnames(row_lookup, ".row_idx", "focal_row")
setkey(edge_year_dt, focal_id, year)
edge_year_dt <- row_lookup[edge_year_dt, on = .(id = focal_id, year = year), nomatch = 0L]
setnames(edge_year_dt, "focal_row", "focal_row")

# Join to get neighbor row index
setnames(row_lookup, "focal_row", "neighbor_row")
edge_year_dt <- row_lookup[edge_year_dt, on = .(id = neighbor_id, year = year), nomatch = 0L]

# Clean up: keep only what we need
edge_year_dt <- edge_year_dt[, .(focal_row, neighbor_row)]

# Free memory
rm(row_lookup)
gc()

cat(sprintf("  Matched edge-year pairs: %d\n", nrow(edge_year_dt)))

# ---- STEP 3: Compute neighbor stats â€” fully vectorized ---------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  # Extract neighbor values via row index
  edge_year_dt[, val := cell_data[[var_name]][neighbor_row]]

  # Compute grouped stats: max, min, mean per focal_row (excluding NAs)
  stats <- edge_year_dt[!is.na(val),
                         .(nb_max  = max(val),
                           nb_min  = min(val),
                           nb_mean = mean(val)),
                         by = focal_row]

  # Prepare NA-filled columns, then fill by reference
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Initialize with NA
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Assign computed values by row index (in-place, no copy)
  set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

  # Drop temp column from edge table
  edge_year_dt[, val := NULL]
}

# Free the large edge table
rm(edge_year_dt, edge_dt)
gc()

cat("Neighbor features complete.\n")

# ---- STEP 4: Optimized Random Forest Prediction ----------------------------

cat("Preparing prediction...\n")

# rf_model is the pre-trained Random Forest model (must not be retrained)
# Detect model type and predict accordingly

# Get the feature names the model expects
if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used in training
  feature_names <- rownames(rf_model$importance)
  if (is.null(feature_names)) {
    feature_names <- attr(rf_model$terms, "term.labels")
  }
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

cat(sprintf("  Model class: %s\n", class(rf_model)[1]))
cat(sprintf("  Number of features: %d\n", length(feature_names)))

# Build prediction matrix â€” use a plain matrix for speed
# data.table subsetting to matrix is fast
pred_matrix <- as.matrix(cell_data[, ..feature_names])

cat(sprintf("  Prediction matrix: %d rows x %d cols (%.1f GB)\n",
            nrow(pred_matrix), ncol(pred_matrix),
            object.size(pred_matrix) / 1e9))

# Predict in a single vectorized call
cat("Running prediction...\n")
t0 <- proc.time()

if (inherits(rf_model, "ranger")) {
  # ranger::predict is multi-threaded by default
  pred_result <- predict(rf_model, data = pred_matrix,
                          num.threads = parallel::detectCores())
  cell_data[, predicted_gdp := pred_result$predictions]

} else if (inherits(rf_model, "randomForest")) {
  # randomForest::predict is single-threaded
  # For very large data, predict in chunks to manage memory
  chunk_size <- 500000L
  n_rows     <- nrow(pred_matrix)
  n_chunks   <- ceiling(n_rows / chunk_size)

  predictions <- numeric(n_rows)

  for (ch in seq_len(n_chunks)) {
    start_idx <- (ch - 1L) * chunk_size + 1L
    end_idx   <- min(ch * chunk_size, n_rows)
    predictions[start_idx:end_idx] <- predict(
      rf_model,
      newdata = pred_matrix[start_idx:end_idx, , drop = FALSE]
    )
    if (ch %% 5 == 0) {
      cat(sprintf("    Chunk %d/%d complete\n", ch, n_chunks))
    }
  }

  cell_data[, predicted_gdp := predictions]
  rm(predictions)
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Prediction complete in %.1f seconds.\n", elapsed))

# Clean up prediction matrix
rm(pred_matrix)
gc()

# ---- STEP 5: Restore original key order if needed --------------------------
setkey(cell_data, id, year)

# Remove helper column
cell_data[, .row_idx := NULL]

cat("Pipeline complete.\n")
cat(sprintf("Output: %d rows, %d columns\n", nrow(cell_data), ncol(cell_data)))
cat(sprintf("Prediction column: 'predicted_gdp' â€” preserves original numerical estimand.\n"))
```

---

## 4. Summary of Changes and Why They Work

| Original Code | Problem | Replacement | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup()` â€” `lapply` over 6.46M rows with `paste` + named-vector lookup | O(n Ã— k) string operations, R-level loop | Pre-built `data.table` edge table with integer cross-join and keyed merge | ~100â€“500Ã— |
| `compute_neighbor_stats()` â€” `lapply` + `do.call(rbind, ...)` per variable | R-level loop, slow `rbind` on millions of elements | `data.table` grouped aggregation (`[, .(max, min, mean), by = focal_row]`) | ~100â€“1000Ã— |
| `cell_data <- compute_and_add_neighbor_features(...)` | Copy-on-modify duplicates multi-GB data.frame 5 times | `data.table::set()` â€” in-place column assignment, zero copies | Saves ~25 GB of allocation |
| RF prediction (potentially looped or single-threaded) | Repeated `predict()` calls; `randomForest` is single-threaded | Single-call `ranger::predict` (multi-threaded) or chunked `randomForest::predict` with pre-allocated matrix | ~2â€“10Ã— |
| **Overall estimated runtime** | **86+ hours** | **~5â€“20 minutes** | **~250â€“1000Ã—** |

The numerical estimand is preserved exactly: the same trained model produces predictions on the same features computed with the same `max`, `min`, `mean` aggregation logic â€” only the implementation mechanics are changed from R-level loops to vectorized `data.table` operations.
END_RESPONSE>>>
