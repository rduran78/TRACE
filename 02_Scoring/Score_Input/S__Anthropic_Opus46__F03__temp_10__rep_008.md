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
run_id: Anthropic_F03_T10_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

- It calls `lapply` over **~6.46 million rows**, and inside each iteration it performs **character coercion** (`as.character`), **string pasting** (`paste(..., sep="_")`), **named-vector lookups** (which are hash-table lookups on character keys), and **NA filtering**. This means ~6.46M Ã— (multiple string allocations + hash lookups per neighbor) â€” easily billions of micro-operations.
- The `idx_lookup` named vector has ~6.46M entries. Named lookups on vectors this large are slow because R's internal hashing degrades.
- Every iteration allocates new character vectors (`neighbor_keys`), creating enormous GC (garbage collection) pressure.

**`compute_neighbor_stats`** is the second bottleneck:

- Called **5 times** (once per neighbor source variable), each time iterating over 6.46M rows with `lapply`, extracting subsets, computing `max/min/mean`, and returning a 3-element vector.
- The final `do.call(rbind, result)` binds 6.46M small vectors into a matrix â€” this is a known R anti-pattern that is extremely slow and memory-hungry because `rbind` on a list of millions of small vectors forces repeated memory reallocation.

**Outer loop** compounds the problem: it mutates `cell_data` inside a for loop, triggering potential full-frame copies (R's copy-on-modify semantics). With ~110 columns Ã— 6.46M rows, each accidental copy is ~5â€“8 GB.

### 1.2 Random Forest Inference Bottleneck

- Predicting 6.46M rows Ã— 110 features through a Random Forest (even a pre-trained one) is inherently expensive. If done with `predict.randomForest` from the `randomForest` package, **it is single-threaded** and iterates tree-by-tree in R/C.
- If the model has hundreds of trees, this single-threaded prediction alone can take hours.
- Loading a large serialized RF model from disk with `readRDS` can also be slow and spike memory (the model object for 110 features can be multiple GB).

### 1.3 Memory Pressure

- 6.46M rows Ã— 110 numeric columns â‰ˆ **5.3 GB** just for the data.
- The neighbor lookup list (6.46M elements, each an integer vector of ~4 neighbors) â‰ˆ **0.5â€“1 GB**.
- The RF model itself may be 1â€“4 GB.
- On a 16 GB laptop, this leaves almost no headroom, causing swap thrashing and GC storms.

### Summary of Root Causes (ranked by impact)

| Rank | Bottleneck | Estimated Share |
|------|-----------|----------------|
| 1 | `build_neighbor_lookup` â€” per-row string ops + hash lookups on 6.46M rows | ~30-40% |
| 2 | `compute_neighbor_stats` â€” per-row `lapply` + `do.call(rbind, ...)` Ã— 5 vars | ~25-30% |
| 3 | RF prediction â€” single-threaded, 6.46M rows | ~20-25% |
| 4 | Data frame copies on mutation inside the for loop | ~10-15% |

---

## 2. OPTIMIZATION STRATEGY

### Feature Preparation

1. **Replace character-key hash lookups with integer arithmetic.** Since `id` and `year` are bounded integers, encode `(id, year)` â†’ row index via a `data.table` keyed join or direct integer mapping. Eliminate all `paste` and named-vector lookups.

2. **Use `data.table` throughout.** Convert `cell_data` to a `data.table`. Use keyed joins for neighbor lookups. Compute neighbor stats vectorized using `data.table` grouping or pre-built integer index matrices.

3. **Vectorize `compute_neighbor_stats`.** Unroll the neighbor lookup into a long-form table `(row_i, neighbor_row_j)`, join the variable values, and compute grouped `max/min/mean` with `data.table` â€” this replaces 6.46M `lapply` iterations with a single vectorized grouped aggregation.

4. **Compute all 5 variables' neighbor stats in one pass** over the long-form neighbor table, avoiding 5 separate iterations.

5. **Avoid mutating the data frame in a loop.** Build all new columns at once via `:=` in `data.table`.

### Random Forest Inference

6. **Use a multi-threaded prediction backend.** If the model is from `ranger`, it already supports multi-threaded `predict`. If it is from `randomForest`, convert the predict call to use `ranger`'s prediction on the existing forest structure, or chunk the prediction and parallelize with `future.apply` / `parallel::mclapply`.

7. **Predict in chunks** to control peak memory â€” e.g., 500K rows at a time.

### Memory

8. **Remove intermediate objects aggressively** (`rm()` + `gc()`).
9. **Use single-precision (`float` package) for the prediction matrix** if the RF predict method supports it (saves 50% memory).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table, ranger (or randomForest), parallel
# =============================================================================

library(data.table)

# ---- Configuration ---------------------------------------------------------
CHUNK_SIZE     <- 500000L   # rows per prediction chunk (tune to RAM)
N_CORES        <- parallel::detectCores() - 1L
NEIGHBOR_VARS  <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
STAT_SUFFIXES  <- c("max", "min", "mean")

# ---- Step 0: Load data & model ---------------------------------------------
# cell_data           : data.frame / data.table with columns id, year, + predictors
# rook_neighbors_unique : spdep nb object (list of integer neighbor indices into id_order)
# id_order            : integer vector of cell IDs in the order matching nb object
# rf_model            : pre-trained Random Forest model (loaded via readRDS)

cat("Converting to data.table...\n")
if (!is.data.table(cell_data)) setDT(cell_data)

# ---- Step 1: Build neighbor lookup via integer indexing ---------------------
cat("Building optimized neighbor lookup...\n")

build_neighbor_lookup_fast <- function(dt, id_order, nb_list) {
  # Map each cell id -> position in the nb object
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Create row-index lookup keyed on (id, year) using data.table
  dt[, .row_idx := .I]
  row_lookup <- dt[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # For each row, find its neighbors' row indices

# Vectorised: expand each row to its neighbor cell IDs, then join
  cat("  Expanding neighbor relationships...\n")

  # Step A: For every unique cell id, get its neighbor cell ids
  unique_ids <- unique(dt$id)
  # ref index into nb_list for each unique id
  ref_indices <- id_to_ref[as.character(unique_ids)]

  # Build a data.table: (cell_id, neighbor_cell_id)
  # Only for cells that actually appear in the data
  nb_edges <- rbindlist(lapply(seq_along(unique_ids), function(k) {
    ri <- ref_indices[k]
    if (is.na(ri) || length(nb_list[[ri]]) == 0) return(NULL)
    nb_cell_ids <- id_order[nb_list[[ri]]]
    data.table(id = unique_ids[k], neighbor_id = nb_cell_ids)
  }))

  cat("  Edge table built:", nrow(nb_edges), "unique (cell, neighbor) pairs\n")

  # Step B: Join with data to get (row_i, neighbor_row_j) for every year
  # For each row in dt, its neighbors share the same year
  # Approach: join nb_edges to dt on 'id', carrying year and .row_idx,
  #           then join again to get neighbor's row index in the same year.

  # Left table: every (row, year, neighbor_cell_id)
  cat("  Joining to panel years...\n")
  dt_slim <- dt[, .(id, year, row_i = .row_idx)]
  setkey(dt_slim, id)
  setkey(nb_edges, id)

  # merge: for each row, attach its neighbor cell ids
  expanded <- nb_edges[dt_slim, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # expanded has columns: id, neighbor_id, year, row_i

  # Now find neighbor's row index in the same year
  setnames(row_lookup, c("id", "year", ".row_idx"), c("neighbor_id", "year", "row_j"))
  setkey(row_lookup, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  result <- row_lookup[expanded, on = c("neighbor_id", "year"), nomatch = 0L]
  # result has columns: neighbor_id, year, row_j, id, row_i

  # Clean up temporary column
  dt[, .row_idx := NULL]

  return(result[, .(row_i, row_j)])
}

edge_dt <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
cat("Neighbor edge table:", nrow(edge_dt), "directed row-pairs\n")

# ---- Step 2: Compute all neighbor stats vectorized --------------------------
cat("Computing neighbor stats for all variables...\n")

compute_all_neighbor_stats <- function(dt, edge_dt, var_names) {
  n <- nrow(dt)

  # Pre-allocate result columns
  new_cols <- character(0)
  for (v in var_names) {
    for (s in STAT_SUFFIXES) {
      col_name <- paste0("neighbor_", v, "_", s)
      new_cols <- c(new_cols, col_name)
      set(dt, j = col_name, value = rep(NA_real_, n))
    }
  }

  for (v in var_names) {
    cat("  Processing variable:", v, "\n")
    # Attach the neighbor's value to each edge
    edge_dt[, val := dt[[v]][row_j]]

    # Remove NAs and compute grouped stats
    stats <- edge_dt[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     by = row_i]

    # Write results back into dt
    col_max  <- paste0("neighbor_", v, "_max")
    col_min  <- paste0("neighbor_", v, "_min")
    col_mean <- paste0("neighbor_", v, "_mean")

    set(dt, i = stats$row_i, j = col_max,  value = stats$nb_max)
    set(dt, i = stats$row_i, j = col_min,  value = stats$nb_min)
    set(dt, i = stats$row_i, j = col_mean, value = stats$nb_mean)
  }

  edge_dt[, val := NULL]  # clean up
  invisible(dt)
}

compute_all_neighbor_stats(cell_data, edge_dt, NEIGHBOR_VARS)

# Free the edge table
rm(edge_dt)
gc()

cat("Feature preparation complete. Columns:", ncol(cell_data), "\n")

# ---- Step 3: Random Forest prediction (chunked, multi-threaded) -------------
cat("Starting Random Forest prediction...\n")

predict_rf_chunked <- function(dt, model, chunk_size = CHUNK_SIZE) {
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)

  # Identify the feature columns the model expects
  # Works for both 'randomForest' and 'ranger' objects
  if (inherits(model, "ranger")) {
    feature_names <- model$forest$independent.variable.names
  } else if (inherits(model, "randomForest")) {
    # randomForest stores feature names in the model
    feature_names <- rownames(model$importance)
  } else {
    stop("Unsupported model class: ", class(model)[1])
  }

  # Pre-allocate prediction vector
  predictions <- numeric(n)

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    cat("  Predicting chunk", i, "/", n_chunks,
        " (rows", start_row, "-", end_row, ")\n")

    chunk_data <- dt[start_row:end_row, ..feature_names]

    if (inherits(model, "ranger")) {
      # ranger predict is already multi-threaded via num.threads
      pred <- predict(model, data = chunk_data, num.threads = N_CORES)
      predictions[start_row:end_row] <- pred$predictions
    } else {
      # randomForest predict â€” single-threaded in C, but chunking controls memory
      predictions[start_row:end_row] <- predict(model, newdata = chunk_data)
    }

    rm(chunk_data)
    if (i %% 5 == 0) gc()  # periodic GC to release chunk memory
  }

  return(predictions)
}

cell_data[, predicted_gdp := predict_rf_chunked(cell_data, rf_model)]

cat("Prediction complete.\n")

# ---- OPTIONAL Step 4: Parallelize randomForest predict if not ranger --------
# If the model is 'randomForest' and prediction is still too slow,
# you can parallelize across trees manually:

predict_rf_parallel <- function(dt, model, n_cores = N_CORES) {
  stopifnot(inherits(model, "randomForest"))
  feature_names <- rownames(model$importance)
  pred_data <- as.matrix(dt[, ..feature_names])

  n_trees <- model$ntree
  tree_groups <- split(seq_len(n_trees), cut(seq_len(n_trees), n_cores))

  cl <- parallel::makeCluster(n_cores)
  on.exit(parallel::stopCluster(cl))

  # Export minimal objects to workers
  parallel::clusterExport(cl, varlist = c("model", "pred_data"), envir = environment())

  # Each worker predicts using a subset of trees, then we average
  tree_predictions <- parallel::parLapply(cl, tree_groups, function(tree_idx) {
    # predict.randomForest supports predict.all to get per-tree predictions
    p <- predict(model, newdata = pred_data, predict.all = TRUE)
    # p$individual is a matrix: n_rows Ã— n_trees
    rowMeans(p$individual[, tree_idx, drop = FALSE])
  })

  # Average across all tree groups (weighted by group size)
  result <- Reduce(`+`, Map(`*`, tree_predictions,
                            lapply(tree_groups, length))) / n_trees
  return(result)
}

# To use (uncomment if model is randomForest and chunked prediction is too slow):
# cell_data[, predicted_gdp := predict_rf_parallel(cell_data, rf_model)]
```

---

## 4. EXPECTED IMPROVEMENT SUMMARY

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46M `lapply` with string ops (est. 20â€“30 hrs) | Single `data.table` keyed join (~2â€“5 min) | **~300Ã—** |
| `compute_neighbor_stats` (Ã—5 vars) | 5 Ã— 6.46M `lapply` + `do.call(rbind, ...)` (est. 30â€“40 hrs) | 5 Ã— one `data.table` grouped aggregation (~1â€“3 min each) | **~200Ã—** |
| Data frame mutation (copy-on-modify) | 5 full-frame copies (~25â€“40 GB total writes) | In-place `set()` on `data.table` (zero copies) | **eliminates** |
| RF prediction (6.46M rows) | Single-threaded, full-data (est. 10â€“15 hrs) | Chunked + multi-threaded (ranger) or tree-parallel | **4â€“8Ã—** |
| **Total pipeline** | **~86+ hours** | **~30 min â€“ 2 hours** (depending on RF backend) | **~40â€“170Ã—** |

### Key Invariants Preserved
- **The trained Random Forest model is never retrained or modified** â€” only used for `predict()`.
- **The original numerical estimand is preserved** â€” predictions are the same numbers (identical tree traversals, same `max/min/mean` aggregations), just computed far more efficiently.
END_RESPONSE>>>
