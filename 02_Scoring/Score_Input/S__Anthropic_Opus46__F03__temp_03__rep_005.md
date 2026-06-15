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
run_id: Anthropic_F03_T03_R005
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 5
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_005.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations

| Problem | Detail |
|---|---|
| **String key creation inside a 6.46M-iteration `lapply`** | `paste(neighbor_cell_ids, data$year[i], sep="_")` and the named-vector lookup `idx_lookup[neighbor_keys]` are executed once per row. Named-vector lookup in R is hash-based but the overhead of creating millions of character keys and performing named indexing is enormous. |
| **`id_to_ref` and `idx_lookup` are named character vectors** | Every access is a hash-table probe on a character key. With 6.46M entries in `idx_lookup`, each probe is non-trivial, and it's done ~6.46M Ã— avg_neighbors times. |
| **Result is a list of 6.46M integer vectors** | This list alone consumes significant memory (each list element is a separate R object with overhead â‰¥ 64 bytes + data). For 6.46M elements this is â‰¥ 400 MB of overhead before data. |
| **Single-threaded R `lapply`** | No parallelism, no vectorization. |

### B. `compute_neighbor_stats` â€” repeated per variable

| Problem | Detail |
|---|---|
| **Called 5 times, each iterating over 6.46M list elements** | Each call does `vals[idx]`, NA removal, and `max/min/mean` â€” all inside an `lapply`. That's ~32.3M R function calls. |
| **`do.call(rbind, result)` on 6.46M 3-element vectors** | This is a well-known R anti-pattern: it creates a list of 6.46M tiny vectors then row-binds them. Very slow and memory-hungry. |

### C. Random Forest Prediction (inferred)

| Problem | Detail |
|---|---|
| **`predict.randomForest` on 6.46M rows Ã— 110 features** | The default `predict()` method in the `randomForest` package is pure R and single-threaded. It iterates tree-by-tree in R. |
| **Possible chunking / row-loop** | If prediction is done in a loop (row-by-row or small chunks), overhead is catastrophic. |
| **Model object copying** | If the model is large (hundreds of MB) and R's copy-on-modify triggers, memory pressure causes swapping on a 16 GB machine. |
| **Data frame coercion** | `predict()` may coerce the input to a data.frame internally, doubling memory. |

### Estimated time breakdown (86+ hours)

| Phase | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~25â€“35% |
| `compute_neighbor_stats` (Ã—5) | ~30â€“40% |
| RF prediction | ~20â€“30% |
| Memory pressure / GC / swapping | ~10â€“20% |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Replace R-level loops with vectorized `data.table` joins and C-level operations.

| Bottleneck | Fix |
|---|---|
| `build_neighbor_lookup` (per-row `lapply` + string keys) | Build an **edge-list `data.table`** from the `nb` object in one vectorized pass. Join on integer `(id, year)` pairs â€” no strings, no list-of-vectors. |
| `compute_neighbor_stats` (per-row `lapply` Ã— 5 vars) | **One grouped `data.table` aggregation** over the edge-list computes max/min/mean for **all 5 variables simultaneously**. |
| `do.call(rbind, ...)` on millions of rows | Eliminated â€” `data.table` returns a single table. |
| RF prediction (single-threaded, possible row-loop) | Use `predict()` on the **full matrix at once**. If the model is from `ranger`, it's already C++-threaded. If from `randomForest`, convert to `ranger`-compatible format or just ensure single-call prediction on a `matrix` (not `data.frame`). Chunk if memory-limited. |
| Memory pressure | Operate in-place with `data.table` `:=`. Avoid copies. Remove intermediates. Use `gc()` at key points. |

**Expected speedup: from 86+ hours to roughly 10â€“30 minutes** (depending on RF tree count).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” cell-level GDP prediction
# Requirements: data.table, randomForest (or ranger)
# =============================================================================

library(data.table)

# ---- Step 0: Ensure cell_data is a data.table with an integer row key -------

setDT(cell_data)

# Create a unique integer row identifier (preserves original order for output)
cell_data[, .row_id := .I]

# Ensure id and year are integer for fast joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Set key for fast joins
setkey(cell_data, id, year)


# ---- Step 1: Build edge-list from nb object (vectorized, no lapply) ---------

build_edge_list_dt <- function(id_order, nb_obj) {

  # nb_obj is a list of length N; nb_obj[[i]] contains integer indices into
  # id_order for the neighbors of id_order[i].
  # We expand this into a two-column data.table: (focal_id, neighbor_id).

  n <- length(nb_obj)
  lens <- lengths(nb_obj)                       # integer vector, C-level
  focal_idx <- rep.int(seq_len(n), lens)         # vectorized
  neighbor_idx <- unlist(nb_obj, use.names = FALSE)

  # Remove the spdep "0 = no neighbors" sentinel if present
  valid <- neighbor_idx > 0L
  focal_idx    <- focal_idx[valid]
  neighbor_idx <- neighbor_idx[valid]

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building edge list...\n")
edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))


# ---- Step 2: Compute neighbor stats for all variables in one join -----------

compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # Join the edge list to cell_data to get neighbor rows.
  # For each (focal row, year), we look up all neighbor_id rows in the same year.

  # Minimal columns needed from cell_data for the neighbor side:
  neighbor_cols <- c("id", "year", var_names)
  neighbor_dt   <- cell_dt[, ..neighbor_cols]
  setnames(neighbor_dt, "id", "neighbor_id")
  setkey(neighbor_dt, neighbor_id, year)

  # Attach year from focal rows to the edge list so we can join on (neighbor_id, year)
  focal_key <- cell_dt[, .(focal_id = id, year, .row_id)]
  setkey(focal_key, focal_id)

  # Expand edges Ã— years: for every focal row, attach its year
  # edge_dt has (focal_id, neighbor_id); focal_key has (focal_id, year, .row_id)
  cat("  Expanding edges Ã— years...\n")
  setkey(edge_dt, focal_id)
  edges_with_year <- edge_dt[focal_key, on = "focal_id",
                             allow.cartesian = TRUE, nomatch = 0L]
  # Result columns: focal_id, neighbor_id, year, .row_id

  # Now join to get neighbor variable values
  cat("  Joining neighbor values...\n")
  setkey(edges_with_year, neighbor_id, year)
  edges_with_vals <- neighbor_dt[edges_with_year, on = c("neighbor_id", "year"),
                                  nomatch = NA]

  # Aggregate: group by .row_id (= focal row), compute max/min/mean per variable
  cat("  Aggregating neighbor stats...\n")

  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(var_names, function(v) {
    list(
      bquote(as.numeric(max(.(as.name(v)),   na.rm = TRUE))),
      bquote(as.numeric(min(.(as.name(v)),   na.rm = TRUE))),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("neighbor_", c("max_", "min_", "mean_"), v)
  }))

  names(agg_exprs) <- agg_names

  stats_dt <- edges_with_vals[, lapply(agg_exprs, eval), by = .row_id]

  # Replace Inf / -Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(stats_dt, which(is.infinite(stats_dt[[col]])), col, NA_real_)
  }

  setkey(stats_dt, .row_id)
  return(stats_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
neighbor_stats <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Merge back into cell_data by .row_id (in-place)
cat("Merging neighbor features into cell_data...\n")
cell_data <- neighbor_stats[cell_data, on = ".row_id"]

# Clean up large intermediates
rm(edge_dt, neighbor_stats)
gc()


# ---- Step 3: Optimized Random Forest Prediction -----------------------------

predict_rf_optimized <- function(model, newdata, feature_names,
                                  chunk_size = 500000L) {
  # Ensure we pass a matrix (avoids internal data.frame coercion overhead).
  # Predict in chunks to stay within memory limits on a 16 GB laptop.

  n <- nrow(newdata)
  preds <- numeric(n)

  # Pre-extract the feature matrix once (data.table -> matrix is fast)
  cat("  Extracting feature matrix...\n")
  feat_mat <- as.matrix(newdata[, ..feature_names])

  n_chunks <- ceiling(n / chunk_size)
  cat(sprintf("  Predicting in %d chunks of up to %s rows...\n",
              n_chunks, format(chunk_size, big.mark = ",")))

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    idx       <- start_row:end_row

    chunk_mat <- feat_mat[idx, , drop = FALSE]

    # Works for both randomForest::predict and ranger::predict
    if (inherits(model, "ranger")) {
      preds[idx] <- predict(model, data = chunk_mat)$predictions
    } else {
      # randomForest package
      preds[idx] <- predict(model, newdata = chunk_mat)
    }

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("    Chunk %d/%d done (rows %sâ€“%s)\n",
                  i, n_chunks,
                  format(start_row, big.mark = ","),
                  format(end_row,   big.mark = ",")))
    }
  }

  rm(feat_mat)
  gc()

  return(preds)
}

# Load the trained model ONCE
cat("Loading trained RF model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")  # adjust path

# Identify feature names (must match training)
# Exclude non-feature columns:
exclude_cols <- c("id", "year", ".row_id", "gdp", "gdp_predicted")
feature_names <- setdiff(names(cell_data), exclude_cols)
# Ensure they match the model's expected features:
if (inherits(rf_model, "ranger")) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores row names of importance matrix
  feature_names <- intersect(rownames(rf_model$importance), names(cell_data))
}

cat(sprintf("Predicting with %d features across %s rows...\n",
            length(feature_names),
            format(nrow(cell_data), big.mark = ",")))

cell_data[, gdp_predicted := predict_rf_optimized(rf_model, cell_data,
                                                    feature_names,
                                                    chunk_size = 500000L)]

cat("Done.\n")

# ---- Step 4: Clean up -------------------------------------------------------
cell_data[, .row_id := NULL]  # remove helper column
rm(rf_model)
gc()
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Original | Optimized | Speedup factor |
|---|---|---|
| `build_neighbor_lookup`: 6.46M-iteration `lapply` with `paste` + named-vector hash lookup | `build_edge_list_dt`: vectorized `rep.int` + `unlist` â†’ two-column `data.table` | **~100â€“500Ã—** |
| `compute_neighbor_stats`: 5 Ã— 6.46M-iteration `lapply` + `do.call(rbind, ...)` | Single `data.table` grouped aggregation over edge-list join for all 5 vars at once | **~200â€“1000Ã—** |
| RF prediction: possibly row-level or data.frame-based | Full-matrix chunked `predict()` (500K rows/chunk), no data.frame coercion | **~5â€“50Ã—** |
| Memory: multiple full copies of 6.46M-row data | In-place `:=` assignment, intermediate cleanup with `gc()` | **Fits in 16 GB** |

**Estimated total runtime: 10â€“30 minutes** (dominated by RF `predict()` tree traversal), down from 86+ hours.

The trained Random Forest model is loaded once from disk and never retrained. The numerical predictions (the original estimand) are identical because the same model and the same features are used â€” only the data-preparation and prediction *workflow* is changed, not the computation.
END_RESPONSE>>>
