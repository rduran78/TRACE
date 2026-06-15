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
run_id: Anthropic_F03_T00_R001
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 1
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_001.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M â€” each lookup is O(n) hash probe on a very large vector)
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations Ã— multiple string allocations and hash lookups against a 6.46M-entry named vector. This alone can take many hours.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M small vectors into a matrix â€” this is notoriously slow because `rbind` on a list of vectors has quadratic-like overhead in base R.

**Outer loop:** This is called 5 times (once per neighbor source variable), so `compute_neighbor_stats` runs 5 Ã— 6.46M = ~32.3M iterations total.

### 1.2 Prediction Bottleneck

With ~6.46M rows and ~110 predictors, calling `predict()` on a Random Forest in one shot can:
- Require the entire prediction matrix (~6.46M Ã— 110 â‰ˆ 5.4 GB as double) plus the model object in memory simultaneously.
- On a 16 GB laptop, this risks swapping to disk.
- If `predict` is called row-by-row or in very small batches (not shown but implied by "prediction-loop"), overhead is catastrophic.

### 1.3 Summary of Root Causes

| Bottleneck | Root Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M `paste`/named-vector lookups in R loop | ~hours |
| `compute_neighbor_stats` | 6.46M R-level iterations + `do.call(rbind, ...)` | ~hours Ã— 5 vars |
| Object copying | `cell_data` is modified in a `for` loop â†’ potential full-copy each iteration | ~minutesâ€“hours |
| Prediction | Possible row-level predict loop; memory pressure from full matrix | ~hours |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorize with `data.table`

- Replace the named-vector key lookups with `data.table` integer-keyed joins.
- Replace the per-row `lapply` in `compute_neighbor_stats` with a single grouped `data.table` aggregation: explode the neighbor lookup into an edge-list `(row_i, neighbor_row_j)`, join the variable values, and group-by `row_i` to compute max/min/mean in one vectorized pass.
- Eliminate `do.call(rbind, ...)` entirely.
- Eliminate repeated copying of `cell_data` by adding columns in-place via `:=`.

**Expected speedup:** From hours to minutes.

### 2.2 Prediction â€” Batched, Memory-Efficient

- Call `predict()` in moderately sized chunks (e.g., 500K rows) to keep peak memory well under 16 GB.
- Avoid any row-level loop.
- Pre-allocate the output vector.

**Expected speedup:** From hours to minutes.

---

## 3. WORKING R CODE

```r
# =============================================================================
# DEPENDENCIES
# =============================================================================
library(data.table)
library(randomForest) # or library(ranger) â€” adapt predict call as needed

# =============================================================================
# 3A. OPTIMIZED NEIGHBOR LOOKUP â€” returns an edge-list data.table
#     instead of a list-of-vectors
# =============================================================================
build_neighbor_edgelist <- function(data_dt, id_order, neighbors) {
 # data_dt : data.table with columns 'id' and 'year' (and all feature cols)
 #           must have a column 'row_idx' = 1:.N  (added below if missing)
 # id_order: integer vector; position k â†’ cell id at position k in nb object
 # neighbors: spdep nb object (list of integer vectors of neighbor positions)

  if (!"row_idx" %in% names(data_dt)) {
    data_dt[, row_idx := .I]
  }

  # --- Step 1: map cell-id â†’ position in id_order (vectorised) ---
  id_to_pos <- data.table(id = id_order, pos = seq_along(id_order))

  # --- Step 2: explode the nb object into a cell-level edge list ---
  #     (pos_from, pos_to)  â€” positions in id_order
  from_pos <- rep(seq_along(neighbors), lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)
  nb_edges <- data.table(pos_from = from_pos, pos_to = to_pos)

  # Map positions back to cell ids
  nb_edges[, id_from := id_order[pos_from]]
  nb_edges[, id_to   := id_order[pos_to]]
  nb_edges[, c("pos_from", "pos_to") := NULL]

  # --- Step 3: for every (id_from, year) row, find the row_idx of each
  #     neighbor (id_to, year) ---
  # Key the data for fast join
  row_key <- data_dt[, .(id, year, row_idx)]
  setkey(row_key, id, year)

  # Expand edges by year: join source rows to get year + row_idx of source
  src <- data_dt[, .(id_from = id, year, src_row = row_idx)]
  setkey(src, id_from)

  # For each source row, attach its neighbor cell ids
  # This is a many-to-many join: each src row Ã— its neighbors
  setkey(nb_edges, id_from)
  edge_year <- nb_edges[src, on = "id_from", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: id_from, id_to, year, src_row

  # Join to find the row_idx of the neighbor in the same year
  setkey(edge_year, id_to, year)
  setkey(row_key, id, year)
  edge_year[row_key, nbr_row := i.row_idx, on = c(id_to = "id", "year")]

  # Drop edges where the neighbor-year row doesn't exist
 edge_year <- edge_year[!is.na(nbr_row)]

  # Return slim edge list: (src_row, nbr_row)
  edge_year[, .(src_row, nbr_row)]
}

# =============================================================================
# 3B. OPTIMIZED NEIGHBOR STATS â€” fully vectorised via data.table grouping
# =============================================================================
compute_neighbor_stats_vec <- function(data_dt, edge_dt, var_name) {
  # edge_dt: data.table with columns src_row, nbr_row
  # Attach the neighbor's value
  vals <- data_dt[[var_name]]
  work <- edge_dt[, .(src_row, nbr_val = vals[nbr_row])]
  work <- work[!is.na(nbr_val)]

  stats <- work[, .(
    nb_max  = max(nbr_val),
    nb_min  = min(nbr_val),
    nb_mean = mean(nbr_val)
  ), by = src_row]

  # Build full-length columns (NA for rows with no valid neighbors)
  n <- nrow(data_dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)
  out_max[stats$src_row]  <- stats$nb_max
  out_min[stats$src_row]  <- stats$nb_min
  out_mean[stats$src_row] <- stats$nb_mean

  list(nb_max = out_max, nb_min = out_min, nb_mean = out_mean)
}

# =============================================================================
# 3C. FULL FEATURE-PREPARATION PIPELINE
# =============================================================================
prepare_features <- function(cell_data, id_order, rook_neighbors_unique) {
  # Convert to data.table in place (no copy if already data.table)
  if (!is.data.table(cell_data)) {
    cell_data <- as.data.table(cell_data)
  }
  cell_data[, row_idx := .I]

  message("Building neighbor edge list â€¦")
  edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
  setkey(edge_dt, src_row)
  message(sprintf("  Edge list: %s edges", format(nrow(edge_dt), big.mark = ",")))

  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

  for (var_name in neighbor_source_vars) {
    message(sprintf("Computing neighbor stats for '%s' â€¦", var_name))
    stats <- compute_neighbor_stats_vec(cell_data, edge_dt, var_name)

    # Add columns in-place (no copy of the whole table)
    set(cell_data, j = paste0(var_name, "_nb_max"),  value = stats$nb_max)
    set(cell_data, j = paste0(var_name, "_nb_min"),  value = stats$nb_min)
    set(cell_data, j = paste0(var_name, "_nb_mean"), value = stats$nb_mean)
  }

  cell_data[, row_idx := NULL]
  cell_data
}

# =============================================================================
# 3D. BATCHED RANDOM FOREST PREDICTION
# =============================================================================
predict_rf_batched <- function(model, newdata, batch_size = 500000L) {
  # model   : pre-trained randomForest / ranger model (loaded from disk)
  # newdata : data.table / data.frame of predictor columns only
  # Returns : numeric vector of predictions, same length as nrow(newdata)

  n <- nrow(newdata)
  preds <- numeric(n)  # pre-allocate

  starts <- seq(1L, n, by = batch_size)
  message(sprintf("Predicting %s rows in %d batches â€¦",
                  format(n, big.mark = ","), length(starts)))

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + batch_size - 1L, n)
    batch <- newdata[i1:i2, , drop = FALSE]

    # --- adapt this block to your model class ---
    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = batch)$predictions
    } else {
      # randomForest, or caret-wrapped RF
      preds[i1:i2] <- predict(model, newdata = batch)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("  Batch %d / %d done (rows %sâ€“%s)",
                      k, length(starts),
                      format(i1, big.mark = ","),
                      format(i2, big.mark = ",")))
    }
    # Free batch memory explicitly
    rm(batch); gc(verbose = FALSE)
  }

  preds
}

# =============================================================================
# 3E. MAIN PIPELINE
# =============================================================================
run_pipeline <- function(cell_data_path, model_path, id_order, rook_neighbors_unique,
                         predictor_cols, output_path) {
  # --- Load data ---
  message("Loading cell data â€¦")
  cell_data <- fread(cell_data_path)   # or readRDS / qs::qread

  # --- Feature preparation ---
  cell_data <- prepare_features(cell_data, id_order, rook_neighbors_unique)

  # --- Load pre-trained model (once) ---
  message("Loading Random Forest model â€¦")
  model <- readRDS(model_path)

  # --- Prepare prediction matrix ---
  pred_data <- cell_data[, ..predictor_cols]  # data.table column subset, no copy

  # --- Predict in batches ---
  cell_data[, predicted_gdp := predict_rf_batched(model, pred_data)]

  # --- Write results ---
  message("Writing results â€¦")
  fwrite(cell_data, output_path)
  message("Done.")
  invisible(cell_data)
}
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Stage | Original | Optimized | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M R-loop iterations with string ops) | ~1â€“3 min (vectorised `data.table` joins) | Eliminates per-row `paste`, named-vector lookup |
| `compute_neighbor_stats` (Ã—5 vars) | ~hours (32.3M R iterations + `do.call(rbind)`) | ~2â€“5 min total (grouped `data.table` aggregation) | Single vectorised pass per variable; no `rbind` |
| Column binding / object copy | Repeated full-copy of 6.46M Ã— 110 data.frame | Zero-copy `set()` in-place | `data.table` reference semantics |
| RF prediction | Potentially hours (row-loop or single OOM call) | ~5â€“20 min (batched, memory-safe) | Avoids swap; avoids per-row overhead |
| **Total** | **~86+ hours** | **~15â€“30 minutes** | **~170â€“340Ã— speedup** |

### Key Principles Applied

1. **Vectorisation over iteration:** Every per-row `lapply` is replaced by a `data.table` join or grouped aggregation that runs in C.
2. **Edge-list representation:** The neighbor lookup is materialized once as a two-column integer edge list (~38M rows for the full panel), enabling all subsequent variable computations to reuse it without rebuilding keys.
3. **In-place mutation:** `data.table::set()` adds columns without triggering R's copy-on-modify, saving ~5.4 GB of redundant copies per iteration.
4. **Batched prediction:** Keeps peak memory well within 16 GB while avoiding per-row `predict()` overhead.
5. **Model preservation:** The trained Random Forest is loaded from disk and used read-only; no retraining occurs. The numerical estimand (predicted GDP) is computed identically â€” only the surrounding data-handling code changes.
END_RESPONSE>>>
