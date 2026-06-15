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
run_id: Anthropic_F03_T10_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy & Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and `NA` filtering â€” all inside an interpreted R loop. |
| **Named-vector lookups (`idx_lookup[neighbor_keys]`)** | R's named-vector indexing is O(n) hashing per call; doing this 6.46 M times on a vector of length 6.46 M is extremely slow. |
| **`paste` inside the loop** | Creates ~6.46 M Ã— (avg ~4 neighbors) â‰ˆ 25 M temporary strings inside a loop. |
| **Output is a list of 6.46M integer vectors** | Each small vector is a separate R object â€” enormous allocation overhead (~200â€“500 MB just in SEXP headers). |

**`compute_neighbor_stats`** is the second major bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46 M elements** | Pure R loop with per-element subsetting, `NA` removal, and three summary stats. |
| **Called 5 times** (once per source variable) | 5 Ã— 6.46 M = 32.3 M R-level iterations. |
| **`do.call(rbind, result)` on 6.46 M rows** | Binds 6.46 M small 3-element vectors into a matrix â€” notoriously slow. |

### 1.2 Prediction-Workflow Bottlenecks

| Problem | Detail |
|---|---|
| **Model loading** | If the Random Forest is re-loaded from disk per chunk or per year, deserialization of a large RF object (often 1â€“4 GB) dominates wall time. Load **once**. |
| **`predict()` on full 6.46 M rows Ã— 110 features** | `ranger::predict` and `randomForest::predict` both need a contiguous `data.frame`/`matrix`. If the data is a `data.frame` with 110 columns, R copies it internally. |
| **Object copying / COW triggers** | Any in-place column addition to `cell_data` (a 6.46 M Ã— 110+ `data.frame`) triggers full-frame copy-on-write in base R. Five iterations of `compute_and_add_neighbor_features` â†’ up to 5 full copies â‰ˆ 5 Ã— ~5.7 GB = ~28 GB of transient allocation on a 16 GB machine â†’ swapping â†’ hours of I/O. |
| **Memory pressure** | 6.46 M Ã— 110 doubles â‰ˆ 5.7 GB. Neighbor lookup list â‰ˆ 0.5 GB. RF model â‰ˆ 1â€“4 GB. On 16 GB RAM, every unnecessary copy causes swap death. |

### 1.3 Estimated Time Attribution (86+ hours baseline)

| Phase | Est. share |
|---|---|
| `build_neighbor_lookup` | ~15â€“25 % |
| `compute_neighbor_stats` Ã— 5 | ~30â€“40 % |
| Data-frame COW copies | ~15â€“20 % |
| `predict()` (if poorly batched) | ~10â€“20 % |
| Model re-loading (if repeated) | variable |

---

## 2. OPTIMIZATION STRATEGY

### Principle: vectorize everything, eliminate R loops, use `data.table` for zero-copy column addition, batch prediction.

| Strategy | Technique | Speedup factor |
|---|---|---|
| Replace `build_neighbor_lookup` list with a **flat edge table** (`data.table`) | Pre-join cell-id Ã— year to row indices using `data.table` keyed joins â€” fully vectorized. | 50â€“200Ã— |
| Replace `compute_neighbor_stats` per-row `lapply` with **grouped `data.table` aggregation** | One `dt[, .(max, min, mean), by = row_idx]` call â€” C-level grouped ops. | 100â€“500Ã— |
| Switch `cell_data` to **`data.table`** | Column addition via `:=` is in-place â€” **no COW copies**. | Saves ~28 GB allocation |
| **Load RF model once**, predict in **batched chunks** | Avoid repeated `readRDS`; chunk prediction to stay within RAM. | Eliminates redundant I/O |
| Use `ranger` if possible (column-major prediction, C++ back-end) | `ranger::predict` is 5â€“10Ã— faster than `randomForest::predict` on large data. | 5â€“10Ã— on predict |

**Expected total runtime: ~5â€“20 minutes** (vs. 86+ hours), depending on RF model size and disk speed.

---

## 3. WORKING R CODE

```r
# ==============================================================================
# 0. LIBRARIES
# ==============================================================================
library(data.table)

# ==============================================================================
# 1. ONE-TIME SETUP: Load model and convert data
# ==============================================================================

# --- Load the trained RF model ONCE ------------------------------------------
# Adjust path and object name to your setup.
# If it is a `ranger` model, predict() below works the same way.
rf_model <- readRDS("trained_rf_model.rds")

# --- Convert cell_data to data.table IN PLACE --------------------------------
# This avoids a full copy if cell_data is already a data.frame.
setDT(cell_data)

# Ensure id and year columns are integer for fast keyed joins
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Add a sequential row index (used to map predictions back)
cell_data[, .row_idx := .I]


# ==============================================================================
# 2. BUILD FLAT NEIGHBOR EDGE TABLE (vectorized, replaces build_neighbor_lookup)
# ==============================================================================
build_neighbor_edges <- function(cell_data, id_order, neighbors) {
  # id_order  : integer vector, length N_cells (344,208)
  # neighbors : spdep nb object â€” list of integer index vectors into id_order
  #
  # Returns: data.table with columns  [focal_id, neighbor_id]
  #          where both are cell IDs (not positional indices).

  n <- length(neighbors)
  # Pre-compute total edges for single allocation
  n_edges <- sum(lengths(neighbors))

  focal_idx    <- rep.int(seq_len(n), lengths(neighbors))
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
}

cat("Building flat neighbor edge table...\n")
edge_dt <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)

# ==============================================================================
# 3. BUILD FULL NEIGHBOR-PAIR LOOKUP (join edges Ã— years, vectorised)
# ==============================================================================
# For each (focal cell-year row) we need the ROW INDICES of its neighbors
# in the SAME year.

cat("Building neighbor-row lookup via keyed join...\n")

# Keyed index: cell id + year â†’ row index in cell_data
idx_dt <- cell_data[, .(id, year, .row_idx)]
setkey(idx_dt, id, year)

# Expand edges to every year present in the data
# (all cells share the same year set, so cross-join edge pairs Ã— years)
years_vec <- sort(unique(cell_data$year))

# Instead of a massive cross-join (edges Ã— years), we join via the data itself.
# Step A: For every row, get its neighbor cell IDs.
focal_rows <- cell_data[, .(focal_row = .row_idx, focal_id = id, year)]
setkey(focal_rows, focal_id)
setkey(edge_dt, focal_id)

# Merge: for each focal row, attach all its neighbor_ids
# This produces one row per (focal_row, neighbor_id) pair, sharing the year.
pair_dt <- edge_dt[focal_rows, on = "focal_id", allow.cartesian = TRUE,
                   nomatch = NULL]
# pair_dt columns: focal_id, neighbor_id, focal_row, year

# Step B: Look up the ROW INDEX of each neighbor in the same year.
setkey(pair_dt, neighbor_id, year)
setkey(idx_dt, id, year)

pair_dt[idx_dt, neighbor_row := i..row_idx,
        on = .(neighbor_id = id, year = year)]

# Drop pairs where the neighbor has no data for that year
pair_dt <- pair_dt[!is.na(neighbor_row)]

# Keep only what we need
pair_dt <- pair_dt[, .(focal_row, neighbor_row)]
setkey(pair_dt, focal_row)

cat(sprintf("  Neighbor-pair table: %s rows\n", format(nrow(pair_dt), big.mark = ",")))

# ==============================================================================
# 4. FAST GROUPED NEIGHBOR STATISTICS (replaces compute_neighbor_stats)
# ==============================================================================
compute_and_add_neighbor_features_fast <- function(cell_dt, var_name, pair_dt) {
  # Extracts neighbor values, computes max/min/mean grouped by focal row,
  # and adds three new columns to cell_dt BY REFERENCE (no copy).

  col_max  <- paste0("neighbor_max_",  var_name)
  col_min  <- paste0("neighbor_min_",  var_name)
  col_mean <- paste0("neighbor_mean_", var_name)

  # Pull the variable values for all neighbor rows (vectorized)
  vals <- cell_dt[[var_name]][pair_dt$neighbor_row]

  # Build a small data.table for grouped aggregation
  agg_dt <- data.table(focal_row = pair_dt$focal_row, val = vals)

  # Remove NAs before aggregation
  agg_dt <- agg_dt[!is.na(val)]

  # Grouped aggregation â€” executed at C level inside data.table
  stats <- agg_dt[, .(vmax = max(val), vmin = min(val), vmean = mean(val)),
                  keyby = focal_row]

  # Initialize columns to NA, then update matched rows BY REFERENCE
  n <- nrow(cell_dt)
  set(cell_dt, j = col_max,  value = rep(NA_real_, n))
  set(cell_dt, j = col_min,  value = rep(NA_real_, n))
  set(cell_dt, j = col_mean, value = rep(NA_real_, n))

  matched <- stats$focal_row
  set(cell_dt, i = matched, j = col_max,  value = stats$vmax)
  set(cell_dt, i = matched, j = col_min,  value = stats$vmin)
  set(cell_dt, i = matched, j = col_mean, value = stats$vmean)

  invisible(cell_dt)
}

# --- Run for all 5 neighbor source variables ----------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Computing neighbor features for: %s\n", var_name))
  compute_and_add_neighbor_features_fast(cell_data, var_name, pair_dt)
}

cat("Neighbor feature engineering complete.\n")

# Free the large pair table if memory is tight
# rm(pair_dt, edge_dt, focal_rows, idx_dt, agg_dt); gc()


# ==============================================================================
# 5. BATCHED RANDOM FOREST PREDICTION (memory-safe, single model load)
# ==============================================================================
# Identify the predictor columns the model expects.
# For ranger:   rf_model$forest$independent.variable.names
# For randomForest: names which(rf_model$forest$ncat > 0)) or colnames(rf_model$forest$xbestsplit)
# Adjust the line below to your model type:

if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else {
  # randomForest â€” predictors are stored in the model's xlevels or can be

  # inferred from the training call. Adjust if needed:
  pred_vars <- attr(rf_model$terms, "term.labels")
  if (is.null(pred_vars)) {
    pred_vars <- rownames(rf_model$importance)
  }
}

cat(sprintf("Predicting with %d features across %s rows...\n",
            length(pred_vars), format(nrow(cell_data), big.mark = ",")))

# --- Batch prediction to limit peak memory -----------------------------------
BATCH_SIZE <- 500000L
n_rows     <- nrow(cell_data)
n_batches  <- ceiling(n_rows / BATCH_SIZE)

# Pre-allocate predictions vector
cell_data[, predicted_gdp := NA_real_]

for (b in seq_len(n_batches)) {
  start_i <- (b - 1L) * BATCH_SIZE + 1L
  end_i   <- min(b * BATCH_SIZE, n_rows)
  cat(sprintf("  Batch %d/%d  (rows %s â€“ %s)\n",
              b, n_batches,
              format(start_i, big.mark = ","),
              format(end_i,   big.mark = ",")))

  # Extract ONLY the predictor columns for the batch (minimises memory)
  batch_df <- as.data.frame(cell_data[start_i:end_i, ..pred_vars])

  if (inherits(rf_model, "ranger")) {
    preds <- predict(rf_model, data = batch_df)$predictions
  } else {
    preds <- predict(rf_model, newdata = batch_df)
  }

  set(cell_data, i = start_i:end_i, j = "predicted_gdp", value = preds)

  # Immediately free batch memory

  rm(batch_df, preds)
  if (b %% 5 == 0) gc()
}

cat("Prediction complete.\n")

# ==============================================================================
# 6. CLEAN UP HELPER COLUMN
# ==============================================================================
cell_data[, .row_idx := NULL]
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Concern | Guarantee |
|---|---|
| **Trained RF model unchanged** | The model object is loaded once with `readRDS` and only its `predict()` method is called â€” no retraining, no parameter changes. |
| **Numerical estimand preserved** | The neighbor features are computed with the **identical formulas** (`max`, `min`, `mean` of non-NA neighbor values), just vectorized via `data.table` grouped aggregation instead of R-level `lapply`. The floating-point results are bit-identical. |
| **Same neighbor definition** | The flat edge table is constructed directly from the original `rook_neighbors_unique` nb object and `id_order`, reproducing the same directed neighbor pairs. |

---

## 5. EXPECTED PERFORMANCE

| Phase | Before | After | Reason |
|---|---|---|---|
| `build_neighbor_lookup` | ~8â€“15 hrs | ~30â€“90 sec | Vectorized `data.table` keyed join replaces 6.46 M R-loop iterations |
| `compute_neighbor_stats` Ã— 5 | ~25â€“35 hrs | ~2â€“5 min | Grouped C-level aggregation replaces 32.3 M R-loop iterations; no `do.call(rbind, ...)` |
| COW copies of `cell_data` | ~15â€“20 hrs (swap) | **0** | `data.table` `:=` / `set()` modifies in place â€” no copy triggered |
| RF prediction | ~5â€“15 hrs | ~5â€“15 min | Single model load; batched `predict()`; only predictor columns extracted |
| **Total** | **86+ hours** | **~10â€“25 minutes** | |
END_RESPONSE>>>
