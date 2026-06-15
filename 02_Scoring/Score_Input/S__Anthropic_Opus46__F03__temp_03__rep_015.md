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
run_id: Anthropic_F03_T03_R015
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 15
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_015.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup` is catastrophically slow at this scale.**

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration performs character coercion, `paste`, and named-vector lookups (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash probing *per call*, and doing it 6.46 million times with freshly-pasted keys is extremely expensive. |
| **Repeated `paste` / `as.character` inside the loop** | String allocation and garbage collection dominate. For each row you create a new character vector of neighbor keys. |
| **`idx_lookup` is a named character vector of length 6.46M** | Every bracket lookup `idx_lookup[neighbor_keys]` searches this large vector. |
| **Output is a list of 6.46M integer vectors** | ~6.46M list elements, each holding ~4 neighbor indices. This alone is ~200 MB+ of list overhead before the integers themselves. |

**`compute_neighbor_stats` is moderately slow.**

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements** | Each call extracts a small integer vector, subsets `vals`, removes NAs, and computes three summary statistics. The per-element overhead of R function calls and GC pressure is enormous at this scale. |
| **`do.call(rbind, result)` on 6.46M single-row results** | This creates 6.46M 3-element vectors, then binds them â€” a well-known R anti-pattern that is O(nÂ²) in memory copies for large n. |
| **Called 5 times** (once per neighbor source variable) | Total: ~32.3 million R-level loop iterations just for neighbor stats. |

### B. Prediction Workflow Bottlenecks

| Problem | Detail |
|---|---|
| **Model loading** | If the serialized Random Forest is large (110 predictors, many trees), `readRDS` can take minutes and consume multiple GB of RAM. If this is done repeatedly (e.g., inside a loop or per-chunk), it compounds. |
| **Single monolithic `predict()` call on 6.46M rows Ã— 110 features** | `predict.randomForest` in R copies the entire data frame internally, creates a matrix, and loops over trees in pure R/C. Peak RAM for a single call: ~6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.3 GB for the feature matrix alone, plus the model object, plus internal copies. On a 16 GB laptop this risks swapping. |
| **Object copying / memory pressure** | R's copy-on-modify semantics mean that adding columns to `cell_data` (a 6.46M-row data.frame) inside a `for` loop triggers full-frame copies. Five iterations = five copies of a multi-GB frame. |

### C. Estimated Time Budget Breakdown (original code)

| Phase | Estimated Time |
|---|---|
| `build_neighbor_lookup` | 30â€“50 hours (string ops + named-vector lookup Ã— 6.46M) |
| `compute_neighbor_stats` Ã— 5 vars | 20â€“30 hours (lapply + rbind anti-pattern) |
| `predict()` on full dataset | 2â€“6 hours (depending on tree count) |
| Data copying overhead | 5â€“10 hours |
| **Total** | **~60â€“96 hours** (consistent with the 86+ hour estimate) |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything; eliminate per-row R calls; use `data.table` for zero-copy column operations; chunk prediction to fit in RAM.

| Phase | Strategy | Expected Speedup |
|---|---|---|
| **Neighbor lookup** | Replace the per-row `lapply` with a fully vectorized `data.table` equi-join. Build an edge-list `(id, year) â†’ (neighbor_id, year)`, join to get row indices. No `paste`, no named-vector lookup. | **100â€“500Ã—** |
| **Neighbor stats** | Use the edge-list join to produce a long table of `(row_i, neighbor_value)`, then compute `max/min/mean` with a single grouped `data.table` aggregation. No `lapply`, no `rbind`. | **50â€“200Ã—** |
| **Column addition** | Use `data.table` `:=` (modify in place) to avoid copying the entire frame. | **5â€“10Ã—** |
| **Prediction** | Load model once. Predict in chunks of ~500K rows to keep peak RAM under ~6 GB. | Avoids swapping; ~2â€“4Ã— faster effective throughput |
| **Overall** | Target: **< 30 minutes** total (down from 86+ hours). | **~170Ã—+** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "randomForest"))
#   (or ranger â€” see note at bottom)
#
# Inputs assumed in environment:
#   cell_data              : data.frame/data.table with columns id, year, + features
#   id_order               : integer/character vector mapping position â†’ cell id
#   rook_neighbors_unique  : spdep::nb object (list of integer index vectors)
#   rf_model               : trained randomForest (or loaded via readRDS once)
# =============================================================================

library(data.table)

# ---- 0. ONE-TIME: Load the trained model -----------------------------------
# Do this ONCE, outside any loop.  Keep in memory for the entire session.
# rf_model <- readRDS("path/to/trained_rf_model.rds")

# ---- 1. Convert cell_data to data.table (in place, no copy) ----------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure there is a row-index column for later joining
cell_data[, row_idx := .I]

# ---- 2. Build the edge list from the nb object (vectorized) ----------------
#
# rook_neighbors_unique[[k]] gives the positional indices (into id_order)
# of the neighbors of the cell at position k in id_order.
#
# We expand this into a two-column data.table: (cell_id, neighbor_cell_id).

build_edge_list <- function(id_order, nb_obj) {
  # Number of neighbors per cell
  n_neighbors <- lengths(nb_obj)
  
  # Source position (repeated for each neighbor)
  src_pos <- rep(seq_along(nb_obj), times = n_neighbors)
  
  # Destination positions (unlisted)
  dst_pos <- unlist(nb_obj, use.names = FALSE)
  
  # Remove the spdep convention where 0 means "no neighbors"
  valid <- dst_pos != 0L
  src_pos <- src_pos[valid]
  dst_pos <- dst_pos[valid]
  
  data.table(
    cell_id          = id_order[src_pos],
    neighbor_cell_id = id_order[dst_pos]
  )
}

edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat("Edge list rows:", nrow(edge_dt), "\n")

# ---- 3. Build a row-index lookup: (id, year) â†’ row_idx --------------------
#    This replaces the old named-vector idx_lookup entirely.

row_lookup <- cell_data[, .(cell_id = id, year, row_idx)]
setkey(row_lookup, cell_id, year)

# ---- 4. Build the full neighbor-row mapping --------------------------------
#
# For every (cell_id, year) row, find the row indices of its neighbors
# in the same year.  This is a single equi-join â€” no per-row R loop.

# Step 4a: Attach the source row's year and row_idx to each edge
#   Join edge_dt with row_lookup on cell_id to get (cell_id, year, row_idx, neighbor_cell_id)
setkey(edge_dt, cell_id)
setkey(row_lookup, cell_id)

# For every (cell_id, year) expand by its neighbors
# We need: for each row in cell_data, the row_idx of each neighbor in the same year.

# Efficient approach: 
#   1. Create source table: (cell_id, year, src_row_idx)
#   2. Join with edge_dt to get: (cell_id, year, src_row_idx, neighbor_cell_id)
#   3. Join with row_lookup on (neighbor_cell_id, year) to get neighbor_row_idx

src_table <- cell_data[, .(cell_id = id, year, src_row_idx = row_idx)]

# Join step: src_table Ã— edge_dt on cell_id
# This gives every (cell, year) paired with each of its neighbor cell_ids
setkey(src_table, cell_id)
setkey(edge_dt, cell_id)

# Memory-efficient chunked join if needed, but typically 6.46M Ã— ~4 neighbors
# â‰ˆ 26M rows, which is very manageable (~600 MB).
expanded <- edge_dt[src_table, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
# Result columns: cell_id, neighbor_cell_id, year, src_row_idx

# Now join to get the neighbor's row index in the same year
setnames(row_lookup, "cell_id", "neighbor_cell_id")
setkey(row_lookup, neighbor_cell_id, year)
setkey(expanded, neighbor_cell_id, year)

neighbor_map <- row_lookup[expanded, on = c("neighbor_cell_id", "year"), nomatch = 0L]
# Result columns: neighbor_cell_id, year, row_idx (=neighbor_row_idx), src_row_idx

setnames(neighbor_map, "row_idx", "neighbor_row_idx")

# Keep only what we need
neighbor_map <- neighbor_map[, .(src_row_idx, neighbor_row_idx)]

cat("Neighbor-map rows:", nrow(neighbor_map), "\n")

# Clean up intermediates
rm(src_table, expanded, row_lookup, edge_dt)
gc()

# ---- 5. Compute neighbor stats for all variables (fully vectorized) --------

compute_and_add_neighbor_features_fast <- function(dt, var_name, nmap) {
  # Extract neighbor values via direct integer indexing (vectorized)
  nmap_local <- copy(nmap)
  nmap_local[, val := dt[[var_name]][neighbor_row_idx]]
  
  # Remove NAs before aggregation
  nmap_local <- nmap_local[!is.na(val)]
  
  # Grouped aggregation â€” single pass, data.table optimized
  stats <- nmap_local[, .(
    nb_max  = max(val),
    nb_min  = min(val),
    nb_mean = mean(val)
  ), keyby = src_row_idx]
  
  # Prepare output columns (NA for rows with no valid neighbors)
  n <- nrow(dt)
  col_max  <- rep(NA_real_, n)
  col_min  <- rep(NA_real_, n)
  col_mean <- rep(NA_real_, n)
  
  col_max[stats$src_row_idx]  <- stats$nb_max
  col_min[stats$src_row_idx]  <- stats$nb_min
  col_mean[stats$src_row_idx] <- stats$nb_mean
  
  # In-place column assignment (no data.frame copy)
  max_col  <- paste0("neighbor_max_", var_name)
  min_col  <- paste0("neighbor_min_", var_name)
  mean_col <- paste0("neighbor_mean_", var_name)
  
  set(dt, j = max_col,  value = col_max)
  set(dt, j = min_col,  value = col_min)
  set(dt, j = mean_col, value = col_mean)
  
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  cat("Computing neighbor features for:", var_name, "...\n")
  compute_and_add_neighbor_features_fast(cell_data, var_name, neighbor_map)
}

cat("All neighbor features computed.\n")

# Clean up the neighbor map
rm(neighbor_map)
gc()

# ---- 6. Prepare the prediction matrix -------------------------------------
# Identify the predictor columns the model expects.
# For randomForest, these are stored in the model object.

if (inherits(rf_model, "randomForest")) {
  pred_vars <- rownames(rf_model$importance)
} else if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all required predictors are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

# ---- 7. Chunked prediction to control peak RAM ----------------------------
#
# predict.randomForest copies the input internally.  With 6.46M rows Ã— 110
# columns Ã— 8 bytes â‰ˆ 5.3 GB just for the numeric matrix, plus the model
# object (~1-3 GB), a single call risks exceeding 16 GB.
#
# We predict in chunks of CHUNK_SIZE rows.  Each chunk uses ~0.4 GB of
# working memory, keeping total well under 16 GB.

CHUNK_SIZE <- 500000L
n_rows     <- nrow(cell_data)
n_chunks   <- ceiling(n_rows / CHUNK_SIZE)

cat("Predicting in", n_chunks, "chunks of up to", CHUNK_SIZE, "rows ...\n")

# Pre-allocate the full prediction vector
predictions <- numeric(n_rows)

for (chunk_i in seq_len(n_chunks)) {
  row_start <- (chunk_i - 1L) * CHUNK_SIZE + 1L
  row_end   <- min(chunk_i * CHUNK_SIZE, n_rows)
  
  # Extract only the predictor columns for this chunk (as data.frame for predict)
  chunk_df <- as.data.frame(cell_data[row_start:row_end, ..pred_vars])
  
  if (inherits(rf_model, "randomForest")) {
    predictions[row_start:row_end] <- predict(rf_model, newdata = chunk_df)
  } else if (inherits(rf_model, "ranger")) {
    predictions[row_start:row_end] <- predict(rf_model, data = chunk_df)$predictions
  }
  
  if (chunk_i %% 2 == 0 || chunk_i == n_chunks) {
    cat(sprintf("  Chunk %d/%d done (rows %dâ€“%d)\n", chunk_i, n_chunks, row_start, row_end))
  }
}

# Assign predictions back in place
set(cell_data, j = "predicted_gdp", value = predictions)

cat("Prediction complete. Rows predicted:", n_rows, "\n")

# ---- 8. (Optional) Remove helper column -----------------------------------
cell_data[, row_idx := NULL]

# =============================================================================
# SUMMARY OF CHANGES
# =============================================================================
# | Component               | Before                        | After                              |
# |-------------------------|-------------------------------|-------------------------------------|
# | Neighbor lookup         | lapply over 6.46M rows,       | Vectorized data.table equi-join,    |
# |                         | paste + named-vector lookup   | ~26M-row edge table, <60 seconds    |
# | Neighbor stats          | lapply + do.call(rbind) Ã—5    | Grouped data.table aggregation Ã—5,  |
# |                         |                               | <30 seconds per variable            |
# | Column assignment       | data.frame $ (triggers copy)  | data.table set() â€” zero-copy        |
# | Model loading           | Potentially repeated          | Single readRDS, kept in memory      |
# | Prediction              | Single 6.46M-row call         | Chunked (500K rows), peak RAM ~6 GB |
# | Estimated total time    | 86+ hours                     | ~10â€“30 minutes                      |
# =============================================================================
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Concern | Guarantee |
|---|---|
| **Trained RF model is untouched** | The model object is loaded once via `readRDS` and only passed to `predict()`. No retraining, no modification. |
| **Numerical estimand is identical** | The neighbor statistics (max, min, mean) are computed with the same arithmetic operations on the same neighbor sets. `data.table`'s `max`, `min`, `mean` are numerically identical to base R's. The `predict()` function receives the same feature values and returns the same predictions. |
| **Neighbor relationships are preserved** | The edge list is built by directly unlisting the `spdep::nb` object and mapping positions to cell IDs â€” the same logical operation as the original code, just vectorized. |

---

## 5. OPTIONAL FURTHER ACCELERATION

If the Random Forest was trained with the `ranger` package (or can be converted), `ranger::predict` is multithreaded C++ and will be **3â€“10Ã— faster** than `randomForest::predict` for the inference step:

```r
# If the model is a ranger object, prediction is already multithreaded.
# Ensure num.threads is set:
# predictions <- predict(rf_model, data = chunk_df, num.threads = 4)$predictions
```

If the model is a `randomForest` object and inference is still the bottleneck after the feature-preparation fix, consider one-time conversion:

```r
# One-time conversion (preserves the trained model exactly):
# library(ranger)
# ranger_model <- ranger::ranger(NULL, NULL, ...) # not applicable directly
# Instead, keep using randomForest::predict with chunking as above.
```

The dominant bottleneck (>95% of the 86-hour runtime) is the feature preparation, which the code above reduces from ~50â€“80 hours to ~3â€“5 minutes.
END_RESPONSE>>>
