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
run_id: Anthropic_F03_T10_R013
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 13
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_013.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup` is catastrophically slow at this scale.**

| Problem | Root Cause | Impact |
|---|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, and named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookup in R is O(n) hash-probe per key, but the sheer volume (6.46M Ã— ~4 neighbors each â‰ˆ 26M lookups via character matching) is brutal. | Estimated 30â€“60 min alone |
| **Character key joins** (`paste(id, year)`) | Creates ~6.46M temporary character strings, then does named-vector indexingâ€”far slower than integer indexing or a proper hash table. | High memory churn + slow |
| **`compute_neighbor_stats` uses `lapply` + `do.call(rbind, ...)`** | 6.46M list elements each producing a length-3 vector, then bound with `do.call(rbind, ...)`. This copies the entire result matrix repeatedly. | Another 20â€“40 min per variable Ã— 5 variables |
| **Outer loop copies `cell_data` 5 times** | `cell_data <- compute_and_add_neighbor_features(...)` likely copies the full data.frame (6.46M Ã— 110+ cols â‰ˆ several GB) on each assignment. | Massive memory pressure, possible swapping |

### B. Random Forest Inference Bottlenecks

| Problem | Root Cause | Impact |
|---|---|---|
| **Single `predict()` call on 6.46M rows Ã— 110 features** | `ranger`/`randomForest` predict loads every tree and traverses every row. With 500 trees, this is ~3.2 billion tree-row traversals. For `randomForest` (R's default), this is single-threaded. | Could take 2â€“10+ hours |
| **If prediction is done in a loop (row-by-row or chunk-by-chunk without batching)** | Per-call overhead of `predict()` is non-trivial; calling it millions of times is disastrous. | Potentially the dominant cost |
| **Model object size** | A trained RF on 110 features with 500 trees can be 1â€“4 GB. If it's an R `randomForest` object (not `ranger`), it stores the full OOB data, proximity matrix, etc. | RAM contention with the 6.46M-row data |
| **`data.frame` conversion inside `predict`** | Many RF implementations internally coerce to matrix. If the input is a `data.frame`, this creates a full copy. | +5â€“10 GB transient allocation |

### C. Overall Memory Arithmetic

| Object | Estimated Size |
|---|---|
| `cell_data` (6.46M Ã— 110 cols, numeric) | ~5.3 GB |
| Neighbor lookup (list of 6.46M integer vectors) | ~0.5â€“1 GB |
| RF model | 1â€“4 GB |
| Prediction working copies | 2â€“5 GB transient |
| **Total** | **9â€“15 GB on a 16 GB machine** |

This means you are likely **swapping to disk**, which alone can explain the 86-hour runtime.

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything, minimize copies, batch predictions, use `data.table`.

| Layer | Current | Optimized |
|---|---|---|
| Data structure | `data.frame` | `data.table` (in-place `:=` assignment, no copies) |
| Neighbor lookup build | `lapply` over 6.46M rows with character keys | Vectorized merge via `data.table` keyed join |
| Neighbor stats | `lapply` + `do.call(rbind)` | Vectorized `data.table` group-by aggregation on exploded neighbor-edge table |
| RF prediction | Unknown (possibly row-level or `randomForest`) | Single batched `predict()` call; convert model to `ranger` if possible; pass matrix not data.frame |
| Memory | Repeated full-data copies | In-place column addition via `:=`; `gc()` strategically; chunked prediction if needed |

**Expected speedup: from 86+ hours to approximately 15â€“45 minutes.**

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# Preserves: trained RF model, original numerical estimand
# =============================================================================

library(data.table)

# ---- Step 0: Convert to data.table (once) -----------------------------------
# Assume cell_data is your data.frame, already loaded.
# This converts in-place (no deep copy if already a data.table).
setDT(cell_data)

# Ensure key columns are proper types
cell_data[, id := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a unique row index for fast joining
cell_data[, .row_idx := .I]


# ---- Step 1: Build exploded edge table (vectorized) -------------------------
# rook_neighbors_unique is an nb object: a list of length = # unique spatial cells.
# id_order is the vector mapping position in nb list -> cell id.
# rook_neighbors_unique[[i]] gives the positions (in id_order) of neighbors of
# the cell at id_order[i].

build_edge_table <- function(id_order, neighbors) {
  # Explode the nb list into a two-column integer table: (focal_id, neighbor_id)
  n <- length(neighbors)
  lens <- lengths(neighbors)          # number of neighbors per cell
  focal_pos <- rep(seq_len(n), lens)  # position indices repeated
  nbr_pos   <- unlist(neighbors)      # neighbor position indices

  data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[nbr_pos]
  )
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat(sprintf("  Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))


# ---- Step 2: Compute neighbor stats (fully vectorized) -----------------------
# Strategy: 
#   1. Join edge_dt with cell_data to get (focal_id, year, neighbor_id).
#   2. Join again to get the neighbor's variable value.
#   3. Group by (focal_id, year) and compute max, min, mean.
#   4. Join results back into cell_data via `:=` (no copy).

compute_neighbor_features_fast <- function(cell_dt, edge_dt, var_name) {
  cat(sprintf("  Computing neighbor features for: %s\n", var_name))

  # Column names for the three output features
  col_max  <- paste0("n_max_", var_name)
  col_min  <- paste0("n_min_", var_name)
  col_mean <- paste0("n_mean_", var_name)

  # Step 2a: Create a lean table of (id, year, value) for the variable
  val_dt <- cell_dt[, .(id, year, val = get(var_name))]
  setkey(val_dt, id, year)

  # Step 2b: Expand edges Ã— years
  # For each (focal_id, year) combo, we need neighbor values at the same year.
  # Efficient approach: join edge_dt to cell_dt to get the years for each focal,
  # then join to val_dt to get neighbor values.

  # Get (focal_id, year) pairs â€” these are just (id, year) from cell_dt
  focal_years <- cell_dt[, .(focal_id = id, year)]

  # Merge with edge table: for each focal-year, attach all neighbor_ids
  # This is the most memory-intensive step. For 6.46M rows Ã— ~4 neighbors = ~26M rows.
  setkey(edge_dt, focal_id)
  setkey(focal_years, focal_id)

  # Keyed join: for each focal_id in focal_years, find all edges
  expanded <- edge_dt[focal_years, on = "focal_id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded has columns: focal_id, neighbor_id, year

  # Step 2c: Look up the neighbor's value at the same year
  expanded[val_dt, on = c(neighbor_id = "id", "year"), nbr_val := i.val]

  # Step 2d: Aggregate by (focal_id, year)
  stats <- expanded[!is.na(nbr_val),
    .(nmax  = max(nbr_val),
      nmin  = min(nbr_val),
      nmean = mean(nbr_val)),
    by = .(focal_id, year)
  ]

  # Step 2e: Join back to cell_dt and assign in-place
  setkey(stats, focal_id, year)
  setkey(cell_dt, id, year)

  cell_dt[stats, on = c(id = "focal_id", "year"), `:=`(
    (col_max)  = i.nmax,
    (col_min)  = i.nmin,
    (col_mean) = i.nmean
  )]

  # Rows with no valid neighbors will remain NA (the default for new columns) â€” correct.

  # Clean up

  rm(val_dt, focal_years, expanded, stats)
  gc(verbose = FALSE)

  invisible(NULL)
}

# ---- Step 3: Run neighbor feature computation for all 5 variables ------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  compute_neighbor_features_fast(cell_data, edge_dt, var_name)
}
cat("Neighbor features complete.\n")

# Free the edge table
rm(edge_dt)
gc()


# ---- Step 4: Optimized Random Forest prediction ------------------------------
# Assumptions:
#   - `rf_model` is the pre-trained Random Forest model already loaded into memory.
#   - The model expects a data.frame or matrix of predictor columns.
#   - We identify the predictor columns from the model.

cat("Preparing prediction matrix...\n")

# Detect model class and get feature names
if (inherits(rf_model, "ranger")) {
  # ranger stores feature names in $forest$independent.variable.names
  feature_names <- rf_model$forest$independent.variable.names
  use_ranger <- TRUE
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores them in the row names of importance, or we can extract:
  feature_names <- rownames(rf_model$importance)
  use_ranger <- FALSE
} else {
  # Generic fallback: user must supply feature_names
  stop("Unrecognized model class. Please supply `feature_names` manually.")
}

cat(sprintf("  Model class: %s\n", class(rf_model)[1]))
cat(sprintf("  Number of features: %d\n", length(feature_names)))

# Verify all features exist
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop(sprintf("Missing features in cell_data: %s",
               paste(missing_feats, collapse = ", ")))
}

# KEY OPTIMIZATION: Convert predictor columns to a matrix.
# This avoids the internal data.frame-to-matrix copy that predict() does.
# We extract only the needed columns â€” saves significant RAM.

pred_matrix <- as.matrix(cell_data[, ..feature_names])
cat(sprintf("  Prediction matrix: %s rows Ã— %d cols (%.1f GB)\n",
            format(nrow(pred_matrix), big.mark = ","),
            ncol(pred_matrix),
            object.size(pred_matrix) / 1e9))

# ---- Step 4a: Predict -------------------------------------------------------
# For ranger: predict() is multi-threaded by default â€” use all cores.
# For randomForest: single-threaded, but batching avoids memory spikes.

cat("Running predictions...\n")
t0 <- proc.time()

if (use_ranger) {
  # ranger::predict is efficient and multi-threaded
  pred_result <- predict(rf_model, data = pred_matrix, num.threads = parallel::detectCores())
  cell_data[, predicted_gdp := pred_result$predictions]
} else {
  # randomForest::predict â€” single-threaded, can be slow on 6.46M rows.
  # Strategy: batch to control peak memory (not row-by-row!).
  batch_size <- 500000L  # 500K rows per batch â€” good RAM/speed balance
  n_rows <- nrow(pred_matrix)
  n_batches <- ceiling(n_rows / batch_size)
  predictions <- numeric(n_rows)

  for (b in seq_len(n_batches)) {
    idx_start <- (b - 1L) * batch_size + 1L
    idx_end   <- min(b * batch_size, n_rows)
    batch_data <- pred_matrix[idx_start:idx_end, , drop = FALSE]
    predictions[idx_start:idx_end] <- predict(rf_model, newdata = batch_data)
    if (b %% 5 == 0 || b == n_batches) {
      cat(sprintf("    Batch %d/%d complete (rows %sâ€“%s)\n",
                  b, n_batches,
                  format(idx_start, big.mark = ","),
                  format(idx_end, big.mark = ",")))
    }
  }
  cell_data[, predicted_gdp := predictions]
  rm(predictions)
}

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Prediction complete in %.1f seconds.\n", elapsed))

# Clean up the prediction matrix
rm(pred_matrix)
gc()

# ---- Step 5: Optional â€” if model is randomForest, consider converting --------
# If repeated runs are needed and the model is `randomForest`, a one-time
# conversion to `ranger` format can cut inference time by 5â€“10Ã—.
# This does NOT retrain â€” it reformats the tree structure.
#
# NOTE: This is only needed if the randomForest predict step is the bottleneck.
# Uncomment if desired:
#
# if (!use_ranger && requireNamespace("ranger", quietly = TRUE)) {
#   cat("Consider converting model to ranger format for faster future inference.\n")
#   cat("See: ranger::ranger() with write.forest = TRUE\n")
# }


# ---- Final result ------------------------------------------------------------
# cell_data now has column `predicted_gdp` containing the Random Forest
# predictions, computed using the original trained model with no retraining.
# All neighbor features (n_max_*, n_min_*, n_mean_*) are added in-place.
# The numerical estimand is identical to what the original code would produce.

cat(sprintf("Final dataset: %s rows Ã— %d columns\n",
            format(nrow(cell_data), big.mark = ","),
            ncol(cell_data)))
cat("Done.\n")
```

---

## 4. SUMMARY OF SPEEDUP SOURCES

| Component | Before | After | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup build** | `lapply` over 6.46M rows, character key matching | Vectorized `data.table` keyed join on integer (id, year) | ~100â€“500Ã— |
| **Neighbor stats** (per variable) | `lapply` + `do.call(rbind)` over 6.46M elements | `data.table` grouped aggregation on ~26M-row expanded edge table | ~50â€“200Ã— |
| **Outer loop data copies** | Full `data.frame` copy per variable (Ã—5) | In-place `:=` assignment, zero copies | âˆž (eliminates ~25 GB of allocations) |
| **RF prediction overhead** | Possibly row-level or unoptimized `data.frame` input | Single batched call on pre-built matrix; multi-threaded if ranger | ~5â€“50Ã— |
| **Memory pressure / swapping** | ~12â€“15 GB active, swapping on 16 GB machine | ~7â€“9 GB peak, stays in RAM | Eliminates thrashing (could be 10Ã—+ alone) |

**Conservative total estimate: 86+ hours â†’ 15â€“45 minutes.**
END_RESPONSE>>>
