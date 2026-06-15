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
run_id: Anthropic_F03_T10_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_016.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** runs an `lapply` over ~6.46 million rows. Inside every iteration it:

- Calls `as.character()` on a single ID,
- Performs a named-vector lookup (`id_to_ref`),
- Subsets the `neighbors` list,
- Builds a `paste(â€¦, sep="_")` key for every neighbor,
- Looks those keys up in `idx_lookup` (another named-vector lookup), and
- Filters NAs.

Named-vector lookup in R is O(n) per call when the vector is long (it's a linear scan of the `names` attribute unless hashed). With ~6.46M rows and multiple lookups per row, this is **extremely slow** â€” likely the single largest time sink in the pipeline.

**`compute_neighbor_stats`** is called 5 times (one per variable). Each call runs another `lapply` over 6.46M rows, extracting neighbor values, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end creates 6.46M small 3-element vectors and binds them â€” itself an O(nÂ²) memory-copy pattern.

**In total**, the feature-preparation stage alone performs ~6Ã—6.46M â‰ˆ 39M R-level function calls with slow lookups, small vector allocations, and repeated list-to-matrix conversions.

### 1.2 Random Forest Inference Bottleneck

Calling `predict(rf_model, newdata = <6.46M Ã— 110>)` in a single shot on a `randomForest` or `ranger` object will:

- Allocate a single massive prediction matrix internally.
- On a 16 GB laptop, the working set for 6.46M Ã— 110 `numeric` columns is ~5.3 GB for the data alone, plus the forest's node matrices, plus the prediction scratch space. This can easily exceed RAM and cause swapping / crash.
- If the user is calling `predict` inside a row-level or chunk-level loop (common in poorly optimised pipelines), the per-call overhead is amplified millions of times.

### 1.3 Summary of Root Causes

| Component | Root Cause | Impact |
|---|---|---|
| `build_neighbor_lookup` | R-level loop + named-vector linear scan over 6.46M rows | ~hours |
| `compute_neighbor_stats` | 5 Ã— R-level loop over 6.46M rows + `do.call(rbind,â€¦)` | ~hours |
| Prediction | Possible single giant `predict()` â†’ OOM; or row-by-row predict â†’ overhead | ~hours |
| Memory | Repeated full-copy of `cell_data` via `cell_data <- cbind(cell_data, â€¦)` | GBs wasted |

---

## 2. OPTIMIZATION STRATEGY

| # | Strategy | Expected Speedup |
|---|---|---|
| 1 | Replace named-vector lookups with `data.table` hash joins or integer-keyed environments | 100â€“1000Ã— for lookup step |
| 2 | Vectorise `build_neighbor_lookup` entirely with `data.table` merge/join | Eliminates 6.46M R function calls |
| 3 | Vectorise `compute_neighbor_stats` using `data.table` grouped aggregation over an edge-list representation | Eliminates 5 Ã— 6.46M R function calls |
| 4 | Avoid `do.call(rbind, â€¦)` and in-place column addition; add columns by reference with `:=` | Eliminates O(nÂ²) copy |
| 5 | Batch `predict()` in chunks (~500K rows) to stay within RAM while avoiding per-row overhead | Safe RAM use, fast |
| 6 | Use `ranger` for prediction if possible (C++ back-end, faster than `randomForest::predict`) | 2â€“5Ã— for predict |

---

## 3. WORKING R CODE

```r
# ============================================================
# OPTIMISED PIPELINE
# ============================================================
# Requirements: data.table, ranger (if model is ranger), randomForest
# Preserves: trained RF model object, original numerical estimand.

library(data.table)

# ----------------------------------------------------------
# 0.  Convert working data to data.table (by reference if already is one)
# ----------------------------------------------------------
setDT(cell_data)

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Create an integer row-index column (used for joins)
cell_data[, .row_idx := .I]


# ----------------------------------------------------------
# 1.  Build edge-list representation of neighbour graph
#     (replaces build_neighbor_lookup entirely)
# ----------------------------------------------------------
build_neighbor_edgelist <- function(cell_dt, id_order, neighbors) {
  # id_order : vector of cell IDs in the same order as the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  # --- step a: expand nb list to directed edge-list of cell IDs ---
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-entries spdep uses for cells with no neighbours

  valid    <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edge_ids <- data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )

  # --- step b: for every (id_from, year) find the row index of the
  #             focal cell, and for every (id_to, year) find the row
  #             index of the neighbour cell.
  # We cross-join edges with all years present for the focal cell.
  # Then join to get the neighbour's row index (if it exists for
  # that year).
  # ---

  # Lookup: cell_id + year -> .row_idx
  id_year_lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)

  # Get unique years per focal id  (all years a focal cell appears)
  focal_years <- cell_dt[, .(year), keyby = .(id)]

  # Join focal_years with edges to get (id_from, id_to, year)
  setkey(edge_ids, id_from)
  setkey(focal_years, id)
  edge_year <- edge_ids[focal_years, on = .(id_from = id),
                        .(id_from, id_to, year),
                        allow.cartesian = TRUE, nomatch = NULL]

  # Join to get focal row index
  edge_year[id_year_lookup,
            focal_row := i..row_idx,
            on = .(id_from = id, year)]

  # Join to get neighbour row index
  edge_year[id_year_lookup,
            nbr_row := i..row_idx,
            on = .(id_to = id, year)]

  # Keep only edges where both focal and neighbour exist
  edge_year <- edge_year[!is.na(focal_row) & !is.na(nbr_row)]

  edge_year
}

message("Building neighbour edge-list â€¦")
edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
message(sprintf("  Edge-year rows: %s", format(nrow(edge_dt), big.mark = ",")))


# ----------------------------------------------------------
# 2.  Compute neighbour stats for all variables at once
#     (replaces compute_neighbor_stats + outer loop)
# ----------------------------------------------------------
compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # For each variable, compute max, min, mean of neighbour values
  # grouped by focal_row, then join back.

  for (vn in var_names) {
    message(sprintf("  Neighbour features for: %s", vn))

    # Attach the neighbour's value for this variable to each edge
    edge_dt[, nbr_val := cell_dt[[vn]][nbr_row]]

    # Aggregate: one row per focal_row
    agg <- edge_dt[!is.na(nbr_val),
                   .(vmax  = max(nbr_val),
                     vmin  = min(nbr_val),
                     vmean = mean(nbr_val)),
                   keyby = .(focal_row)]

    # Build target column names (must match what downstream code expects)
    col_max  <- paste0(vn, "_neighbor_max")
    col_min  <- paste0(vn, "_neighbor_min")
    col_mean <- paste0(vn, "_neighbor_mean")

    # Initialise with NA, then fill matched rows by reference
    set(cell_dt, j = col_max,  value = NA_real_)
    set(cell_dt, j = col_min,  value = NA_real_)
    set(cell_dt, j = col_mean, value = NA_real_)

    matched <- agg$focal_row
    set(cell_dt, i = matched, j = col_max,  value = agg$vmax)
    set(cell_dt, i = matched, j = col_min,  value = agg$vmin)
    set(cell_dt, i = matched, j = col_mean, value = agg$vmean)
  }

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbour features â€¦")
compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Free the (potentially large) edge table
rm(edge_dt); gc()

# Remove helper column
cell_data[, .row_idx := NULL]


# ----------------------------------------------------------
# 3.  Batched Random Forest Prediction
# ----------------------------------------------------------
predict_rf_batched <- function(model, newdata, batch_size = 500000L) {
  # Works with both randomForest and ranger model objects.
  # Returns a numeric vector of predictions (preserves estimand).

  n <- nrow(newdata)
  preds <- numeric(n)

  is_ranger <- inherits(model, "ranger")
  starts <- seq(1L, n, by = batch_size)

  message(sprintf("Predicting %s rows in %d batches â€¦",
                  format(n, big.mark = ","), length(starts)))

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + batch_size - 1L, n)

    batch <- newdata[i1:i2, , drop = FALSE]

    if (is_ranger) {
      preds[i1:i2] <- predict(model, data = batch)$predictions
    } else {
      preds[i1:i2] <- predict(model, newdata = batch)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("  batch %d / %d done", k, length(starts)))
    }
  }

  preds
}

# --- Prepare prediction data.frame ---------------------------
# Identify predictor columns (exclude target, id, year, etc.)
# Adjust 'target_col' to whatever your response variable is named.
target_col   <- "gdp"  # <-- change if different
exclude_cols <- c(target_col, "id", "year")
pred_vars    <- setdiff(names(cell_data), exclude_cols)

# Build a plain data.frame / matrix for predict
# (randomForest & ranger both accept data.frame)
pred_df <- as.data.frame(cell_data[, ..pred_vars])

# --- Run prediction ------------------------------------------
message("Running Random Forest prediction â€¦")
cell_data[, predicted_gdp := predict_rf_batched(rf_model, pred_df,
                                                 batch_size = 500000L)]
rm(pred_df); gc()

message("Done.")
```

---

## 4. EXPECTED IMPACT

| Stage | Before (est.) | After (est.) | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~20â€“40 h | ~1â€“3 min | Hash joins replace 6.46M named-vector scans |
| `compute_neighbor_stats` (5 vars) | ~20â€“40 h | ~2â€“5 min | Vectorised `data.table` grouped aggregation; no R-level loop |
| Column binding / copies | ~hours of GC | ~0 s | `:=` by-reference assignment; no copy |
| RF `predict()` | variable / OOM risk | ~5â€“20 min | 500K-row batches; stays in RAM; no per-row overhead |
| **Total** | **~86+ h** | **~10â€“30 min** | **~150â€“500Ã— overall speedup** |

### Key Guarantees

- **Trained RF model is never modified or retrained** â€” only `predict()` is called.
- **Numerical estimand is preserved** â€” no rounding, discretisation, or approximation is introduced; `max`, `min`, `mean`, and RF predictions are computed identically.
- **RAM-safe** â€” the edge-list is the largest interim object (~50â€“100M rows Ã— 5 int columns â‰ˆ 2â€“4 GB), well within 16 GB. Prediction batches cap working memory.
END_RESPONSE>>>
