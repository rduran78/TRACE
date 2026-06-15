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
run_id: Anthropic_F03_T00_R007
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 7
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_007.md

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

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix â€” this is a known slow pattern in R.

**Outer loop:** Calls `compute_neighbor_stats` 5 times (once per variable), and each call likely also copies the entire `cell_data` data.frame when adding columns (R's copy-on-modify semantics). With ~110+ columns and 6.46M rows, each copy is ~5â€“7 GB of memory churn.

### 1.2 Prediction Bottleneck

With a Random Forest model and 6.46M rows Ã— 110 predictors:
- `predict.randomForest` (or `predict.ranger`) must push every row through every tree.
- If the model object is large (e.g., `randomForest` package stores full trees in R lists), just loading it and calling `predict` on 6.46M rows can exhaust 16 GB RAM.
- If prediction is done row-by-row or in a naive loop, it will be catastrophically slow.

### 1.3 Summary of Root Causes

| Bottleneck | Root Cause | Estimated Impact |
|---|---|---|
| `build_neighbor_lookup` | 6.46M string-paste + named-vector lookups | ~10â€“30 hours |
| `compute_neighbor_stats` | 6.46M `lapply` + `do.call(rbind, ...)` Ã— 5 vars | ~10â€“20 hours |
| Column binding / data copy | Copy-on-modify of 6.46M Ã— 110+ data.frame | ~5â€“10 hours of GC/memory |
| RF prediction | Possibly row-level or un-batched predict call | ~10â€“20+ hours |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation â€” Vectorized with `data.table`

- Replace the named-vector key lookups with **integer-indexed join** via `data.table`.
- Build the neighbor lookup as a **two-column data.table** (`row_idx`, `neighbor_row_idx`) â€” a single flat edge list â€” instead of a list-of-vectors.
- Compute all neighbor stats (max, min, mean) in one **grouped aggregation** per variable using `data.table`'s `[, .(max, min, mean), by=row_idx]` â€” fully vectorized C-level operations.
- Add all 15 new columns (5 vars Ã— 3 stats) **by reference** (`:=`) â€” zero copies.

### 2.2 Prediction â€” Batched with `ranger` (if possible) or chunked `predict`

- If the model is a `ranger` object, `predict` is already fast and vectorized in C++. Call it once on the full matrix.
- If the model is a `randomForest` object, convert prediction input to a `matrix` and call `predict` once (not in a loop). If memory is tight, chunk into ~500K-row batches.
- Ensure the prediction input is a plain `matrix` or `data.frame` with exactly the columns the model expects â€” no extra columns, no factors with mismatched levels.

### 2.3 Memory Management

- Use `data.table` throughout to avoid copies.
- Remove intermediate objects and call `gc()` before prediction.
- If the RF model is from the `randomForest` package and is very large, consider converting it to `ranger` format or using `predict` in chunks.

### Expected Speedup

| Component | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup build | ~10â€“30 hr | ~1â€“3 min | ~500Ã— |
| Neighbor stats (5 vars) | ~10â€“20 hr | ~1â€“5 min | ~300Ã— |
| Column addition | ~5â€“10 hr (copies) | ~0 (by-ref) | âˆž |
| RF prediction | ~10â€“20 hr | ~5â€“30 min | ~30â€“60Ã— |
| **Total** | **~86+ hr** | **~10â€“40 min** | **~100â€“500Ã—** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Random Forest Prediction
# Preserves trained RF model and original numerical estimand.
# =============================================================================

library(data.table)

# ---- 3.1 Build Neighbor Edge List (vectorized, integer-indexed) -------------

build_neighbor_edgelist_dt <- function(cell_dt, id_order, rook_neighbors) {
 
  # Map each cell id to its position in id_order (1-based reference index)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))
 
  # Map each (id, year) pair to its row index in cell_dt
  # cell_dt must have columns: id, year (and be keyed or ordered as desired)
  cell_dt[, .row_idx := .I]
 
  # Build a lookup: for each unique id, which row indices does it occupy?
  # We'll join on (id, year) so we need a fast keyed lookup.
  id_year_lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)
 
  # Get unique years
  unique_years <- sort(unique(cell_dt$year))
 
  # For each cell-row, find its neighbors' row indices.
  # Strategy: build a flat edge list from the nb object, then join on year.
 
  # Step 1: Expand nb object into a data.table of (ref_idx, neighbor_ref_idx)
  #   where ref_idx is position in id_order.
  nb_edges <- rbindlist(lapply(seq_along(rook_neighbors), function(ref_i) {
    nb_i <- rook_neighbors[[ref_i]]
    if (length(nb_i) == 0L || (length(nb_i) == 1L && nb_i[1] == 0L)) {
      return(NULL)
    }
    data.table(ref_idx = ref_i, nb_ref_idx = as.integer(nb_i))
  }))
 
  # Step 2: Map ref_idx -> cell id
  nb_edges[, cell_id := id_order[ref_idx]]
  nb_edges[, nb_cell_id := id_order[nb_ref_idx]]
 
  # Step 3: Cross-join with years to get (cell_id, year, nb_cell_id)
  #   Then join to get row indices for both the focal cell and the neighbor.
  #   This is the key insight: a neighbor relationship is constant across years,
  #   so we replicate it for each year.
 
  # To avoid a massive cross join (1.37M edges Ã— 28 years = 38.4M rows),
  # we do it in a memory-efficient way.
 
  years_dt <- data.table(year = unique_years)
  nb_by_year <- nb_edges[, .(cell_id, nb_cell_id)][
    , CJ_dt := TRUE  # placeholder
  ]
 
  # Cross join edges Ã— years
  nb_full <- nb_edges[, .(cell_id, nb_cell_id)][
    , .(year = unique_years), by = .(cell_id, nb_cell_id)
  ]
 
  # Join to get focal row index
  setkey(nb_full, cell_id, year)
  setkey(id_year_lookup, id, year)
  nb_full[id_year_lookup, focal_row := i..row_idx, on = .(cell_id = id, year)]
 
  # Join to get neighbor row index
  nb_full[id_year_lookup, nb_row := i..row_idx, on = .(nb_cell_id = id, year)]
 
  # Remove rows where either focal or neighbor row is missing
  nb_full <- nb_full[!is.na(focal_row) & !is.na(nb_row)]
 
  # Clean up temporary column in cell_dt
  cell_dt[, .row_idx := NULL]
 
  return(nb_full[, .(focal_row, nb_row)])
}

# ---- 3.2 Compute All Neighbor Stats at Once (vectorized) -------------------

compute_all_neighbor_features_dt <- function(cell_dt, edge_dt, var_names) {
 
  n <- nrow(cell_dt)
 
  for (var_name in var_names) {
    message("  Computing neighbor stats for: ", var_name)
   
    # Extract the variable values indexed by row
    vals <- cell_dt[[var_name]]
   
    # Attach neighbor values to edge list
    edge_dt[, nb_val := vals[nb_row]]
   
    # Compute grouped stats â€” fully vectorized in C
    stats <- edge_dt[!is.na(nb_val),
      .(
        nb_max  = max(nb_val),
        nb_min  = min(nb_val),
        nb_mean = mean(nb_val)
      ),
      by = focal_row
    ]
   
    # Initialize result columns with NA
    max_col  <- paste0("neighbor_max_", var_name)
    min_col  <- paste0("neighbor_min_", var_name)
    mean_col <- paste0("neighbor_mean_", var_name)
   
    set(cell_dt, j = max_col,  value = rep(NA_real_, n))
    set(cell_dt, j = min_col,  value = rep(NA_real_, n))
    set(cell_dt, j = mean_col, value = rep(NA_real_, n))
   
    # Fill in computed values by reference (no copy)
    rows <- stats$focal_row
    set(cell_dt, i = rows, j = max_col,  value = stats$nb_max)
    set(cell_dt, i = rows, j = min_col,  value = stats$nb_min)
    set(cell_dt, i = rows, j = mean_col, value = stats$nb_mean)
  }
 
  # Clean up temp column
  edge_dt[, nb_val := NULL]
 
  invisible(cell_dt)
}

# ---- 3.3 Batched Random Forest Prediction ----------------------------------

predict_rf_batched <- function(model, newdata, batch_size = 500000L) {
  # Works with both randomForest and ranger model objects.
  # Preserves the original numerical estimand exactly.
 
  n <- nrow(newdata)
 
  if (n <= batch_size) {
    # Small enough to predict in one shot
    if (inherits(model, "ranger")) {
      return(predict(model, data = newdata)$predictions)
    } else {
      return(as.numeric(predict(model, newdata = newdata)))
    }
  }
 
  # Chunked prediction
  predictions <- numeric(n)
  starts <- seq(1L, n, by = batch_size)
 
  for (k in seq_along(starts)) {
    i_start <- starts[k]
    i_end   <- min(i_start + batch_size - 1L, n)
    idx     <- i_start:i_end
   
    batch <- newdata[idx, , drop = FALSE]
   
    if (inherits(model, "ranger")) {
      predictions[idx] <- predict(model, data = batch)$predictions
    } else {
      predictions[idx] <- as.numeric(predict(model, newdata = batch))
    }
   
    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("  Predicted %d / %d rows (%.1f%%)",
                       i_end, n, 100 * i_end / n))
    }
  }
 
  return(predictions)
}

# ---- 3.4 Main Pipeline Orchestration ----------------------------------------

run_optimized_pipeline <- function(cell_data,
                                    id_order,
                                    rook_neighbors_unique,
                                    rf_model,
                                    predictor_names,
                                    batch_size = 500000L) {
 
  # --- Step 0: Convert to data.table (by reference if already one) ---
  if (!is.data.table(cell_data)) {
    cell_dt <- as.data.table(cell_data)
  } else {
    cell_dt <- copy(cell_data)  # safety copy to avoid mutating caller's object
  }
 
  message("=== Step 1/3: Building neighbor edge list ===")
  t0 <- proc.time()
 
  edge_dt <- build_neighbor_edgelist_dt(cell_dt, id_order, rook_neighbors_unique)
  setkey(edge_dt, focal_row)
 
  message(sprintf("  Edge list: %s edges. Elapsed: %.1f sec",
                   format(nrow(edge_dt), big.mark = ","),
                   (proc.time() - t0)[3]))
 
  # --- Step 2: Compute neighbor features ---
  message("=== Step 2/3: Computing neighbor features ===")
  t1 <- proc.time()
 
  neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
  compute_all_neighbor_features_dt(cell_dt, edge_dt, neighbor_source_vars)
 
  message(sprintf("  Neighbor features done. Elapsed: %.1f sec",
                   (proc.time() - t1)[3]))
 
  # Free edge list memory

rm(edge_dt)
  gc()
 
  # --- Step 3: Random Forest Prediction ---
  message("=== Step 3/3: Running Random Forest prediction ===")
  t2 <- proc.time()
 
  # Prepare prediction input: only the columns the model needs
  # Convert to a plain data.frame or matrix for predict()
  pred_input <- as.data.frame(cell_dt[, ..predictor_names])
 
  # Batched prediction â€” preserves original numerical estimand
  cell_dt[, predicted_gdp := predict_rf_batched(rf_model, pred_input, batch_size)]
 
  rm(pred_input)
  gc()
 
  message(sprintf("  Prediction done. Elapsed: %.1f sec",
                   (proc.time() - t2)[3]))
 
  message("=== Pipeline complete ===")
  return(cell_dt)
}

# =============================================================================
# USAGE EXAMPLE (uncomment and adapt to your environment)
# =============================================================================
#
# library(data.table)
# library(ranger)        # or library(randomForest)
#
# # Load pre-trained model
# rf_model <- readRDS("trained_rf_model.rds")
#
# # Load data
# cell_data            <- readRDS("cell_data.rds")
# id_order             <- readRDS("id_order.rds")
# rook_neighbors_unique <- readRDS("rook_neighbors_unique.rds")
#
# # The exact predictor column names the model was trained on
# predictor_names <- readRDS("predictor_names.rds")
#   # Or: predictor_names <- rf_model$forest$independent.variable.names  (ranger)
#   # Or: names(rf_model$forest$xlevels)                                 (randomForest)
#
# # Run
# result <- run_optimized_pipeline(
#   cell_data             = cell_data,
#   id_order              = id_order,
#   rook_neighbors_unique = rook_neighbors_unique,
#   rf_model              = rf_model,
#   predictor_names       = predictor_names,
#   batch_size            = 500000L
# )
#
# # result is a data.table with column "predicted_gdp"
```

---

## 4. KEY DESIGN DECISIONS EXPLAINED

| Decision | Rationale |
|---|---|
| **Flat edge list instead of list-of-vectors** | A two-column `data.table` of `(focal_row, nb_row)` enables grouped aggregation in C, eliminating 6.46M R-level `lapply` iterations. |
| **`data.table` grouped aggregation** | `edge_dt[, .(max, min, mean), by=focal_row]` runs in compiled C with radix-sort grouping â€” orders of magnitude faster than R-level loops. |
| **`set()` for column assignment** | Modifies `cell_dt` by reference with zero memory copies, critical on a 16 GB machine with a ~5 GB dataset. |
| **Batched prediction** | Keeps peak memory under control. A 500K-row batch Ã— 110 columns â‰ˆ 440 MB â€” safe headroom alongside the model and data. |
| **Model-agnostic predict wrapper** | Detects `ranger` vs `randomForest` and calls the correct API, preserving the exact numerical output of the trained model. |
| **No model retraining** | The trained model is loaded and used as-is; only the inference and feature-prep paths are optimized. |

This pipeline should reduce total runtime from **86+ hours to approximately 10â€“40 minutes** on the described hardware.
END_RESPONSE>>>
