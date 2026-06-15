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
run_id: Anthropic_F03_T07_R022
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 22
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_022.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations**

The function iterates over ~6.46 million rows, and for every single row it:
1. Converts `data$id[i]` to character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes each neighbor cell ID with the year to create string keys.
4. Looks up those keys in `idx_lookup` (another named character vector).

Named-vector lookup in R is **hash-based but has significant per-call overhead** when done millions of times inside `lapply`. The `paste()` calls create millions of temporary character vectors. The result: this single function likely takes **hours** on 6.46M rows.

**B. `compute_neighbor_stats` â€” repeated per-variable full-data sweeps**

Called 5 times (once per neighbor source variable). Each call does an `lapply` over 6.46M rows, subsetting a numeric vector by index vectors and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M list elements is extremely slow â€” `do.call(rbind, ...)` on a list of millions of small vectors is a known R anti-pattern (quadratic memory allocation).

**C. Object copying in the outer loop**

`cell_data <- compute_and_add_neighbor_features(cell_data, ...)` likely triggers full-copy of the ~6.46M Ã— 110-column data.frame on each iteration (R's copy-on-modify semantics). With 5 variables, that's 5 full copies of a multi-GB object.

**D. Random Forest prediction**

Predicting 6.46M rows Ã— 110 features through a Random Forest (especially one with many trees) is inherently expensive. If `predict()` is called row-by-row or in small batches rather than as a single vectorized call, the overhead multiplies enormously. Model loading from disk (if done repeatedly) also adds cost.

### Summary of Time Sinks (estimated share of 86+ hours)

| Component | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~30-40% | Per-row string ops, named-vector lookup Ã—6.46M |
| `compute_neighbor_stats` (Ã—5) | ~25-35% | `lapply` + `do.call(rbind,...)` on millions of rows |
| Data.frame copying (outer loop) | ~10-15% | Copy-on-modify, repeated column binding |
| RF prediction | ~15-25% | Large matrix, possibly suboptimal call pattern |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Replace row-level R loops with vectorized / `data.table` operations

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Pre-explode the `nb` object into a `data.table` edge list; merge with data by `(neighbor_id, year)` to get row indices â€” fully vectorized, no per-row `paste`/lookup | 50â€“200Ã— |
| `compute_neighbor_stats` | Group-by aggregation on the edge-list `data.table` using `[, .(max, min, mean), by = source_row]` | 50â€“100Ã— |
| `do.call(rbind, ...)` | Eliminated entirely â€” results come from `data.table` aggregation | 10â€“50Ã— |
| Data.frame copy | Use `data.table` with `:=` (modify in place) â€” zero copies | 5Ã— per iteration |
| RF prediction | Single `predict()` call on the full `data.table`/matrix; load model once | Ensures no unnecessary overhead |

**Target runtime: ~5â€“20 minutes** (down from 86+ hours).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest â€” adapts to either)
# Preserves: trained RF model object, original numerical estimand
# =============================================================================

library(data.table)

# ---- 0. Load model once ------------------------------------------------------
# Adjust path/object name to your setup.
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# The model is assumed to already be in memory as `rf_model`.

# ---- 1. Convert working data to data.table (in-place efficiency) -------------
# Assumes `cell_data` is your ~6.46M-row data.frame/data.table with columns
#   id, year, ntl, ec, pop_density, def, usd_est_n2, ... (110 predictors)
# Assumes `id_order` is the vector mapping position in the nb object to cell id.
# Assumes `rook_neighbors_unique` is the spdep::nb list.

if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place â€” no copy
}

# ---- 2. Build a vectorized edge list from the nb object ----------------------
# This replaces `build_neighbor_lookup` entirely.

build_edge_list <- function(id_order, nb_obj) {
  # Explode the nb list into (source_position, neighbor_position) pairs
  n <- length(nb_obj)
  source_pos <- rep(seq_len(n), lengths(nb_obj))
  neighbor_pos <- unlist(nb_obj)

  # Remove the spdep convention where 0 means "no neighbors"
  valid <- neighbor_pos != 0L
  source_pos <- source_pos[valid]
  neighbor_pos <- neighbor_pos[valid]

  # Map positions back to cell IDs
  data.table(
    source_id   = id_order[source_pos],
    neighbor_id = id_order[neighbor_pos]
  )
}

cat("Building edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- 3. Compute all neighbor features in one vectorized pass -----------------
# Strategy:
#   - Join edge_dt with cell_data on (neighbor_id, year) to pull neighbor values.
#   - Group by (source_id, year) to compute max/min/mean.
#   - Join results back to cell_data using := (in-place, no copy).

compute_all_neighbor_features <- function(dt, edge_dt, var_names) {
  # Ensure keys for fast joins
  # We need a row-reference column so we can join back
  dt[, .row_idx := .I]

  # Slim table for joining: only id, year, and the source variables
  # Plus the row index for final join-back
  cols_needed <- c("id", "year", var_names, ".row_idx")
  slim <- dt[, ..cols_needed]

  # --- Step A: Expand edges Ã— years -------------------------------------------
  # For each (source_id, neighbor_id) edge, we need every year present in the data.
  # Instead of a cross-join (expensive), we join edge_dt to the data directly.

  # Create the lookup: for each (id, year) â†’ values of the source vars
  # Key the slim table on (id, year) for fast join
  setkey(slim, id, year)

  # Join: for each edge, pull the neighbor's variable values for every year
  # edge_dt has (source_id, neighbor_id)
  # We want: for each row in cell_data identified by (source_id, year),
  #          find all neighbors, look up their values at the same year.

  # Step A1: Get (source_id, year) pairs from the data
  source_years <- slim[, .(source_id = id, year, .row_idx)]

  # Step A2: Cross edge list with years via join on source_id
  #   Result: (source_id, year, neighbor_id, .row_idx)
  cat("  Joining edges with year dimension...\n")
  setkey(edge_dt, source_id)
  setkey(source_years, source_id)
  expanded <- edge_dt[source_years, on = "source_id", allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded now has columns: source_id, neighbor_id, year, .row_idx
  # .row_idx refers to the source row in cell_data

  # Step A3: Look up neighbor values by (neighbor_id, year)
  cat("  Looking up neighbor values...\n")
  setnames(slim, "id", "neighbor_id")  # rename for join
  # Keep only the var columns + key columns in the right-side table
  neighbor_vals <- slim[, c("neighbor_id", "year", var_names), with = FALSE]
  setkey(neighbor_vals, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  merged <- neighbor_vals[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # merged has: neighbor_id, year, <var_names>, source_id, .row_idx

  # Step A4: Aggregate by source row
  cat("  Computing neighbor statistics...\n")
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(var_names, function(v) {
    vn <- as.name(v)
    list(
      bquote(max(.(vn), na.rm = TRUE)),
      bquote(min(.(vn), na.rm = TRUE)),
      bquote(mean(.(vn), na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))
  names(agg_exprs) <- agg_names

  # Suppress -Inf/Inf warnings from max/min on empty sets
  agg_result <- suppressWarnings(
    merged[, lapply(agg_exprs, eval, envir = .SD), by = .row_idx]
  )

  # Replace Inf/-Inf with NA (from groups where all neighbor values were NA)
  for (col_name in agg_names) {
    set(agg_result, which(is.infinite(agg_result[[col_name]])), col_name, NA_real_)
  }

  # Step A5: Join back to cell_data by .row_idx (in place)
  cat("  Joining neighbor features back to main table...\n")
  setkey(agg_result, .row_idx)
  setkey(dt, .row_idx)

  for (col_name in agg_names) {
    dt[agg_result, (col_name) := get(paste0("i.", col_name)), on = ".row_idx"]
  }

  # Clean up
  dt[, .row_idx := NULL]
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})
# cell_data now has 15 new columns: n_{var}_{max,min,mean} for each of the 5 vars
# All added in-place via := â€” no copies of the 6.46M-row table.

cat(sprintf("  cell_data now has %d columns\n", ncol(cell_data)))

# ---- 4. Random Forest prediction (single vectorized call) --------------------

predict_gdp <- function(dt, model, predictor_cols = NULL, batch_size = 500000L) {
  # Determine predictor columns
  if (is.null(predictor_cols)) {
    # Attempt to extract from model
    if (inherits(model, "ranger")) {
      predictor_cols <- model$forest$independent.variable.names
    } else if (inherits(model, "randomForest")) {
      # randomForest stores variable names used in training
      predictor_cols <- rownames(model$importance)
    } else {
      stop("Cannot auto-detect predictor columns. Please supply `predictor_cols`.")
    }
  }

  # Verify all predictor columns exist
  missing_cols <- setdiff(predictor_cols, names(dt))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing predictor columns in data: %s",
                 paste(missing_cols, collapse = ", ")))
  }

  n <- nrow(dt)
  cat(sprintf("Predicting %s rows with %d predictors...\n",
              format(n, big.mark = ","), length(predictor_cols)))

  # For very large datasets, predict in batches to manage memory peaks
  # while still using vectorized predict() within each batch.
  if (n <= batch_size) {
    # Single call â€” most efficient
    if (inherits(model, "ranger")) {
      preds <- predict(model, data = dt[, ..predictor_cols])$predictions
    } else {
      preds <- predict(model, newdata = dt[, ..predictor_cols])
    }
  } else {
    # Batch prediction to avoid memory spikes on 16 GB laptop
    n_batches <- ceiling(n / batch_size)
    cat(sprintf("  Using %d batches of ~%s rows\n",
                n_batches, format(batch_size, big.mark = ",")))
    preds <- numeric(n)
    for (b in seq_len(n_batches)) {
      start_idx <- (b - 1L) * batch_size + 1L
      end_idx   <- min(b * batch_size, n)
      batch_data <- dt[start_idx:end_idx, ..predictor_cols]

      if (inherits(model, "ranger")) {
        preds[start_idx:end_idx] <- predict(model, data = batch_data)$predictions
      } else {
        preds[start_idx:end_idx] <- predict(model, newdata = batch_data)
      }

      if (b %% 5 == 0 || b == n_batches) {
        cat(sprintf("  Batch %d/%d complete\n", b, n_batches))
      }
    }
  }

  return(preds)
}

cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_gdp(cell_data, rf_model)]
})

cat("Pipeline complete.\n")
cat(sprintf("  Rows predicted: %s\n", format(nrow(cell_data), big.mark = ",")))
cat(sprintf("  Output column:  'predicted_gdp'\n"))

# ---- 5. (Optional) Memory cleanup -------------------------------------------
rm(edge_dt)
gc()
```

---

## 4. WHY THIS WORKS â€” KEY CHANGES SUMMARIZED

| Original | Optimized | Mechanism |
|---|---|---|
| `build_neighbor_lookup`: `lapply` over 6.46M rows with `paste`, named-vector lookup | `build_edge_list`: vectorized `rep`/`unlist` on the `nb` object â†’ `data.table` with ~1.37M edges | Eliminates per-row string operations entirely |
| `compute_neighbor_stats`: `lapply` over 6.46M rows + `do.call(rbind, 6.46M-element list)` Ã— 5 variables | Single `data.table` equi-join + grouped aggregation for all 5 variables simultaneously | `data.table` parallelized C-level grouping; no R-level row loop |
| `cell_data <- cbind(cell_data, ...)` Ã— 5 | `cell_data[, (cols) := values]` in place | Zero-copy column addition via `:=` |
| Potentially row-by-row or repeated `predict()` calls | Single (or large-batch) `predict()` call on the full feature matrix | Amortizes tree-traversal setup; vectorized C/C++ path in ranger/randomForest |
| Model potentially reloaded per iteration | Model loaded once, held in memory | Eliminates repeated deserialization |

**Expected runtime on a 16 GB laptop: ~5â€“20 minutes** (depending on the number of RF trees and their depth), down from the estimated 86+ hours. The trained model and original numerical estimand are fully preserved â€” no retraining, no transformation of the target variable.
END_RESPONSE>>>
