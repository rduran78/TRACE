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
run_id: Anthropic_F03_T03_R014
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 14
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_014.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### Primary Bottlenecks

**A. `build_neighbor_lookup` â€” O(n) `lapply` over 6.46M rows with per-row string operations**

The function creates a character-keyed lookup (`paste(id, year)`) and then, for every single row, does:
- `as.character()` conversion,
- subsetting a named character vector (`id_to_ref`),
- `paste()` to build neighbor keys,
- named-vector subsetting of `idx_lookup` (which is a 6.46M-element named character vector â€” each lookup is O(n) in base R's linear-scan of names).

With ~6.46M rows and an average of ~4 rook neighbors per cell, this produces **~25M string-match lookups against a 6.46M-element named vector**. Base R named-vector lookup is not hash-based; it is **linear scan**. This alone can take tens of hours.

**B. `compute_neighbor_stats` â€” repeated per-variable `lapply` over 6.46M rows**

Each call iterates over all 6.46M rows, subsets a numeric vector, removes NAs, and computes `max/min/mean`. This is called 5 times (once per source variable), producing 5 Ã— 6.46M = ~32M R-level function calls with per-element allocation.

**C. `do.call(rbind, result)` on a 6.46M-element list of 3-vectors**

This is a classic R anti-pattern. `do.call(rbind, ...)` on millions of small vectors is extremely slow because it repeatedly allocates and copies.

**D. Random Forest Prediction (downstream)**

With ~6.46M rows Ã— 110 predictors, a single `predict()` call on a large `ranger` or `randomForest` object will:
- Attempt to allocate a full prediction matrix in memory (~6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.3 GB for doubles alone).
- If using `randomForest::predict`, the object is copied internally via R's copy-on-modify semantics.
- If the model is a `randomForest` object (not `ranger`), prediction is single-threaded and extremely slow at this scale.

**E. Memory pressure**

The full data frame at 6.46M Ã— 110 columns â‰ˆ 5.3 GB. Adding neighbor features, intermediate copies, and the RF model can easily exceed 16 GB, causing swap thrashing.

---

## 2. Optimization Strategy

| Bottleneck | Fix | Expected Speedup |
|---|---|---|
| Named-vector lookup in `build_neighbor_lookup` | Replace with `data.table` hash join or integer-indexed matrix | 100â€“500Ã— |
| Per-row `lapply` in `build_neighbor_lookup` | Vectorized merge/join via `data.table` producing a flat edge-list, then grouped operations | 50â€“200Ã— |
| Per-row `lapply` + `do.call(rbind)` in `compute_neighbor_stats` | Flat edge-list join + `data.table` grouped aggregation (`max`, `min`, `mean` by row) | 50â€“100Ã— |
| 5 separate passes over 6.46M rows | Single pass: compute all 5 variables' neighbor stats simultaneously | 5Ã— |
| RF prediction on 6.46M rows at once | Chunked prediction (e.g., 500K rows) to control peak memory | Avoids OOM / swap |
| `randomForest::predict` (if applicable) | Convert to `ranger` for multi-threaded prediction, or use chunked predict | 4â€“8Ã— (multi-core) |
| Repeated data.frame copies (`cell_data <- ...`) | Use `data.table` set-by-reference (`:=`) | 2â€“5Ã— |

**Core idea:** Replace the row-level `lapply` + string-key pattern with a **flat edge-list + `data.table` grouped join**, then compute all neighbor statistics in one vectorized pass.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================
# Requirements:
#   install.packages(c("data.table", "ranger"))
#   - If your trained model is a `randomForest` object, the chunked predict
#     wrapper below handles it. If it's `ranger`, even better (native threads).
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (by reference, no copy) --------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---- Step 1: Build flat edge-list from nb object (once) ---------------------
# This replaces build_neighbor_lookup entirely.
# rook_neighbors_unique is an nb object: a list of integer index vectors
# id_order is the vector of cell IDs corresponding to each nb element.

build_edge_list_dt <- function(id_order, neighbors) {
  # Pre-allocate: count total edges
  n_edges <- sum(vapply(neighbors, length, integer(1)))

  from_idx <- integer(n_edges)
  to_idx   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(neighbors)) {
    nb_i <- neighbors[[i]]
    len  <- length(nb_i)
    if (len > 0L) {
      from_idx[pos:(pos + len - 1L)] <- i
      to_idx[pos:(pos + len - 1L)]   <- nb_i
      pos <- pos + len
    }
  }

  data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
}

cat("Building edge list...\n")
edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)
# edge_dt has columns: from_id, to_id  (cell-level, year-agnostic)

cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- Step 2: Compute all neighbor features in one vectorized pass -----------

compute_all_neighbor_features_dt <- function(cell_dt, edge_dt, source_vars) {
  # We need to join edges with data by (from_id, year) -> row in cell_dt
  # and look up neighbor values by (to_id, year).
  #
  # Strategy:
  #   1. Create a slim table of just id, year, and the source vars.
  #   2. Cross-join edges with years (edges are year-agnostic).
  #   3. Join neighbor values, then group by (from_id, year) to get stats.

  # Slim table for neighbor value lookup
  keep_cols <- c("id", "year", source_vars)
  slim <- cell_dt[, ..keep_cols]
  setkey(slim, id, year)

  # Get unique years
  years <- sort(unique(cell_dt$year))

  # Expand edge list across all years: each edge exists in every year
  # To avoid a massive cross-join in memory, process in year chunks
  cat("Computing neighbor statistics across all years and variables...\n")

  # Pre-allocate result columns in cell_dt
  for (v in source_vars) {
    for (stat in c("max", "min", "mean")) {
      col_name <- paste0("n_", stat, "_", v)
      set(cell_dt, j = col_name, value = NA_real_)
    }
  }

  # Create a row-index lookup: (id, year) -> row position in cell_dt
  cell_dt[, .row_idx := .I]
  setkey(cell_dt, id, year)

  # Process year by year to control memory
  for (yr in years) {
    # Rows for this year
    yr_rows <- cell_dt[year == yr, .(id, .row_idx)]
    setkey(yr_rows, id)

    # Neighbor values for this year
    yr_vals <- slim[year == yr]
    setkey(yr_vals, id)

    # Join: for each edge (from_id -> to_id), get to_id's values in this year
    # edges_yr: from_id, to_id + neighbor values
    edges_yr <- edge_dt[yr_vals, on = .(to_id = id), nomatch = 0L, allow.cartesian = TRUE]
    # edges_yr now has: from_id, to_id, year, and all source_vars (neighbor's values)

    if (nrow(edges_yr) == 0L) next

    # Group by from_id and compute stats for each source var
    agg_exprs <- list()
    for (v in source_vars) {
      v_sym <- as.name(v)
      agg_exprs[[paste0("n_max_", v)]]  <- substitute(max(x, na.rm = TRUE),  list(x = v_sym))
      agg_exprs[[paste0("n_min_", v)]]  <- substitute(min(x, na.rm = TRUE),  list(x = v_sym))
      agg_exprs[[paste0("n_mean_", v)]] <- substitute(mean(x, na.rm = TRUE), list(x = v_sym))
    }

    # Build the aggregation call
    agg_call <- as.call(c(as.name("list"), agg_exprs))
    stats_yr <- edges_yr[, eval(agg_call), by = from_id]

    # Fix Inf/-Inf from max/min on all-NA groups
    stat_cols <- names(stats_yr)[-1]
    for (sc in stat_cols) {
      vals <- stats_yr[[sc]]
      vals[is.infinite(vals)] <- NA_real_
      set(stats_yr, j = sc, value = vals)
    }

    # Map from_id back to row indices in cell_dt
    setkey(stats_yr, from_id)
    matched <- yr_rows[stats_yr, on = .(id = from_id), nomatch = 0L]

    # Write results directly into cell_dt by reference
    row_positions <- matched$.row_idx
    for (sc in stat_cols) {
      set(cell_dt, i = row_positions, j = sc, value = matched[[sc]])
    }
  }

  # Clean up helper column
  cell_dt[, .row_idx := NULL]

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

system.time({
  compute_all_neighbor_features_dt(cell_data, edge_dt, neighbor_source_vars)
})
# cell_data is modified in place (data.table reference semantics) â€” no copy.


# ---- Step 3: Chunked Random Forest Prediction -------------------------------

predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  # Works with both ranger and randomForest model objects.
  # Preserves the original numerical estimand exactly.

  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)

  is_ranger <- inherits(model, "ranger")

  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))

  for (i in seq_len(n_chunks)) {
    start_row <- (i - 1L) * chunk_size + 1L
    end_row   <- min(i * chunk_size, n)
    idx       <- start_row:end_row

    chunk <- newdata[idx, ]

    if (is_ranger) {
      # ranger::predict is multi-threaded by default
      preds[idx] <- predict(model, data = chunk)$predictions
    } else {
      # randomForest::predict
      preds[idx] <- predict(model, newdata = chunk)
    }

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %sâ€“%s)\n",
                  i, n_chunks,
                  format(start_row, big.mark = ","),
                  format(end_row, big.mark = ",")))
    }

    # Release chunk memory
    rm(chunk)
    if (i %% 10 == 0) gc(verbose = FALSE)
  }

  preds
}

# ---- Step 4: Prepare prediction data and run --------------------------------

# Identify the predictor columns your trained model expects.
# If using ranger:
#   predictor_cols <- model$forest$independent.variable.names
# If using randomForest:
#   predictor_cols <- attr(model$terms, "term.labels")  # or stored separately

# Example (adjust to your actual model object name and type):
# pred_data <- cell_data[, ..predictor_cols]   # data.table column subset

cat("Running chunked prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model    = rf_model,        # your pre-trained model object
    newdata  = cell_data,       # or pred_data with only predictor columns
    chunk_size = 500000L
  )]
})

cat("Done.\n")
```

---

## 4. Summary of Expected Improvements

| Component | Before | After | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~hours (linear-scan named vector Ã— 6.46M Ã— ~4 neighbors) | **~5â€“15 seconds** (pre-built flat edge-list, integer ops) | Eliminated O(n) string matching entirely |
| `compute_neighbor_stats` | ~hours (5 Ã— 6.46M `lapply` + `do.call(rbind)`) | **~30â€“90 seconds** (vectorized `data.table` grouped aggregation, year-chunked) | Vectorized C-level grouping replaces R-level loops |
| Feature binding to `cell_data` | Repeated full-copy of 5+ GB data frame | **Zero-copy** (`set()` by reference) | `data.table` reference semantics |
| RF prediction | Single 6.46M-row call (potential OOM / swap) | **Chunked 500K-row calls** (peak ~0.5 GB per chunk) | Stays within 16 GB; avoids swap thrashing |
| RF prediction speed | Single-threaded if `randomForest` | **Multi-threaded if `ranger`** | `ranger` uses all cores by default |
| **Total estimated time** | **86+ hours** | **~5â€“20 minutes** (feature prep) + RF prediction time (model-dependent, typically 10â€“40 min for `ranger`) | |

The trained Random Forest model is never modified â€” only `predict()` is called. The numerical predictions are identical to the original pipeline (same model, same features, same arithmetic). The only changes are to **how** the data is prepared and **how** prediction is dispatched (in memory-safe chunks).
END_RESPONSE>>>
