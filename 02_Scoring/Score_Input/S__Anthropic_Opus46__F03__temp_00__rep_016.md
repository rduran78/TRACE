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
run_id: Anthropic_F03_T00_R016
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 16
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_016.md

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

**Cost:** ~6.46M iterations Ã— multiple string allocations and hash lookups = extremely slow. This alone could take hours.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing three summary statistics. The `do.call(rbind, result)` at the end binds 6.46M small vectors into a matrix â€” this is a known R anti-pattern that is very slow and memory-hungry.

**Outer loop:** This is called 5 times (once per neighbor source variable), so `compute_neighbor_stats` runs 5 Ã— 6.46M = ~32.3M iterations total.

### 1.2 Prediction Workflow Bottlenecks (Inferred)

Although the prediction code isn't shown, common problems at this scale include:
- **Loading the model from disk on every call** instead of once.
- **Predicting row-by-row** instead of in a single batch `predict(model, newdata)`.
- **Unnecessary `data.frame` copies** (R's copy-on-modify semantics) when adding columns inside loops.
- **Memory pressure:** 6.46M rows Ã— 110 columns of doubles â‰ˆ 5.4 GB. Combined with a Random Forest object (often 1â€“4 GB) and intermediate copies, 16 GB RAM is tight, causing swapping.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| Neighbor lookup | Per-row string pasting and named-vector lookup | Vectorize entirely with `data.table` keyed joins â€” no `lapply`, no `paste` per row |
| Neighbor stats | 6.46M `lapply` iterations + `do.call(rbind, ...)` | Expand neighbor pairs into a long table, compute grouped `max/min/mean` via `data.table` |
| Column addition | Repeated `cell_data <- cbind(...)` copies the whole data.frame | Use `data.table` `:=` (modify in place, zero copies) |
| Prediction | Possibly row-by-row or chunked sub-optimally | Single batch `predict()` call; if memory-constrained, chunk in ~500K blocks |
| Model loading | Possibly reloaded repeatedly | Load once with `readRDS`, keep in memory |
| Memory | ~5.4 GB data + model + intermediates > 16 GB | Use `data.table` (lower overhead), `gc()` between stages, predict in chunks |

**Expected speedup:** From 86+ hours to roughly 10â€“30 minutes for feature preparation, plus prediction time depending on the forest size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites: install.packages(c("data.table", "ranger")) or whatever RF
# package was used for training. The code below is generic to both
# randomForest::predict and ranger::predict.

library(data.table)

# ---- 0. Load model ONCE ------------------------------------------------------
rf_model <- readRDS("path/to/trained_rf_model.rds")  # load once, reuse

# ---- 1. Convert cell_data to data.table in place ----------------------------
#    Assumes cell_data is a data.frame/data.table with columns: id, year, and
#    all predictor columns.
setDT(cell_data)

# ---- 2. Build neighbor edge list (vectorised, no lapply) ---------------------
build_neighbor_edgelist <- function(id_order, nb_object) {
  # nb_object: spdep nb list â€” nb_object[[i]] gives integer indices of
  # neighbors of the i-th element in id_order.
  #
  # Returns a data.table with columns: id, neighbor_id
  # where both refer to the original cell IDs (not positional indices).

  # Pre-allocate vectors
  n <- length(nb_object)
  lengths_vec <- lengths(nb_object)
  total_edges <- sum(lengths_vec)

  from_idx <- rep.int(seq_len(n), lengths_vec)
  to_idx   <- unlist(nb_object, use.names = FALSE)

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed edges)

# ---- 3. Compute ALL neighbor features in one vectorised pass -----------------
compute_all_neighbor_features <- function(cell_dt, edge_dt,
                                          source_vars) {
  # Join edges with year to create (id, year, neighbor_id) triples,
  # then look up neighbor values and aggregate.

  # Step A: Get unique (id, year) and their row indices
  cell_dt[, .rowid := .I]

  # Step B: Create the join table â€” every (id, year) paired with its neighbors
  #   We need: for each row (id, year), find all neighbor_ids, then look up
  #   the neighbor's value for that same year.

  # Keyed join: edge_dt on cell_dt to get years for each id
  # But that would explode to 6.46M Ã— avg_neighbors rows.
  # Instead, work with (id, year, neighbor_id) and join neighbor values.

  # Unique id-year combinations (same as cell_dt rows)
  id_year <- cell_dt[, .(id, year, .rowid)]

  # Merge with edge list: for each (id, year), get all neighbor_ids
  # This is the most memory-intensive step.
  # Estimated size: 6.46M rows Ã— avg ~4 neighbors = ~25.8M rows
  setkey(edge_dt, id)
  setkey(id_year, id)

  cat("  Expanding id-year-neighbor triples...\n")
  # Use allow.cartesian because each id maps to multiple neighbors
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded has columns: id, neighbor_id, year, .rowid
  # .rowid refers to the original row in cell_dt

  # Step C: For each source variable, look up the neighbor's value
  #   by joining on (neighbor_id == id, year == year)
  # Prepare a lookup keyed on (id, year)
  cat("  Preparing value lookup...\n")
  value_lookup <- cell_dt[, c("id", "year", source_vars), with = FALSE]
  setnames(value_lookup, "id", "neighbor_id")
  setkeyv(value_lookup, c("neighbor_id", "year"))
  setkeyv(expanded, c("neighbor_id", "year"))

  cat("  Joining neighbor values...\n")
  expanded <- value_lookup[expanded, on = c("neighbor_id", "year"),
                           nomatch = NA]
  # Now expanded has: neighbor_id, year, <source_vars>, id, .rowid

  # Step D: Aggregate by .rowid (= original row) to get max, min, mean
  cat("  Aggregating neighbor statistics...\n")
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(
      suppressWarnings(max(X, na.rm = TRUE)), list(X = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(
      suppressWarnings(min(X, na.rm = TRUE)), list(X = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(
      mean(X, na.rm = TRUE), list(X = v_sym))
  }

  agg_result <- expanded[, eval(as.call(c(as.name("list"),
                                           agg_exprs))),
                          by = .rowid]

  # Replace Inf/-Inf (from max/min of all-NA) with NA
  inf_cols <- grep("^n_max_|^n_min_", names(agg_result), value = TRUE)
  for (col in inf_cols) {
    set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
  }

  # Step E: Join aggregated features back to cell_dt by .rowid
  setkey(agg_result, .rowid)
  feat_cols <- setdiff(names(agg_result), ".rowid")

  cat("  Merging features back to main table...\n")
  cell_dt[agg_result, (feat_cols) := mget(feat_cols), on = ".rowid"]

  # Handle rows with NO neighbors (they won't appear in agg_result)
  # They already have NA from the join (data.table default), which is correct.

  # Clean up
  cell_dt[, .rowid := NULL]

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})
# cell_data is modified in place â€” no copy

# Free intermediate memory
rm(edge_dt)
gc()

# ---- 4. Prediction (batch, memory-aware) -------------------------------------
predict_in_chunks <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }

  cat("Predicting in chunks of", chunk_size, "...\n")
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  for (i in seq_along(starts)) {
    s <- starts[i]
    e <- min(s + chunk_size - 1L, n)
    cat(sprintf("  Chunk %d/%d (rows %d-%d)\n", i, length(starts), s, e))
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
    if (i %% 5 == 0) gc()
  }
  preds
}

# Prepare the prediction data.frame (Random Forest predict methods often
# require a plain data.frame, not data.table)
predictor_cols <- setdiff(names(cell_data),
                          c("id", "year", "gdp", "response_variable"))
# Adjust the above exclusion list to match your actual non-predictor columns.

cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_in_chunks(
    rf_model,
    newdata = as.data.frame(.SD),
    chunk_size = 500000L
  ), .SDcols = predictor_cols]
})

cat("Done.\n")

# ---- 5. (Optional) Write results --------------------------------------------
# fwrite(cell_data[, .(id, year, predicted_gdp)], "predictions.csv")
```

---

## 4. SUMMARY OF KEY CHANGES AND WHY THEY WORK

| # | Change | Mechanism | Impact |
|---|---|---|---|
| 1 | Replace `build_neighbor_lookup` (6.46M-iteration `lapply` with string ops) with a vectorised `data.table` edge list built from the `nb` object using `rep.int` + `unlist` | Eliminates all per-row `paste`, `as.character`, and named-vector lookups | **~1000Ã— faster** (seconds vs. hours) |
| 2 | Replace 5 Ã— `compute_neighbor_stats` (each 6.46M-iteration `lapply` + `do.call(rbind, ...)`) with a single `data.table` grouped aggregation on the expanded edge table | One vectorised join + one grouped aggregation replaces 32.3M R-level iterations | **~500â€“1000Ã— faster**; eliminates the `do.call(rbind, 6.46M-element list)` anti-pattern |
| 3 | Use `data.table` `:=` for in-place column addition | Avoids R's copy-on-modify; no full-table copies when adding 15 new columns | **Saves ~5 GB** of transient memory allocation |
| 4 | Load RF model once with `readRDS` | Avoids repeated disk I/O | Trivial but important if the original code reloaded per-chunk |
| 5 | Batch `predict()` with optional chunking | Single `predict` call is orders of magnitude faster than row-by-row; chunking at 500K keeps memory bounded | Prediction itself is typically fast for a pre-trained forest; chunking prevents OOM on 16 GB |
| 6 | Explicit `gc()` calls between stages | Returns freed memory to the OS promptly | Prevents swap thrashing on a 16 GB machine |

**The trained Random Forest model is never modified or retrained.** All numerical computations (`max`, `min`, `mean`) are identical to the originals, preserving the original numerical estimand.
END_RESPONSE>>>
