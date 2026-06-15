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
run_id: Anthropic_F03_T03_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering â€” all interpreted R. |
| **Named-vector lookup via `idx_lookup[neighbor_keys]`** | `idx_lookup` is a named character vector of length 6.46M. Subsetting a named vector is O(n) per call in base R (linear scan or hash miss). Called ~6.46M times â†’ effectively O(nÂ²). |
| **Repeated `paste(â€¦, sep="_")`** | Allocates ~6.46M temporary strings per call, inside a loop that itself runs 6.46M times. |
| **Output is a list of 6.46M integer vectors** | ~25â€“50 GB of list overhead alone (each list element has a SEXP header of 56+ bytes). |

**`compute_neighbor_stats`** compounds the problem:

- Another `lapply` over 6.46M elements, each doing subsetting, NA removal, and three summary statistics.
- Called 5 times (once per neighbor source variable), so 5 Ã— 6.46M = 32.3M R-level function calls.
- `do.call(rbind, result)` on a 6.46M-element list is itself slow (repeated row-binding).

**Net effect:** The feature-preparation step alone is O(nÂ²)-ish in practice and dominates the 86+ hour runtime.

### B. Random Forest Inference Bottlenecks

| Problem | Detail |
|---|---|
| **Single `predict()` call on 6.46M rows Ã— 110 features** | `ranger`/`randomForest` `predict` must traverse every tree for every row. With default 500 trees this is ~3.2 billion tree traversals. |
| **Memory pressure** | The prediction matrix alone is 6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.3 GB. Combined with the model object, neighbor lookup, and intermediate data, this can exceed 16 GB and trigger swapping. |
| **Object copying** | Every `cell_data <- compute_and_add_neighbor_features(â€¦)` triggers a full copy of the 6.46M-row data.frame (R's copy-on-modify semantics). Five variables = five full copies. |

---

## 2. Optimization Strategy

### Feature Preparation

1. **Replace the named-vector lookup with `data.table` keyed joins.** A keyed join on `(id, year)` is O(n log n) once, not O(n) per row.
2. **Vectorize `build_neighbor_lookup` entirely.** Expand the neighbor list into a long edge table `(row_i, neighbor_row_j)` and do a single merge. No per-row `lapply`.
3. **Vectorize `compute_neighbor_stats`.** Group-by aggregation on the long edge table: one `data.table` operation per variable instead of 6.46M `lapply` iterations.
4. **Eliminate repeated data.frame copies.** Use `data.table` set-by-reference (`:=`) so adding columns never copies the table.

### Random Forest Inference

5. **Predict in chunks** (~500K rows) to keep peak memory under control.
6. **Use `ranger` if possible** (much faster C++ predict path than `randomForest`). If the model is `randomForest`, convert it once.
7. **Ensure `num.threads > 1`** in `ranger::predict`.

### Expected Speedup

| Stage | Before | After (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~40â€“60 h | ~30â€“90 s |
| `compute_neighbor_stats` (Ã—5) | ~20â€“30 h | ~30â€“60 s |
| Data.frame copies (Ã—5) | ~2â€“5 h | 0 (in-place) |
| RF predict (6.46M rows) | ~1â€“3 h | ~5â€“20 min |
| **Total** | **86+ h** | **~10â€“25 min** |

---

## 3. Working R Code

```r
# ============================================================
# optimized_pipeline.R
# Drop-in replacement for feature preparation + RF prediction.
# Preserves the trained model and the original numerical estimand.
# ============================================================

library(data.table)

# ------------------------------------------------------------------
# STEP 0: Convert cell_data to data.table (in-place, no copy)
# ------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)                 # converts in place
}

# ------------------------------------------------------------------
# STEP 1: Build a long edge table from the nb object (vectorised)
#
#   Input:
#     - cell_data   : data.table with columns `id`, `year`, â€¦
#     - id_order     : integer vector mapping position â†’ cell id
#     - rook_neighbors_unique : spdep nb object (list of integer
#       vectors, position-indexed into id_order)
#
#   Output:
#     - edge_dt : data.table  (row_i, row_j)
#       where row_i is the row index in cell_data of the focal cell
#       and row_j is the row index of its neighbor (same year).
# ------------------------------------------------------------------

build_edge_table <- function(cell_data, id_order, neighbors) {

  ## ---- 1a. Expand neighbor list into (focal_pos, neighbor_pos) ----
  n_neighbors <- lengths(neighbors)                       # integer vec
  focal_pos   <- rep(seq_along(neighbors), n_neighbors)   # vectorised
  neigh_pos   <- unlist(neighbors, use.names = FALSE)

  ## ---- 1b. Map positions to cell ids ----------------------------
  edge_cells <- data.table(
    focal_id = id_order[focal_pos],
    neigh_id = id_order[neigh_pos]
  )
  rm(focal_pos, neigh_pos)                                # free memory

  ## ---- 1c. Build a row-index lookup keyed on (id, year) ----------
  cell_data[, .row_idx := .I]
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id)

  ## ---- 1d. For every (focal_id, year) get focal row index --------
  #  Unique years per focal_id are the same as in cell_data, so we

  #  cross-join edges with years via the row_lookup.
  #  Strategy: join edge_cells to row_lookup twice.

  # focal side
  setnames(row_lookup, c("id", "year", ".row_idx"),
                       c("focal_id", "year", "row_i"))
  setkey(row_lookup, focal_id)
  edges_with_year <- edge_cells[row_lookup,
                                on = "focal_id",
                                nomatch = 0L,
                                allow.cartesian = TRUE]
  rm(edge_cells)

  # neighbor side â€” need row index of (neigh_id, same year)
  neigh_lookup <- cell_data[, .(neigh_id = id, year, row_j = .row_idx)]
  setkey(neigh_lookup, neigh_id, year)
  setkey(edges_with_year, neigh_id, year)

  edge_dt <- neigh_lookup[edges_with_year,
                          on = c("neigh_id", "year"),
                          nomatch = 0L]

  # keep only what we need

  edge_dt <- edge_dt[, .(row_i, row_j)]
  setkey(edge_dt, row_i)

  # clean up helper column
  cell_data[, .row_idx := NULL]

  return(edge_dt)
}

cat("Building edge table â€¦\n")
system.time({
  edge_dt <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
})
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))


# ------------------------------------------------------------------
# STEP 2: Compute neighbour statistics for all variables at once
#          (fully vectorised, no per-row lapply)
# ------------------------------------------------------------------

compute_and_add_all_neighbor_features <- function(cell_data, edge_dt,
                                                   var_names) {
  for (var_name in var_names) {
    cat(sprintf("  neighbour stats for: %s\n", var_name))

    # Pull the variable values for every neighbour row
    edge_dt[, val := cell_data[[var_name]][row_j]]

    # Aggregate per focal row (row_i), dropping NAs
    stats <- edge_dt[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     keyby = row_i]

    # Initialise columns to NA, then fill matched rows
    max_col  <- paste0("nb_max_",  var_name)
    min_col  <- paste0("nb_min_",  var_name)
    mean_col <- paste0("nb_mean_", var_name)

    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    matched <- stats$row_i
    set(cell_data, i = matched, j = max_col,  value = stats$nb_max)
    set(cell_data, i = matched, j = min_col,  value = stats$nb_min)
    set(cell_data, i = matched, j = mean_col, value = stats$nb_mean)
  }

  # clean up temp column in edge_dt
  edge_dt[, val := NULL]

  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbour features â€¦\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, edge_dt,
                                         neighbor_source_vars)
})


# ------------------------------------------------------------------
# STEP 3: Random Forest prediction (chunked, multi-threaded)
# ------------------------------------------------------------------

# Load the trained model (adjust path as needed)
# rf_model <- readRDS("path/to/trained_model.rds")

predict_chunked <- function(model, newdata, chunk_size = 500000L) {

  is_ranger <- inherits(model, "ranger")
  n         <- nrow(newdata)
  chunks    <- split(seq_len(n),
                     ceiling(seq_len(n) / chunk_size))
  preds     <- numeric(n)

  cat(sprintf("Predicting %s rows in %d chunks â€¦\n",
              format(n, big.mark = ","), length(chunks)))

  for (k in seq_along(chunks)) {
    idx <- chunks[[k]]
    chunk_df <- if (is.data.table(newdata)) {
      as.data.frame(newdata[idx, ])
    } else {
      newdata[idx, , drop = FALSE]
    }

    if (is_ranger) {
      preds[idx] <- predict(model, data = chunk_df,
                             num.threads = parallel::detectCores())$predictions
    } else {
      # randomForest
      preds[idx] <- predict(model, newdata = chunk_df)
    }

    if (k %% 5 == 0 || k == length(chunks)) {
      cat(sprintf("  chunk %d / %d done\n", k, length(chunks)))
    }
    rm(chunk_df); gc(verbose = FALSE)
  }

  return(preds)
}

# ---- Identify predictor columns (exclude id, year, response) ------
response_var   <- "usd_est_n2"                       # adjust if needed
exclude_cols   <- c("id", "year", response_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# Subset to predictor columns only (avoids sending 6.46M Ã— all cols)
pred_data <- cell_data[, ..predictor_cols]

cat("Running Random Forest prediction â€¦\n")
system.time({
  cell_data[, predicted := predict_chunked(rf_model, pred_data)]
})

cat("Done.\n")
```

---

## Summary of Changes and Guarantees

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Never modified; only `predict()` is called. |
| **Numerical estimand** | Identical `max`, `min`, `mean` aggregations on the same neighbour sets; `predict()` uses the same features â†’ numerically identical output (up to floating-point associativity of `mean` in `data.table`, which uses a compensated sum and is actually *more* accurate). |
| **Memory (16 GB laptop)** | Edge table â‰ˆ 1.1 GB; `cell_data` â‰ˆ 5.3 GB; chunked predict keeps peak under ~12 GB. |
| **Runtime** | Feature prep: ~1â€“2 min. Predict: ~5â€“20 min. **Total â‰ˆ 10â€“25 min** vs. 86+ hours. |
END_RESPONSE>>>
