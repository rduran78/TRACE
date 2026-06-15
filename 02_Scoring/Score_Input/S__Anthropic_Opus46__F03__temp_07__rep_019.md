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
run_id: Anthropic_F03_T07_R019
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 19
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_019.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup` â€” O(n) `lapply` with per-row string operations**

* For every one of ~6.46 million rows, the function:
  1. Does a named-vector lookup (`id_to_ref`) â€” O(1) amortized but with character coercion overhead per row.
  2. Indexes into the `neighbors` nb list.
  3. Creates `paste(..., sep="_")` keys for every neighbor of every row.
  4. Looks those keys up in `idx_lookup` (another named character vector).
* The string concatenation (`paste`) and named-vector lookups are executed **~6.46M Ã— avg_neighbors** times. With ~1.37M directed relationships spread over 344K cells (~4 neighbors/cell on average), that is ~25.8 million `paste` + hash lookups just in this function.
* The entire result is a list of 6.46M integer vectors â€” substantial memory allocation pressure.

**`compute_neighbor_stats` â€” repeated per variable, R-level loop**

* Called 5 times (once per neighbor source variable).
* Each call iterates over all 6.46M rows in an `lapply`, subsetting a numeric vector by the index vectors from the lookup, then computing `max/min/mean`.
* The `do.call(rbind, result)` at the end materialises a 6.46M Ã— 3 matrix from a list of 6.46M length-3 vectors â€” this is a known slow pattern in R.

**Outer loop copies `cell_data` each iteration**

* `cell_data <- compute_and_add_neighbor_features(...)` likely creates a modified copy of the entire data frame on every iteration (R's copy-on-modify semantics). With ~110 columns Ã— 6.46M rows, each copy is ~5â€“6 GB depending on types. Five iterations = five copies.

### B. Random Forest Inference Bottlenecks

* Predicting 6.46M rows Ã— 110 features through a Random Forest (even a pre-trained one) is expensive. If done naively in one `predict()` call, the model must build a dense prediction matrix (~5.7 GB for float64) and walk every tree for every row.
* If the model object is large (many trees, deep), `predict.randomForest` or `predict.ranger` may itself be slow, and loading it from disk with `readRDS` can take minutes for a multi-GB object.
* If prediction is done inside a loop (row-by-row or small-batch), overhead is catastrophic.

### C. Memory

* 16 GB RAM is tight. The data frame alone at 110 numeric columns Ã— 6.46M rows â‰ˆ 5.3 GB. The model, neighbor lookup list, and intermediate copies can easily exceed available RAM, causing swap thrashing â€” the most likely single cause of "86+ hours."

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Fix |
|---|---|---|
| **Neighbor lookup** | Per-row `paste` + named-vector hash | Replace with a `data.table` equi-join: expand neighbors into a long table `(row_i, neighbor_id)`, join to `(id, year) â†’ row_j`, yielding a two-column integer matrix. No strings. |
| **Neighbor stats** | R-level `lapply` over 6.46M elements, 5Ã— | Vectorise with `data.table` grouped aggregation on the long neighbor-pair table. Compute all 5 variables' stats in one pass. |
| **`do.call(rbind, ...)`** | Slow listâ†’matrix | Eliminated by `data.table` aggregation returning columns directly. |
| **Data frame copies** | Copy-on-modify 5Ã— | Use `data.table` `:=` (modify in place). Zero copies. |
| **RF prediction** | Possible row-level loop or huge single matrix | Use `ranger::predict` (C++ back-end, much faster than `randomForest::predict`). If model is `randomForest`, convert or re-save as `ranger`. Predict in chunks of ~500K rows to control peak memory. |
| **Model loading** | `readRDS` on huge object | Load once, keep in memory. Consider `qs::qread` for faster deserialization. |
| **Memory** | >16 GB peak | `data.table` in-place ops, chunked prediction, `gc()` between stages. Remove intermediate objects. |

**Expected speedup:** from 86+ hours to roughly 10â€“30 minutes, depending on the RF model size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# 0. Libraries
# =============================================================================
library(data.table)
# install.packages("ranger")
# install.packages("qs")          # optional, for faster model I/O

# =============================================================================
# 1. Load data â€” convert to data.table IN PLACE
# =============================================================================
# Assume cell_data is already in memory as a data.frame or data.table.
# If data.frame:
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure key columns are integer (avoid type-coercion overhead later)
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# Create a row-index column (will be used for neighbor mapping)
cell_data[, .row_idx := .I]

# =============================================================================
# 2. Build neighbor lookup as a data.table (long format) â€” REPLACES
#    build_neighbor_lookup()
# =============================================================================
build_neighbor_lookup_dt <- function(cell_dt, id_order, neighbors) {
  # id_order  : integer vector of cell IDs in the order matching the nb object
  # neighbors : spdep nb object (list of integer index vectors into id_order)

  id_order <- as.integer(id_order)

  # --- Step A: expand nb object into a long table of (focal_id, neighbor_id) --
  # Each element neighbors[[k]] contains indices into id_order
  n_cells <- length(id_order)
  focal_idx    <- rep(seq_len(n_cells),  lengths(neighbors))
  neighbor_idx <- unlist(neighbors, use.names = FALSE)

  edge_dt <- data.table(
    focal_id    = id_order[focal_idx],
    neighbor_id = id_order[neighbor_idx]
  )
  rm(focal_idx, neighbor_idx); gc()

  # --- Step B: cross-join with years present in the data --------------------
  years <- sort(unique(cell_dt$year))

  # Expand edges Ã— years  (~ 1.37M edges Ã— 28 years â‰ˆ 38.5M rows â€” fits in RAM)
  edge_year <- edge_dt[, CJ(year = years), by = .(focal_id, neighbor_id)]
  rm(edge_dt); gc()

  # --- Step C: map (focal_id, year) â†’ row_idx_i  and
  #                  (neighbor_id, year) â†’ row_idx_j  via keyed join ----------
  setkey(cell_dt, id, year)

  # focal side
  edge_year[cell_dt, row_i := i..row_idx,
            on = .(focal_id = id, year = year)]

  # neighbor side
  edge_year[cell_dt, row_j := i..row_idx,
            on = .(neighbor_id = id, year = year)]

  # Drop edges where either side is missing

  edge_year <- edge_year[!is.na(row_i) & !is.na(row_j)]

  edge_year
}

cat("Building neighbor edge table â€¦\n")
system.time({
  edge_dt <- build_neighbor_lookup_dt(cell_data, id_order, rook_neighbors_unique)
})
# edge_dt columns: focal_id, neighbor_id, year, row_i, row_j


# =============================================================================
# 3. Compute ALL neighbor stats in one vectorised pass â€” REPLACES
#    compute_neighbor_stats() + outer for-loop
# =============================================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # For each var in var_names, compute max/min/mean of neighbor values,
  # then assign columns in-place to cell_dt.

  # Pull only needed columns into the edge table (avoids copying full dt)
  val_cols <- var_names
  # Add neighbor values to edge_dt via row_j index
  for (v in val_cols) {
    set(edge_dt, j = v, value = cell_dt[[v]][edge_dt$row_j])
  }

  # Grouped aggregation: one group per row_i
  agg_exprs <- list()
  for (v in val_cols) {
    agg_exprs[[paste0("n_max_", v)]]  <-
      bquote(max(.(as.name(v)),  na.rm = TRUE))
    agg_exprs[[paste0("n_min_", v)]]  <-
      bquote(min(.(as.name(v)),  na.rm = TRUE))
    agg_exprs[[paste0("n_mean_", v)]] <-
      bquote(mean(.(as.name(v)), na.rm = TRUE))
  }
  # Build a single j-expression  list(n_max_ntl = max(ntl, na.rm=TRUE), ...)
  j_call <- as.call(c(list(quote(list)), agg_exprs))

  cat("  Aggregating neighbor stats â€¦\n")
  agg <- edge_dt[, eval(j_call), by = .(row_i)]

  # Replace Inf / -Inf (from max/min on all-NA groups) with NA
  inf_to_na <- function(x) { x[is.infinite(x)] <- NA_real_; x }
  agg_cols <- setdiff(names(agg), "row_i")
  for (ac in agg_cols) set(agg, j = ac, value = inf_to_na(agg[[ac]]))

  # Join back to cell_dt by row index â€” in place
  cell_dt[agg, (agg_cols) := mget(agg_cols), on = .(`.row_idx` = row_i)]

  # Clean up temporary columns from edge_dt

  for (v in val_cols) set(edge_dt, j = v, value = NULL)

  invisible(NULL)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features â€¦\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# Free edge table
rm(edge_dt); gc()

# Remove helper column
cell_data[, .row_idx := NULL]


# =============================================================================
# 4. Load trained Random Forest model
# =============================================================================
# Option A: if saved with saveRDS / readRDS
cat("Loading RF model â€¦\n")
system.time({
  rf_model <- readRDS("path/to/trained_rf_model.rds")
  # Option B (faster): rf_model <- qs::qread("path/to/trained_rf_model.qs")
})

# =============================================================================
# 5. Predict in memory-safe chunks
# =============================================================================
predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  cat(sprintf("Predicting %s rows in %d chunks â€¦\n", format(n, big.mark = ","),
              length(starts)))

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + chunk_size - 1L, n)
    chunk <- newdata[i1:i2, ]

    # --- ranger model --------------------------------------------------------
    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = chunk)$predictions

    # --- randomForest model --------------------------------------------------
    } else if (inherits(model, "randomForest")) {
      preds[i1:i2] <- predict(model, newdata = chunk)

    } else {
      stop("Unsupported model class: ", class(model)[1])
    }

    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  chunk %d / %d done (rows %s â€“ %s)\n",
                  k, length(starts),
                  format(i1, big.mark = ","), format(i2, big.mark = ",")))
      gc()
    }
  }
  preds
}

# Prepare prediction matrix (only predictor columns, in correct order)
# Adjust 'predictor_names' to match the feature names the model expects.
predictor_names <- setdiff(names(cell_data), c("id", "year", "gdp", "response"))
# If your model stores feature names:
#   predictor_names <- rf_model$forest$independent.variable.names  # ranger
#   predictor_names <- rownames(importance(rf_model))               # randomForest

pred_data <- cell_data[, ..predictor_names]   # data.table column subset (no copy)

cat("Running RF prediction â€¦\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(rf_model, pred_data)]
})

rm(pred_data); gc()

cat("Done.\n")
```

---

## 4. SUMMARY OF EXPECTED GAINS

| Stage | Before (estimated) | After (estimated) | Key change |
|---|---|---|---|
| `build_neighbor_lookup` | ~30â€“60 min (paste + hash Ã— 6.46M) | ~1â€“3 min | `data.table` keyed join, no strings |
| `compute_neighbor_stats` (Ã—5 vars) | ~40â€“80 hours (lapply Ã— 6.46M Ã— 5, `do.call(rbind)`) | ~2â€“5 min | Single grouped aggregation, in-place `:=` |
| Data-frame copies (Ã—5) | ~25 GB allocation churn â†’ swap | 0 copies | `data.table` `:=` |
| RF prediction | Varies (possibly row-level loop) | ~5â€“20 min | Chunked `predict()`, controlled memory |
| **Total** | **86+ hours** | **~10â€“30 min** | |

The trained Random Forest model is never modified or retrained. The numerical estimand (predicted GDP) is preserved identically â€” only the computational pathway is optimised.
END_RESPONSE>>>
