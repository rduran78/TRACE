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
run_id: Anthropic_F03_T10_R004
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 4
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_004.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

There are two distinct bottleneck zones in this pipeline:

### Bottleneck 1: Feature Preparation (`build_neighbor_lookup` + `compute_neighbor_stats`)

1. **`build_neighbor_lookup` creates ~6.46 million R lists via `lapply`**, each performing character key construction (`paste`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. With ~1.37M neighbor relationships spread across 28 years, the total number of string-match operations is enormous. The per-row overhead of string allocation, named-vector lookup (linear or hash), and `as.integer` coercion accumulates to hours.

2. **`compute_neighbor_stats` uses `lapply` over 6.46M elements**, each subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end creates a 6.46M-row matrix from a list of 6.46M 3-element vectors â€” this is an O(nÂ²) memory-copy pattern in R. This function is called **5 times** (once per neighbor source variable), multiplying the cost.

3. **String-keyed lookups are the core anti-pattern.** The entire neighbor lookup is mediated by `paste(..., sep="_")` keys matched against a named vector. This is orders of magnitude slower than integer-indexed operations.

### Bottleneck 2: Random Forest Prediction

1. **Predicting 6.46M rows with ~110 features through a Random Forest is inherently expensive.** If done in a single `predict()` call, memory for the prediction matrix (~5.3 GB for a dense numeric matrix) can exceed available RAM, triggering swapping.

2. **If prediction is done row-by-row or in a tight R loop**, the overhead per call to `predict.randomForest` or `predict.ranger` (model object copying, data.frame coercion, factor-level checking) dominates. Even chunked prediction needs appropriately sized chunks.

3. **Model loading**: If the serialized RF model is large (hundreds of MB to multiple GB), repeated `readRDS` calls or unnecessary copies will thrash memory.

---

## Optimization Strategy

| Component | Problem | Solution |
|---|---|---|
| Neighbor lookup | String keys, per-row `lapply` | Replace with pure integer arithmetic: `(id_index - 1) * n_years + year_index`. Pre-expand the spatial `nb` object to cell-year integer row indices in vectorized form using `data.table`. |
| Neighbor stats | 6.46M-element `lapply` + `do.call(rbind,...)` | Vectorized grouped aggregation via `data.table`: explode neighbor pairs into an edge table, join variable values, then `group by` source row to compute max/min/mean in C-level code. |
| Feature binding | Repeated `cell_data <- ...` column addition (copy-on-modify) | Use `data.table` set-by-reference (`:=`) to add columns in place â€” zero copies. |
| RF prediction | Single giant call or row-level loop | Chunk prediction into ~500K-row batches; pre-allocate output vector; use `ranger` or `predict(..., num.threads)` if available. |
| Memory | Multiple large intermediate objects | Reuse edge table across variables; `rm()` + `gc()` intermediaries; never duplicate the model object. |

**Expected speedup:** From 86+ hours to roughly **15â€“45 minutes** depending on model size and disk I/O.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================
# Prerequisites:
#   - data.table, ranger (or randomForest)
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec,
#                pop_density, def, usd_est_n2, ... (all predictor columns)
#   - id_order: integer vector of unique cell IDs in the order matching
#               rook_neighbors_unique
#   - rook_neighbors_unique: nb object (list of integer index vectors into id_order)
#   - rf_model: pre-trained Random Forest model (already loaded or to be loaded once)
# =============================================================================

library(data.table)

# ---- 0. Load model ONCE, keep in memory ----
# rf_model <- readRDS("path/to/trained_rf_model.rds")   # do this once at top

# ---- 1. Convert cell_data to data.table (by reference if possible) ----
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place, no copy
}

# ---- 2. Build integer-indexed neighbor edge table (ONCE) ----
# This replaces build_neighbor_lookup entirely.

build_neighbor_edge_table <- function(cell_data, id_order, nb_obj) {
  # Map each unique cell ID to its position in id_order
  n_ids <- length(id_order)

  # Build spatial edge list: source_spatial_idx -> neighbor_spatial_idx
  # nb_obj[[i]] gives the integer indices (into id_order) of neighbors of
  # id_order[i]. We expand this into a two-column integer matrix.
  src_idx <- rep(
    seq_len(n_ids),
    times = lengths(nb_obj)
  )
  dst_idx <- unlist(nb_obj, use.names = FALSE)

  # Now translate spatial indices to actual cell IDs
  src_id <- id_order[src_idx]
  dst_id <- id_order[dst_idx]

  # Create edge table at the spatial level
  spatial_edges <- data.table(src_id = src_id, dst_id = dst_id)

  # Create a row-index lookup in cell_data: (id, year) -> row number

  cell_data[, .row_idx := .I]

  # Cross-join spatial edges with years present in data:
  # For every (src_id, dst_id) pair, we need every year where BOTH exist.
  # Efficient approach: join edges to cell_data twice (on src and dst).

  # Keyed lookup tables
  src_lookup <- cell_data[, .(src_row = .row_idx, year = year, src_id = id)]
  setkey(src_lookup, src_id, year)

  dst_lookup <- cell_data[, .(dst_row = .row_idx, year = year, dst_id = id)]
  setkey(dst_lookup, dst_id, year)

  # Join: for each spatial edge, find all years where src exists
  setkey(spatial_edges, src_id)
  edges_with_year <- src_lookup[spatial_edges, on = .(src_id), nomatch = 0L,
                                 allow.cartesian = TRUE]
  # edges_with_year now has: src_id, year, src_row, dst_id

  # Now join to find dst_row for the same year
  setkey(edges_with_year, dst_id, year)
  full_edges <- dst_lookup[edges_with_year, on = .(dst_id, year), nomatch = 0L]
  # full_edges has: dst_id, year, dst_row, src_id, src_row

  # Clean up temporary column
  cell_data[, .row_idx := NULL]

  # Return minimal edge table: src_row, dst_row (integer row indices into cell_data)
  full_edges[, .(src_row = src_row, dst_row = dst_row)]
}

cat("Building neighbor edge table...\n")
system.time({
  edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
})
# edge_table: each row says "cell_data row src_row has neighbor at cell_data row dst_row"
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_table), big.mark = ",")))


# ---- 3. Vectorized neighbor feature computation (replaces compute_neighbor_stats) ----

add_neighbor_features_vectorized <- function(cell_data, edge_table, var_names) {
  # For each variable, pull the neighbor values via edge_table,
  # then group by src_row to compute max, min, mean.
  # All done in data.table C-level grouped operations.

  for (vname in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", vname))

    # Extract neighbor values via integer indexing (extremely fast)
    edge_table[, val := cell_data[[vname]][dst_row]]

    # Grouped aggregation
    stats <- edge_table[!is.na(val),
                        .(nmax  = max(val),
                          nmin  = min(val),
                          nmean = mean(val)),
                        by = src_row]

    # Pre-fill with NA
    col_max  <- paste0("n_max_",  vname)
    col_min  <- paste0("n_min_",  vname)
    col_mean <- paste0("n_mean_", vname)

    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    # Fill computed values by reference (no copy)
    set(cell_data, i = stats$src_row, j = col_max,  value = stats$nmax)
    set(cell_data, i = stats$src_row, j = col_min,  value = stats$nmin)
    set(cell_data, i = stats$src_row, j = col_mean, value = stats$nmean)
  }

  # Clean up temp column in edge_table
  edge_table[, val := NULL]

  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  add_neighbor_features_vectorized(cell_data, edge_table, neighbor_source_vars)
})

# Free edge table memory now that features are computed
rm(edge_table)
gc()


# ---- 4. Chunked Random Forest Prediction ----

predict_rf_chunked <- function(model, newdata, chunk_size = 500000L,
                               num_threads = NULL) {
  # newdata: data.table with all predictor columns
  # Returns: numeric vector of predictions, same length as nrow(newdata)

  n <- nrow(newdata)
  predictions <- numeric(n)   # pre-allocate full output vector

  # Determine which columns the model needs
  # For ranger: model$forest$independent.variable.names

  # For randomForest: names(model$forest$xlevels) or colnames(model$forest$xlevels)
  # We pass the full data and let predict() select the right columns.

  n_chunks <- ceiling(n / chunk_size)
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))

  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)

    chunk <- newdata[start_idx:end_idx]

    # Detect model type and predict accordingly
    if (inherits(model, "ranger")) {
      if (!is.null(num_threads)) {
        pred <- predict(model, data = chunk, num.threads = num_threads)$predictions
      } else {
        pred <- predict(model, data = chunk)$predictions
      }
    } else if (inherits(model, "randomForest")) {
      pred <- as.numeric(predict(model, newdata = chunk))
    } else {
      # Generic fallback
      pred <- as.numeric(predict(model, newdata = chunk))
    }

    predictions[start_idx:end_idx] <- pred

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %s-%s)\n",
                  i, n_chunks,
                  format(start_idx, big.mark = ","),
                  format(end_idx, big.mark = ",")))
    }
  }

  predictions
}

# --- Run prediction ---
cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model      = rf_model,
    newdata    = cell_data,
    chunk_size = 500000L,
    num_threads = parallel::detectCores() - 1L   # ignored if not ranger
  )]
})

cat("Done.\n")
```

---

## Summary of Changes and Why They Work

| Original | Optimized | Why |
|---|---|---|
| `paste(id, year)` string keys for 6.46M rows | Integer row-index edge table built once via `data.table` joins | Eliminates ~billions of string allocations and hash lookups |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Vectorized `rep`/`unlist` + `data.table` keyed join | C-level vectorized operations, no R-level loop overhead |
| `lapply` + `do.call(rbind, ...)` in `compute_neighbor_stats` (called 5Ã—) | `data.table` grouped `[, .(max, min, mean), by=src_row]` (called 5Ã—) | Grouped aggregation in C via GForce; no list creation or row-binding |
| `cell_data <- cbind(...)` (copy-on-modify, 15 times for 5 vars Ã— 3 stats) | `set(cell_data, j=..., value=...)` by reference | Zero-copy column addition; no 6.46M-row data.frame duplication |
| Single monolithic `predict()` on 6.46M rows (potential OOM) | Chunked prediction in 500K-row batches with pre-allocated output | Keeps peak memory under control; no intermediate concatenation |
| Possible repeated `readRDS` of model | Load model once, pass by reference | R does not copy large objects on read-access (copy-on-modify semantics) |

**Numerical equivalence is preserved**: the same max/min/mean aggregations are computed on exactly the same neighbor sets, and the trained RF model is used as-is with `predict()` â€” no retraining, no rounding, no approximation.
END_RESPONSE>>>
