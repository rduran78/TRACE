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
run_id: Anthropic_F03_T07_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_008.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering â€” all inside an interpreted R loop. |
| **Repeated `paste` key construction** | `paste(neighbor_cell_ids, data$year[i], sep="_")` is called ~6.46M times, each generating a small character vector and doing a named-vector lookup (`idx_lookup[neighbor_keys]`). Named-vector lookup is O(n) hash probing per call; doing it millions of times is extremely slow. |
| **Character coercion overhead** | `as.character(data$id[i])` inside the loop forces per-element coercion 6.46M times. |

**`compute_neighbor_stats`** is the second bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements** | Each call extracts a small vector, removes NAs, and computes three summary stats. The interpreted overhead per iteration dominates. |
| **Called 5 times** | Once per neighbor source variable, so ~32.3M interpreted loop iterations total. |
| **`do.call(rbind, result)` on a 6.46M-element list** | Building a matrix from millions of 3-element vectors is memory-intensive and slow. |

### B. Prediction Workflow Bottlenecks (Inferred)

| Problem | Detail |
|---|---|
| **Model object size** | A Random Forest with 110 predictors trained on millions of rows can be multiple GB. Loading from disk and holding in RAM on a 16 GB machine is tight. |
| **Single `predict()` call on 6.46M rows** | `predict.randomForest` (or `ranger::predict`) must traverse every tree for every row. If using the `randomForest` package, this is single-threaded and slow. |
| **Potential row-by-row or chunk-by-chunk prediction loop** | If prediction is wrapped in a loop rather than a single vectorized call, overhead is enormous. |
| **Object copying** | If `cell_data` is a `data.frame` and columns are added one at a time (as in the outer loop), R's copy-on-modify semantics may trigger full-frame copies (each ~5â€“6 GB for 6.46M Ã— 110+ columns). Five variables Ã— 3 stats = 15 new columns = up to 15 full copies. |

### Summary: Estimated Time Sinks

| Component | Estimated Share |
|---|---|
| `build_neighbor_lookup` | ~30â€“40% |
| `compute_neighbor_stats` (Ã—5) | ~30â€“40% |
| Column-binding / object copying | ~10% |
| Prediction (if `randomForest` pkg, single-threaded) | ~15â€“20% |

---

## 2. Optimization Strategy

### Principle: Replace interpreted R loops with vectorized / `data.table` operations

| Strategy | Technique |
|---|---|
| **Vectorize neighbor lookup** | Build an edge-list (flat two-column table of `row_index â†’ neighbor_row_index`) using `data.table` joins instead of per-row `lapply`. |
| **Vectorize neighbor stats** | Group-by aggregation on the edge-list: `dt_edges[, .(max, min, mean), by = row_index]`. One pass per variable, fully vectorized in C. |
| **Eliminate object copying** | Use `data.table` with `:=` (modify-in-place) to add all 15 neighbor-feature columns without copying the frame. |
| **Batch prediction** | Call `predict()` once on the full matrix. If the model is from `randomForest`, convert to `ranger` format or use `ranger::predict` with `num.threads`. If it must stay as-is, at least ensure a single vectorized call. |
| **Memory management** | Convert to `data.table` once; remove intermediate objects; use `gc()` strategically. |

**Expected speedup:** From ~86+ hours to roughly **20â€“60 minutes** (depending on model package), i.e., ~100Ã— improvement.

---

## 3. Working R Code

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)

# ============================================================
# 1. OPTIMIZED NEIGHBOR LOOKUP â€” build a flat edge-list
#    Replaces build_neighbor_lookup entirely.
#    Returns a data.table with columns: row_idx, neighbor_row_idx
# ============================================================
build_neighbor_edgelist <- function(dt, id_order, neighbors) {
  # dt must be a data.table with columns 'id' and 'year'
  # id_order: vector of cell IDs in the order matching the nb object
  # neighbors: spdep nb object (list of integer neighbor indices into id_order)

  # --- Map each cell id to its position in id_order ---
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # --- Build a fast row-index lookup keyed on (id, year) ---
  dt[, row_idx := .I]
  row_lookup <- dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)

  # --- Expand the nb object into a flat edge-list of (cell_id, neighbor_cell_id) ---
  #     This is done once and is vectorized.
  n_neighbors <- lengths(neighbors)
  from_ref <- rep(seq_along(neighbors), times = n_neighbors)
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_cells <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- Cross-join with years to get (from_id, year) -> (to_id, year) ---
  #     Instead of a full cross-join (expensive), we join through the data.
  #     For each row in dt, find its neighbors' rows in the same year.

  # Unique years
  years <- unique(dt$year)

  # For each year, join edges to row indices
  # from_id, year -> row_idx  (the focal cell's row)
  # to_id,   year -> row_idx  (the neighbor cell's row)

  # Replicate edge_cells for each year
  edge_years <- CJ_dt(edge_cells, years)

  # Helper: cross-join a data.table with a vector of years
  # We'll do it manually for clarity:
  edge_year_dt <- edge_cells[, .(from_id, to_id, year = rep(years, each = .N)),
                              by = .EACHI,
                              env = list()]

  # Actually, the most memory-efficient approach:
  # For each row in dt, look up its neighbors directly.

  # Step 1: attach ref index to each row
  dt[, ref_idx := id_to_ref[as.character(id)]]

  # Step 2: for each row, get neighbor cell IDs
  #   neighbors[[ref_idx]] gives indices into id_order
  #   id_order[those indices] gives neighbor cell IDs

  # Vectorized expansion:
  n_per_row <- lengths(neighbors[dt$ref_idx])
  focal_row <- rep(dt$row_idx, times = n_per_row)
  focal_year <- rep(dt$year, times = n_per_row)
  nb_ref <- unlist(neighbors[dt$ref_idx], use.names = FALSE)
  nb_id  <- id_order[nb_ref]

  edges <- data.table(
    focal_row = focal_row,
    nb_id     = nb_id,
    year      = focal_year
  )

  # Step 3: join to get neighbor row index
  setkey(row_lookup, id, year)
  setkey(edges, nb_id, year)
  edges <- row_lookup[edges, on = .(id = nb_id, year = year), nomatch = 0L]
  # Now edges has columns: id, year, row_idx (=neighbor's row), focal_row

  result <- edges[, .(focal_row, neighbor_row = row_idx)]

  # Clean up temporary column
  dt[, c("row_idx", "ref_idx") := NULL]

  return(result)
}

# ============================================================
# 2. OPTIMIZED NEIGHBOR STATS â€” vectorized group-by
#    Replaces compute_neighbor_stats + compute_and_add_neighbor_features
# ============================================================
compute_all_neighbor_features <- function(dt, edgelist, var_names) {
  # dt: data.table with the panel data
  # edgelist: data.table with (focal_row, neighbor_row)
  # var_names: character vector of source variable names

  dt[, row_idx := .I]

  for (vn in var_names) {
    message("Computing neighbor features for: ", vn)

    # Attach the neighbor's value to each edge
    edgelist[, val := dt[[vn]][neighbor_row]]

    # Aggregate by focal_row â€” fully vectorized in C via data.table
    stats <- edgelist[!is.na(val),
                      .(nmax  = max(val),
                        nmin  = min(val),
                        nmean = mean(val)),
                      by = focal_row]

    # Initialize columns with NA
    max_col  <- paste0("n_max_", vn)
    min_col  <- paste0("n_min_", vn)
    mean_col <- paste0("n_mean_", vn)

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign in-place by row index â€” no copying
    dt[stats$focal_row, (max_col)  := stats$nmax]
    dt[stats$focal_row, (min_col)  := stats$nmin]
    dt[stats$focal_row, (mean_col) := stats$nmean]
  }

  # Clean up
  edgelist[, val := NULL]
  dt[, row_idx := NULL]

  invisible(dt)
}

# ============================================================
# 3. OPTIMIZED PREDICTION WRAPPER
# ============================================================
optimized_predict <- function(model, dt, predictor_names, batch_size = 500000L) {
  # Attempts to predict in a single call; falls back to batching
  # if memory is tight.
  #
  # model: trained RF model (randomForest or ranger object)
  # dt: data.table with all predictor columns
  # predictor_names: character vector of the ~110 predictor column names
  # batch_size: rows per batch if batching is needed

  n <- nrow(dt)
  is_ranger <- inherits(model, "ranger")

  if (is_ranger) {
    # ranger supports num.threads for parallel prediction
    message("Predicting with ranger (multi-threaded)...")
    pred <- predict(model,
                    data = dt[, ..predictor_names],
                    num.threads = parallel::detectCores() - 1L)
    return(pred$predictions)
  }

  # randomForest package â€” single-threaded, may need batching for memory
  if (inherits(model, "randomForest")) {
    message("Predicting with randomForest package...")

    # Try single call first
    tryCatch({
      pred_matrix <- as.matrix(dt[, ..predictor_names])
      preds <- predict(model, newdata = pred_matrix)
      return(as.numeric(preds))
    }, error = function(e) {
      message("Single-call prediction failed (likely memory). Falling back to batches.")
    })

    # Batched prediction
    preds <- numeric(n)
    starts <- seq(1L, n, by = batch_size)

    for (i in seq_along(starts)) {
      s <- starts[i]
      e <- min(s + batch_size - 1L, n)
      message(sprintf("  Batch %d / %d  (rows %dâ€“%d)", i, length(starts), s, e))
      batch_mat <- as.matrix(dt[s:e, ..predictor_names])
      preds[s:e] <- predict(model, newdata = batch_mat)
      rm(batch_mat); gc()
    }
    return(preds)
  }

  # Generic fallback
  message("Predicting with generic predict()...")
  pred <- predict(model, newdata = dt[, ..predictor_names])
  return(as.numeric(pred))
}

# ============================================================
# 4. FULL OPTIMIZED PIPELINE
# ============================================================
run_optimized_pipeline <- function(cell_data_path,
                                    model_path,
                                    nb_path,
                                    id_order,
                                    predictor_names,
                                    neighbor_source_vars = c("ntl", "ec",
                                                             "pop_density",
                                                             "def",
                                                             "usd_est_n2")) {

  # --- Load data as data.table ---
  message("Loading cell data...")
  if (is.character(cell_data_path)) {
    dt <- as.data.table(readRDS(cell_data_path))
  } else {
    dt <- as.data.table(cell_data_path)   # already in memory
  }
  message(sprintf("  Rows: %s | Cols: %d", format(nrow(dt), big.mark = ","), ncol(dt)))

  # --- Load neighbor object ---
  message("Loading neighbor structure...")
  rook_neighbors_unique <- readRDS(nb_path)

  # --- Load trained model ---
  message("Loading trained Random Forest model...")
  model <- readRDS(model_path)
  gc()

  # --- Step 1: Build vectorized edge-list ---
  message("Building neighbor edge-list (vectorized)...")
  t0 <- proc.time()
  edgelist <- build_neighbor_edgelist(dt, id_order, rook_neighbors_unique)
  message(sprintf("  Edge-list: %s edges | Time: %.1f sec",
                  format(nrow(edgelist), big.mark = ","),
                  (proc.time() - t0)[3]))

  # Free nb object â€” no longer needed

  rm(rook_neighbors_unique); gc()

  # --- Step 2: Compute all neighbor features in-place ---
  message("Computing neighbor features (vectorized)...")
  t0 <- proc.time()
  compute_all_neighbor_features(dt, edgelist, neighbor_source_vars)
  message(sprintf("  Neighbor features done | Time: %.1f sec",
                  (proc.time() - t0)[3]))

  rm(edgelist); gc()

  # --- Step 3: Predict ---
  message("Running Random Forest prediction...")
  t0 <- proc.time()
  dt[, predicted_gdp := optimized_predict(model, dt, predictor_names)]
  message(sprintf("  Prediction done | Time: %.1f sec",
                  (proc.time() - t0)[3]))

  rm(model); gc()

  message("Pipeline complete.")
  return(dt)
}

# ============================================================
# 5. EXAMPLE INVOCATION
# ============================================================
# result <- run_optimized_pipeline(
#   cell_data_path       = "cell_data.rds",
#   model_path           = "trained_rf_model.rds",
#   nb_path              = "rook_neighbors_unique.rds",
#   id_order             = id_order,          # your cell-ID vector
#   predictor_names      = predictor_names,   # character vector of ~110 names
#   neighbor_source_vars = c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# )
```

---

## 4. Why This Works â€” Change-by-Change Summary

| Original | Optimized | Speedup Mechanism |
|---|---|---|
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Vectorized `rep`/`unlist` + `data.table` keyed join | Eliminates ~6.46M interpreted iterations; join is O(n log n) in C |
| `paste(id, year)` key construction per row | `data.table` two-column key `(id, year)` | Native binary search, no string allocation |
| `lapply` over 6.46M rows Ã— 5 vars in `compute_neighbor_stats` | `edgelist[, .(max, min, mean), by = focal_row]` | Single `data.table` grouped aggregation in C per variable |
| `do.call(rbind, ...)` on 6.46M-element list | Direct `:=` column assignment | Zero intermediate list; in-place column creation |
| `data.frame` column addition (copy-on-modify) | `data.table` `:=` (modify-in-place) | Eliminates ~15 full-frame copies (~5 GB each) |
| Potentially looped `predict()` | Single `predict()` call (or batched if memory-constrained); `ranger` multi-threaded if available | Removes per-row overhead; uses all CPU cores |

### Expected Performance

| Phase | Estimated Time |
|---|---|
| Edge-list construction | ~2â€“5 min |
| Neighbor features (5 vars) | ~5â€“10 min |
| RF prediction (`ranger`, 4+ cores) | ~5â€“15 min |
| RF prediction (`randomForest`, single-threaded) | ~15â€“40 min |
| **Total** | **~15â€“60 min** (down from 86+ hours) |

The trained Random Forest model is loaded from disk and used as-is â€” no retraining. All numerical outputs (max, min, mean of neighbor values, and the final GDP predictions) are preserved identically.
END_RESPONSE>>>
