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
run_id: Anthropic_F03_T00_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_0.0/output_F03_Anthropic_temp_00_rep_024.md

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
| **Named-vector lookup via `idx_lookup[neighbor_keys]`** | `idx_lookup` is a named character vector of length 6.46M. Subsetting a named vector is O(n) per call in base R (linear scan or hash miss). Called ~6.46M times â†’ quadratic-class wall time. |
| **Massive character allocation** | `paste(â€¦, sep="_")` creates ~1.37M temporary strings *per row-batch*, and `as.character()` is called repeatedly. |
| **Output is a list of 6.46M integer vectors** | ~50 GB of list overhead (each list element is a separate SEXP with 64-byte header). |

**`compute_neighbor_stats`** compounds the problem:

- Another `lapply` over 6.46M elements, each subsetting a numeric vector by variable-length index.
- Called 5 times (once per neighbor source variable) â†’ ~32.3M interpreted iterations.
- `do.call(rbind, result)` on a 6.46M-element list is itself slow (repeated reallocation).

**Net effect on feature prep alone:** The nested character-key lookups and per-row R-level loops easily account for tens of hours on 6.46M rows.

### B. Random Forest Inference Bottlenecks

| Problem | Detail |
|---|---|
| **Single `predict()` call on 6.46M Ã— 110 matrix** | `ranger`/`randomForest` `predict` will try to allocate the full prediction matrix in RAM. With 110 features Ã— 6.46M rows Ã— 8 bytes â‰ˆ 5.7 GB just for the input matrix, plus internal tree-traversal buffers. On a 16 GB laptop this risks swapping. |
| **Object copying** | If `cell_data` is a `data.frame`, every `cell_data$new_col <- â€¦` triggers a full copy (R's copy-on-modify). With ~110 columns Ã— 6.46M rows â‰ˆ 5.7 GB, each column addition copies the entire frame. Adding 15 neighbor-stat columns (5 vars Ã— 3 stats) means ~85 GB of cumulative copying. |
| **Model loading** | If the model is re-loaded from disk inside a loop, deserialization of a large RF object is expensive. Should be loaded once. |

### C. Summary of Root Causes (ranked)

1. **Copy-on-modify of the data.frame** during column additions (~85 GB wasted copies).
2. **O(nÂ²)-class character-key lookup** in `build_neighbor_lookup`.
3. **6.46M Ã— 5 interpreted R loops** in `compute_neighbor_stats`.
4. **Monolithic predict call** risking RAM exhaustion and swap thrashing.

---

## 2. Optimization Strategy

| # | Technique | Expected Speedup |
|---|---|---|
| 1 | **Use `data.table` with set-by-reference** (`set()` / `:=`) to eliminate all copy-on-modify. | Eliminates ~85 GB of copies â†’ massive. |
| 2 | **Replace character-key lookup with integer-key lookup** using `data.table` keyed join or direct integer indexing. Build the neighbor lookup as a two-column `data.table` (row_idx, neighbor_row_idx) â€” fully vectorized. | Replaces 6.46M R-level iterations with one vectorized join. |
| 3 | **Vectorize `compute_neighbor_stats`** using `data.table` grouped aggregation on the edge list. | Replaces 32.3M R-level iterations with 5 grouped aggregations. |
| 4 | **Chunk the `predict()` call** into batches of ~500K rows to keep peak RAM under control. | Prevents swap thrashing on 16 GB machine. |
| 5 | **Load the model once**, keep in memory. | Trivial but important. |

Estimated new runtime: **minutes**, not hours.

---

## 3. Working R Code

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)
# Use ranger for prediction; works identically with randomForest
# if you replace predict calls accordingly.
# library(ranger)   # if model is ranger
# library(randomForest)  # if model is randomForest

# ============================================================
# 1. Load model ONCE
# ============================================================
rf_model <- readRDS("path/to/trained_rf_model.rds")

# ============================================================
# 2. Convert to data.table (by reference if already a data.table)
# ============================================================
setDT(cell_data)

# Ensure id and year columns exist and are integer-typed for speed
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# ============================================================
# 3. Build vectorized neighbor edge-list
#    (replaces build_neighbor_lookup entirely)
# ============================================================
build_neighbor_edgelist <- function(cell_dt, id_order, neighbors) {
  # id_order : integer vector of cell IDs in the order matching
  #            the spdep nb object (index i â†’ id_order[i])
  # neighbors: spdep nb list (rook_neighbors_unique)

  # --- Step A: expand nb list into an edge table (cell-ID level) ---
  n_nb <- lengths(neighbors)
  from_ref <- rep(seq_along(neighbors), times = n_nb)
  to_ref   <- unlist(neighbors, use.names = FALSE)

  edge_ids <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # --- Step B: map (id, year) â†’ row index in cell_dt ---
  cell_dt[, .row_idx := .I]
  id_year_key <- cell_dt[, .(id, year, .row_idx)]

  # --- Step C: cross-join edges with all years (vectorized) ---
  years <- sort(unique(cell_dt$year))
  edge_full <- edge_ids[, CJ(from_id, to_id, year = years),
                        on = .(from_id, to_id)]
  # But CJ on two existing columns isn't direct; instead:
  edge_full <- CJ(edge_idx = seq_len(nrow(edge_ids)), year = years)
  edge_full[, `:=`(from_id = edge_ids$from_id[edge_idx],
                    to_id   = edge_ids$to_id[edge_idx])]
  edge_full[, edge_idx := NULL]

  # --- Step D: attach row indices for "from" and "to" ---
  setnames(id_year_key, c("id", "year", ".row_idx"),
           c("from_id", "year", "from_row"))
  edge_full <- id_year_key[edge_full, on = .(from_id, year), nomatch = 0L]

  setnames(id_year_key, c("from_id", "year", "from_row"),
           c("to_id", "year", "to_row"))
  edge_full <- id_year_key[edge_full, on = .(to_id, year), nomatch = 0L]

  # Clean up
  setnames(id_year_key, c("to_id", "year", "to_row"),
           c("id", "year", ".row_idx"))  # restore
  cell_dt[, .row_idx := NULL]

  edge_full  # columns: from_id, to_id, year, from_row, to_row
}

cat("Building neighbor edge-list â€¦\n")
edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
cat(sprintf("  Edge-list rows: %s\n", format(nrow(edge_dt), big.mark = ",")))

# ============================================================
# 4. Vectorized neighbor-stat computation
#    (replaces compute_neighbor_stats + outer loop)
# ============================================================
compute_and_add_all_neighbor_features <- function(cell_dt, edge_dt,
                                                   source_vars) {
  # Attach a row-index to cell_dt for reference
  cell_dt[, .row_idx := .I]

  for (var in source_vars) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var))

    # Pull the variable values for the "to" (neighbor) rows
    edge_dt[, val := cell_dt[[var]][to_row]]

    # Grouped aggregation: one pass per variable
    stats <- edge_dt[!is.na(val),
                     .(nb_max  = max(val),
                       nb_min  = min(val),
                       nb_mean = mean(val)),
                     keyby = .(from_row)]

    # Pre-fill with NA, then update by reference
    max_col  <- paste0("nb_max_",  var)
    min_col  <- paste0("nb_min_",  var)
    mean_col <- paste0("nb_mean_", var)

    set(cell_dt, j = max_col,  value = NA_real_)
    set(cell_dt, j = min_col,  value = NA_real_)
    set(cell_dt, j = mean_col, value = NA_real_)

    idx <- stats$from_row
    set(cell_dt, i = idx, j = max_col,  value = stats$nb_max)
    set(cell_dt, i = idx, j = min_col,  value = stats$nb_min)
    set(cell_dt, i = idx, j = mean_col, value = stats$nb_mean)
  }

  edge_dt[, val := NULL]
  cell_dt[, .row_idx := NULL]
  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features â€¦\n")
compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
cat("Done.\n")

# Free the edge list if memory is tight
# rm(edge_dt); gc()

# ============================================================
# 5. Chunked Random Forest prediction
#    (prevents RAM exhaustion on 16 GB laptop)
# ============================================================
predict_rf_chunked <- function(model, dt, feature_cols,
                                chunk_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  cat(sprintf("Predicting %s rows in %d chunks â€¦\n",
              format(n, big.mark = ","), length(starts)))

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + chunk_size - 1L, n)

    chunk <- dt[i1:i2, ..feature_cols]

    # --- ranger model ---
    # preds[i1:i2] <- predict(model, data = chunk)$predictions

    # --- randomForest model ---
    # preds[i1:i2] <- predict(model, newdata = chunk)

    # Generic: try ranger first, fall back to randomForest
    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = chunk)$predictions
    } else {
      preds[i1:i2] <- predict(model, newdata = chunk)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  chunk %d / %d  (rows %s â€“ %s)\n",
                  k, length(starts),
                  format(i1, big.mark = ","),
                  format(i2, big.mark = ",")))
    }
  }

  preds
}

# Identify the feature columns the model expects
# (Adjust this to match your actual feature column names)
feature_cols <- setdiff(names(cell_data), c("id", "year", "gdp", "gdp_pred"))

cat("Running Random Forest prediction â€¦\n")
cell_data[, gdp_pred := predict_rf_chunked(rf_model, cell_data,
                                            feature_cols,
                                            chunk_size = 500000L)]
cat("Prediction complete.\n")
```

---

## 4. Why This Preserves Correctness

| Requirement | How it is met |
|---|---|
| **Trained RF model unchanged** | `rf_model` is loaded once from disk and only used in `predict()`. No retraining. |
| **Original numerical estimand preserved** | The neighbor statistics (max, min, mean) are computed with identical semantics â€” same grouping, same NA handling. The `data.table` grouped aggregation produces bit-identical results to the original `lapply` code. The RF `predict()` function is deterministic for a fixed model and fixed input. |
| **Column-addition semantics** | `set()` modifies the `data.table` **in place** â€” no copy-on-modify, no reordering, no dropped rows. |

---

## 5. Expected Performance

| Stage | Original | Optimized | Reason |
|---|---|---|---|
| `build_neighbor_lookup` | ~hours (6.46M char lookups) | ~30â€“60 s (vectorized join) | Integer keyed join, no per-row R loop |
| `compute_neighbor_stats` (Ã—5) | ~hours (32.3M R iterations) | ~30â€“90 s (5 grouped aggs) | `data.table` radix-sort grouping |
| Column additions (Ã—15) | ~85 GB copies | 0 extra copies | `set()` by reference |
| `predict()` | OOM risk / swap | ~5â€“15 min (chunked) | 500K-row chunks fit in RAM |
| **Total** | **86+ hours** | **~15â€“30 minutes** | |
END_RESPONSE>>>
