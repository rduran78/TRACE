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
run_id: Anthropic_F03_T07_R006
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 6
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_006.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Why it hurts |
|---|---|
| **Row-level `lapply` over 6.46 M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering â€” all interpreted R. |
| **`paste(id, year)` string keys + named-vector lookup** | For 6.46 M rows Ã— ~4 neighbors each â‰ˆ 25 M string constructions and hash lookups. String hashing is slow and memory-hungry. |
| **Output is a 6.46 Mâ€“element list of integer vectors** | ~200â€“400 MB of list overhead alone; every element is a separately allocated SEXP. |

**`compute_neighbor_stats`** compounds the problem:

| Problem | Why it hurts |
|---|---|
| **Another `lapply` over 6.46 M rows** | Each call subsets a numeric vector, removes NAs, computes three summary stats â€” 5 times (once per variable). That is ~32 M interpreted iterations total. |
| **`do.call(rbind, result)` on a 6.46 Mâ€“element list** | Materialises a huge list-of-vectors into a matrix â€” known to be O(nÂ²) for large lists in base R. |

**Combined cost estimate for feature prep alone: tens of hours on the stated hardware.**

### B. Prediction / Inference Bottlenecks

| Problem | Why it hurts |
|---|---|
| **Predicting 6.46 M rows at once with `predict.randomForest` (or `ranger::predict`)** | `randomForest::predict` builds an N Ã— ntree matrix internally. With 6.46 M rows and (say) 500 trees, that is a ~24 GB dense matrix â€” exceeds 16 GB RAM, triggers swapping or crashes. |
| **Model loaded from disk every run (if applicable)** | A large RF model can be 1â€“4 GB; repeated `readRDS` is wasteful. |
| **No chunked prediction** | Even with `ranger`, a single monolithic predict call peaks at high memory. |
| **Potential `data.frame` copies** | Adding columns one-at-a-time with `cell_data$new_col <- ...` can trigger full-frame copies under R's copy-on-modify semantics. |

### C. Summary of Root Causes (ordered by impact)

1. **`build_neighbor_lookup`**: millions of string operations in interpreted R.
2. **`compute_neighbor_stats`**: millions of per-row `lapply` calls, repeated 5Ã—.
3. **`do.call(rbind, ...)` on million-element lists**.
4. **Monolithic RF prediction** on 6.46 M rows exceeds RAM.
5. **Repeated column-binding** triggers copy-on-modify.

---

## 2. Optimization Strategy

### Feature Preparation â†’ Vectorised `data.table` Joins

Replace the entire lookup + stats pipeline with:

1. **Build a flat edge table** (`cell_id â†’ neighbor_id`) from the `nb` object â€” one-time, vectorised.
2. **Join the edge table to the panel data** by `(neighbor_id, year)` using `data.table` keyed joins â€” O(n log n), fully vectorised in C.
3. **Group-by aggregate** `(cell_id, year)` to compute `max`, `min`, `mean` â€” single pass per variable, fully vectorised.

This eliminates all `lapply`, all `paste`-key lookups, and all `do.call(rbind, ...)`.

### Prediction â†’ Chunked Inference

Split the 6.46 M rows into chunks (~500 K each) and call `predict()` per chunk, then `rbind` the results. This keeps peak memory well within 16 GB.

### Additional

- Use `data.table::set()` or pre-allocate columns to avoid copy-on-modify.
- Load the model once with `readRDS` and reuse the in-memory object.
- If the model is `randomForest`, consider converting to `ranger` format (or simply use the model as-is with chunked predict).

**Expected speedup: from 86+ hours â†’ roughly 10â€“30 minutes** (feature prep drops from hours to minutes; prediction from hours to minutes).

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMISED PIPELINE â€” Feature Preparation + Chunked RF Prediction
# =============================================================================
# Requirements: data.table (>= 1.14), ranger or randomForest (whichever was
# used to train the model).
# The trained model object and the original numerical estimand are preserved
# exactly â€” we only change how features are built and how predict() is called.
# =============================================================================

library(data.table)

# ---- 0. Load artefacts (do this ONCE) ----------------------------------------

# Load model once; keep in memory for all subsequent predictions.
# Adjust the path to wherever the model is serialised.
rf_model <- readRDS("trained_rf_model.rds")          # load once
rook_neighbors_unique <- readRDS("rook_neighbors.rds") # spdep nb object
# cell_data is assumed to already exist as a data.frame / data.table
# id_order is assumed to already exist (vector of cell IDs in nb-object order)


# ---- 1. Build flat edge table from nb object (vectorised, one-time) ----------

build_edge_table <- function(id_order, nb_obj) {
  # nb_obj is a list of integer index vectors (spdep::nb).
  # id_order maps position â†’ cell_id.
  lens <- lengths(nb_obj)
  from_idx <- rep(seq_along(nb_obj), lens)
  to_idx   <- unlist(nb_obj, use.names = FALSE)

  # Remove 0-entries (spdep uses 0 for "no neighbours")
  valid <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    cell_id     = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)
cat("Edge table rows:", nrow(edge_dt), "\n")
# Expected: ~1,373,394 directed edges


# ---- 2. Vectorised neighbor-feature computation ------------------------------

compute_and_add_all_neighbor_features <- function(cell_dt, edge_dt,
                                                   source_vars) {
  # cell_dt  : data.table with columns id, year, and all source_vars
  # edge_dt  : data.table with columns cell_id, neighbor_id
  # source_vars : character vector of variable names
  #
  # For each var in source_vars, adds three columns:
  #   <var>_neighbor_max, <var>_neighbor_min, <var>_neighbor_mean

  # Ensure data.table

  if (!is.data.table(cell_dt)) cell_dt <- as.data.table(cell_dt)

  # Key the panel data for fast joins
  setkey(cell_dt, id, year)

  for (var in source_vars) {
    cat("  Computing neighbor stats for:", var, "...\n")

    # --- a. Build a slim lookup: neighbor_id + year â†’ value ---
    # Columns: neighbor_id, year, value
    val_dt <- cell_dt[, .(neighbor_id = id, year, value = get(var))]
    setkey(val_dt, neighbor_id, year)

    # --- b. Join edge table to panel to get (cell_id, year) per edge,
    #         then join to val_dt to get the neighbor's value. ---
    #
    # Start from edge_dt, add the year dimension by joining to the
    # focal cell's years.  Because every cell_id appears for every year
    # it has data, we join on cell_id â†’ id to pull in the year column.

    # Slim focal table: which (cell_id, year) pairs exist?
    focal <- cell_dt[, .(cell_id = id, year)]
    setkey(focal, cell_id)

    # Merge: for each edge, replicate across all years of the focal cell
    # edge_dt has (cell_id, neighbor_id).
    # focal  has (cell_id, year).
    # Result: (cell_id, neighbor_id, year)
    setkey(edge_dt, cell_id)
    edge_year <- edge_dt[focal, on = "cell_id", allow.cartesian = TRUE,
                         nomatch = NULL]
    # edge_year now has columns: cell_id, neighbor_id, year

    # Attach the neighbor's value
    edge_year[val_dt, value := i.value,
              on = .(neighbor_id, year)]

    # --- c. Aggregate: max, min, mean per (cell_id, year) ---
    agg <- edge_year[!is.na(value),
                     .(nmax  = max(value),
                       nmin  = min(value),
                       nmean = mean(value)),
                     by = .(cell_id, year)]

    # --- d. Merge back into cell_dt ---
    col_max  <- paste0(var, "_neighbor_max")
    col_min  <- paste0(var, "_neighbor_min")
    col_mean <- paste0(var, "_neighbor_mean")

    # Use a keyed join to set columns in place (avoids full-frame copy)
    setkey(agg, cell_id, year)
    cell_dt[agg, (col_max)  := i.nmax,  on = .(id = cell_id, year)]
    cell_dt[agg, (col_min)  := i.nmin,  on = .(id = cell_id, year)]
    cell_dt[agg, (col_mean) := i.nmean, on = .(id = cell_id, year)]

    # Clean up large intermediates
    rm(val_dt, focal, edge_year, agg)
    gc()
  }

  return(cell_dt)
}


# ---- 3. Run feature preparation ---------------------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Building neighbour features...\n")
cell_data <- as.data.table(cell_data)

cell_data <- compute_and_add_all_neighbor_features(
  cell_dt     = cell_data,
  edge_dt     = edge_dt,
  source_vars = neighbor_source_vars
)
cat("Neighbour features complete.\n")


# ---- 4. Chunked Random Forest prediction -------------------------------------

predict_chunked <- function(model, newdata, chunk_size = 500000L) {
  # Works with both randomForest and ranger models.
  # Preserves the original numerical estimand exactly.

  n <- nrow(newdata)
  chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  cat("Predicting in", length(chunks), "chunks of up to", chunk_size, "rows\n")

  is_ranger <- inherits(model, "ranger")

  preds <- vector("list", length(chunks))
  for (k in seq_along(chunks)) {
    idx <- chunks[[k]]
    chunk_df <- if (is.data.table(newdata)) {
      as.data.frame(newdata[idx, ])
    } else {
      newdata[idx, , drop = FALSE]
    }

    if (is_ranger) {
      preds[[k]] <- predict(model, data = chunk_df)$predictions
    } else {
      # randomForest
      preds[[k]] <- predict(model, newdata = chunk_df)
    }

    if (k %% 5 == 0 || k == length(chunks)) {
      cat("  Chunk", k, "/", length(chunks), "done\n")
    }
    gc()
  }

  unlist(preds, use.names = FALSE)
}


# ---- 5. Prepare predictor matrix & predict -----------------------------------

# Identify the predictor columns the model expects.
if (inherits(rf_model, "ranger")) {
  predictor_names <- rf_model$forest$independent.variable.names
} else {
  # randomForest stores predictor names in xlevels or can be inferred
  # from the forest$xlevels or the training call.
  # Safest: use the column names from the training data minus the response.
  predictor_names <- attr(rf_model$terms, "term.labels")
  if (is.null(predictor_names)) {
    predictor_names <- rownames(rf_model$importance)
  }
}

cat("Model expects", length(predictor_names), "predictors.\n")

# Subset to predictors only (keeps memory lean for prediction)
pred_data <- cell_data[, ..predictor_names]

# Run chunked prediction
cell_data[, predicted_gdp := predict_chunked(rf_model, pred_data,
                                              chunk_size = 500000L)]

cat("Pipeline complete. Predictions stored in cell_data$predicted_gdp\n")
```

---

## 4. Why This Works â€” Performance Summary

| Component | Before (base R) | After (data.table) | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46 M `paste` + hash lookups | Single vectorised edge table build | ~100â€“500Ã— |
| `compute_neighbor_stats` (Ã—5 vars) | ~32 M `lapply` iterations + `do.call(rbind)` | 5 keyed joins + group-by aggregations | ~50â€“200Ã— |
| RF prediction (6.46 M rows) | Single call, ~24 GB peak â†’ swap/crash | 13 chunks Ã— 500 K rows, ~2â€“3 GB peak | Fits in 16 GB; avoids swap |
| Column addition | Copy-on-modify each time | `data.table` set-by-reference | No copies |

**Estimated total wall-clock time: 10â€“30 minutes** on the described 16 GB laptop, down from 86+ hours.

### Key Guarantees

- **Trained RF model is never retrained** â€” only `predict()` is called.
- **Numerical estimand is preserved exactly** â€” the same features are computed (max, min, mean of the same neighbor values), and the same model produces the same predictions. No approximation is introduced.
END_RESPONSE>>>
