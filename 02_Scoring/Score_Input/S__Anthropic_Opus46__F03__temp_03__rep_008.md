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
run_id: Anthropic_F03_T03_R008
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 8
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_008.md

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
| **Named-vector lookup via `idx_lookup[neighbor_keys]`** | This is O(n) string hashing per call Ã— 6.46M calls. The `idx_lookup` vector itself has 6.46M named elements; repeated name-matching against it is extremely slow. |
| **Redundant string pasting** | `paste(id, year, sep="_")` is computed once to *build* the lookup, then again *inside every row* to query it. |
| **Output is a list of 6.46M integer vectors** | ~6.46M list elements, each a small integer vector â€” heavy memory allocation and GC pressure. |

**`compute_neighbor_stats`** compounds the problem:

| Problem | Detail |
|---|---|
| **Called 5 times, each time `lapply` over 6.46M rows** | 5 Ã— 6.46M = 32.3M interpreted R function calls. |
| **Per-row subsetting, NA removal, summary stats** | `vals[idx]`, `!is.na(...)`, `max/min/mean` â€” all fine individually but catastrophic at 32.3M iterations. |
| **`do.call(rbind, result)` on 6.46M single-row matrices** | This alone can take minutes and doubles peak memory. |

### B. Random Forest Inference Bottleneck

| Problem | Detail |
|---|---|
| **Single monolithic `predict()` call on 6.46M Ã— 110 matrix** | `ranger`/`randomForest` must traverse every tree for every row. With 500+ trees this is CPU-bound but also memory-bound if the model + data exceed RAM. |
| **Data frame copying** | If `cell_data` is a `data.frame`, adding columns with `cell_data$new_col <- ...` triggers a full copy each time (COW semantics). With ~110 columns Ã— 6.46M rows â‰ˆ 5.4 GB, each copy is devastating. |
| **Model object size** | A `randomForest` object on 6.46M rows can be several GB. Loading from disk with `readRDS` is I/O-bound. |

### Estimated time breakdown (86+ hours)

| Phase | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~30â€“40% |
| `compute_neighbor_stats` (Ã—5 vars) | ~30â€“40% |
| RF `predict()` | ~15â€“25% |
| Data copying / GC | ~10â€“15% |

---

## 2. Optimization Strategy

### Principle: Eliminate interpreted-R loops; vectorize everything with `data.table` joins and matrix operations.

| Strategy | Technique | Expected speedup |
|---|---|---|
| **Replace `build_neighbor_lookup`** | Build a `data.table` edge-list `(id, year, neighbor_id)` and do a keyed join to get neighbor row indices. No `lapply`, no `paste` lookup. | 50â€“200Ã— |
| **Replace `compute_neighbor_stats`** | Use the edge-list `data.table` to join variable values, then `[, .(max, min, mean), by=.(id, year)]` â€” fully vectorized C-level grouping. | 50â€“100Ã— |
| **Compute all 5 variables in one pass** | Join all 5 source columns at once, compute 15 stats in a single grouped aggregation. | 5Ã— fewer passes |
| **Use `data.table` throughout** | Avoid `data.frame` COW copies. Add columns by reference with `:=`. | 2â€“5Ã— on memory/copy |
| **Chunk RF prediction** | Call `predict()` in chunks of ~500K rows to control peak memory. | Avoids OOM; marginal speed gain |
| **Use `ranger` not `randomForest`** | If the model is `randomForest`, convert or re-save as `ranger` (much faster predict). If already `ranger`, use `num.threads`. | 2â€“10Ã— on predict |

**Target runtime: 5â€“20 minutes** (down from 86+ hours).

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” cell-level GDP prediction
# Preserves trained RF model and original numerical estimand.
# =============================================================================

library(data.table)

# ---- 0. Helper: convert spdep nb object to a data.table edge list ----------
nb_to_edge_dt <- function(nb_obj, id_order) {
 # nb_obj: list of integer vectors (indices into id_order)
 # id_order: vector of cell IDs in the order matching nb_obj
  lens <- lengths(nb_obj)
  from_idx <- rep(seq_along(nb_obj), lens)
  to_idx   <- unlist(nb_obj, use.names = FALSE)
  # Remove 0-entries that spdep uses for "no neighbours"
  valid <- to_idx > 0L
  data.table(
    id       = id_order[from_idx[valid]],
    nb_id    = id_order[to_idx[valid]]
  )
}

# ---- 1. Build edge list (once) ---------------------------------------------
edge_dt <- nb_to_edge_dt(rook_neighbors_unique, id_order)
# edge_dt has columns: id, nb_id   (~1.37M rows)

# ---- 2. Convert cell_data to data.table (by reference if possible) ---------
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure key columns are proper types for joining
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]
edge_dt[,   id   := as.integer(id)]
edge_dt[,   nb_id := as.integer(nb_id)]

# ---- 3. Vectorised neighbor-feature computation (ALL vars, ONE pass) -------
compute_all_neighbor_features <- function(dt, edge_dt, source_vars) {
  # Step 1: Cross edge_dt with years present in dt.
  #   For each (id, year) we need (nb_id, year) rows from dt.
  #   Strategy: join edge_dt to dt on id to get years, then join
  #   back to dt on (nb_id, year) to get neighbor values.

  # Minimal subset for the join: id, year, and source_vars
  cols_needed <- c("id", "year", source_vars)
  sub <- dt[, ..cols_needed]

  # Step 2: For every row in sub, attach its neighbor IDs via edge_dt
  #   Result: one row per (id, year, nb_id) triple
  setkey(edge_dt, id)
  # Merge: for each (id) in sub, get all nb_id from edge_dt
  # Use edge_dt[sub, on="id", allow.cartesian=TRUE] â€” gives (id, year, nb_id)
  expanded <- edge_dt[sub, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded columns: id, nb_id, year, <source_vars>  (source_vars are from sub = focal cell values, not needed yet)
  # We only need id, year, nb_id from this join; drop source_var columns to save memory
  expanded[, (source_vars) := NULL]

  # Step 3: Join neighbor values from sub on (nb_id = id, year)
  setnames(sub, "id", "nb_id")
  setkey(sub, nb_id, year)
  setkey(expanded, nb_id, year)
  expanded <- sub[expanded, on = c("nb_id", "year"), nomatch = NA]
  # Now expanded has: nb_id, year, <source_vars (neighbor values)>, id

  # Step 4: Grouped aggregation â€” max, min, mean per (id, year) for each var
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("nb_", v, c("_max", "_min", "_mean"))
  }))

  # Build the j-expression programmatically
  j_list <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

  agg <- expanded[, eval(j_list), by = .(id, year)]

  # Replace Inf/-Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(agg, which(is.infinite(agg[[col]])), col, NA_real_)
  }

  return(agg)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorised)...\n")
system.time({
  nb_features <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# ---- 4. Merge neighbor features back into cell_data by reference -----------
# Remove old neighbor columns if they exist (idempotency)
old_nb_cols <- grep("^nb_", names(cell_data), value = TRUE)
if (length(old_nb_cols)) cell_data[, (old_nb_cols) := NULL]

setkey(cell_data, id, year)
setkey(nb_features, id, year)
cell_data <- nb_features[cell_data, on = c("id", "year")]
# This is a right join: all rows of cell_data preserved.

cat("Neighbor features merged. Columns:", ncol(cell_data), "\n")

# ---- 5. Random Forest prediction (chunked, memory-safe) --------------------
predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  # Detect model class to use optimal predict path
  is_ranger <- inherits(model, "ranger")

  for (i in seq_along(starts)) {
    s <- starts[i]
    e <- min(s + chunk_size - 1L, n)
    chunk <- newdata[s:e, ]

    if (is_ranger) {
      preds[s:e] <- predict(model, data = chunk, num.threads = parallel::detectCores())$predictions
    } else {
      # randomForest package
      preds[s:e] <- predict(model, newdata = chunk)
    }

    if (i %% 5 == 0 || i == length(starts)) {
      cat(sprintf("  Predicted %d / %d rows (%.1f%%)\n", e, n, 100 * e / n))
    }
  }
  preds
}

# Load model (once)
cat("Loading trained RF model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")  # <-- adjust path

# Identify predictor columns (exclude id, year, response)
response_var <- "gdp"  # <-- adjust if different
exclude_cols <- c("id", "year", response_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# If model is ranger, it stores feature names; use those to guarantee column order
if (inherits(rf_model, "ranger") && !is.null(rf_model$forest$independent.variable.names)) {
  predictor_cols <- rf_model$forest$independent.variable.names
}

# Build prediction matrix (data.table subsetting avoids copy)
pred_data <- cell_data[, ..predictor_cols]

cat("Running RF prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(rf_model, pred_data, chunk_size = 500000L)]
})

cat("Done. Predictions stored in cell_data$predicted_gdp\n")
```

---

## 4. Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model untouched** | We only call `predict()`; no retraining, no coefficient changes. |
| **Numerical estimand identical** | The neighbor stats (max, min, mean) are computed with the same arithmetic (`max`, `min`, `mean` with `na.rm=TRUE`). `data.table` grouping uses the same R functions â€” results are bit-identical. |
| **Row alignment** | The keyed join on `(id, year)` ensures every original row gets exactly its correct neighbor features. Rows with no neighbors get `NA` â€” same as the original code. |

---

## 5. Expected Performance

| Phase | Original (est.) | Optimised (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~25â€“35 hours | **Eliminated** (edge DT built in <2 sec) |
| `compute_neighbor_stats` Ã—5 | ~25â€“35 hours | **~1â€“3 min** (single vectorised pass) |
| Merge features | ~5â€“10 hours (COW copies) | **~10 sec** (`:=` by reference) |
| RF `predict()` | ~10â€“15 hours | **~5â€“20 min** (chunked, multi-threaded if `ranger`) |
| **Total** | **86+ hours** | **~10â€“30 minutes** |

The key insight: replacing 6.46 million interpreted-R loop iterations with `data.table` keyed joins and grouped aggregations moves all the heavy lifting into C, yielding roughly a **200â€“500Ã— speedup** on the feature-preparation phase.
END_RESPONSE>>>
