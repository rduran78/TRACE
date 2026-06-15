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
run_id: Anthropic_F03_T07_R020
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 20
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_020.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

1. **Row-level `lapply` over 6.46M rows.** Each iteration does character coercion, string pasting (`paste()`), and named-vector lookups (`idx_lookup[neighbor_keys]`). Named-vector lookup in R is O(n) hash probing per call, but doing it 6.46 million times with freshly-pasted keys is extremely slow.
2. **Redundant string operations.** `paste(neighbor_cell_ids, data$year[i], sep = "_")` is called inside a per-row loop, creating millions of small character vectors that pressure the garbage collector.
3. **The lookup returns a list of 6.46M integer vectors.** This object alone can consume several GB of RAM (each list element has R overhead of ~56+ bytes, plus the integer vector).

**`compute_neighbor_stats`** is the second bottleneck:

1. **Per-row `lapply` again over 6.46M elements**, extracting subsets of a numeric vector, removing NAs, and computing max/min/mean. This is called 5 times (once per neighbor source variable), so ~32.3M R-level function calls.
2. **`do.call(rbind, result)` on a 6.46M-element list** creates a large temporary matrix by progressively binding rows â€” extremely memory-intensive.

**Outer loop:** Calls `compute_and_add_neighbor_features` 5 times, each presumably rebuilding the stats matrix and column-binding to `cell_data`. Repeated column-binding to a 6.46M-row data.frame triggers full copies each time.

### B. Random Forest Inference Bottleneck

With ~6.46M rows Ã— ~110 predictors, a single `predict()` call on a `ranger` or `randomForest` object will:

- Attempt to allocate a prediction matrix of ~6.46M Ã— 110 doubles â‰ˆ **5.3 GB** in one shot, likely exceeding available RAM on a 16 GB laptop (the model object, `cell_data`, neighbor lookup, and R overhead already consume many GB).
- If using the `randomForest` package (not `ranger`), prediction is single-threaded and extremely slow at this scale.

### C. Summary of Root Causes

| Rank | Bottleneck | Cause |
|------|-----------|-------|
| 1 | `build_neighbor_lookup` | 6.46M-iteration R loop with string ops and named-vector lookups |
| 2 | `compute_neighbor_stats` | 6.46M-iteration R loop Ã— 5 variables; `do.call(rbind, ...)` on huge list |
| 3 | Column binding in outer loop | Repeated full-copy of 6.46M-row data.frame |
| 4 | RF `predict()` | Single massive allocation; possibly single-threaded package |
| 5 | Object size / GC pressure | Neighbor lookup list of 6.46M elements; multiple large temporaries |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize everything; eliminate per-row R loops; use `data.table` for in-place column addition; chunk the RF prediction.

| Bottleneck | Strategy |
|-----------|----------|
| `build_neighbor_lookup` | Replace with a fully vectorized `data.table` join. Expand the neighbor list into a flat edge table (`cell_id`, `neighbor_id`), join with `(id, year)` to get row indices, and store as a two-column edge matrix â€” no per-row loop at all. |
| `compute_neighbor_stats` | Use the flat edge table + `data.table` grouped aggregation: group by source-row index, compute `max`, `min`, `mean` of the neighbor values in one vectorized pass per variable. |
| Outer loop / column binding | Use `data.table` `:=` (assign by reference) to add columns in place â€” zero copies. |
| RF prediction | Predict in chunks of ~500K rows to cap peak memory; if model is `randomForest`-class, convert to `ranger` for multi-threaded prediction (or use chunked `predict.randomForest`). |
| Memory | Drop the 6.46M-element list entirely; the flat edge table is more compact and GC-friendly. |

**Estimated speedup:** Feature preparation drops from many hours to minutes; RF prediction drops from hours to tens of minutes. Total wall-clock target: **under 1 hour** on the described laptop.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE â€” data.table vectorized neighbor features + chunked RF
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table (by reference if possible) -----
if (!is.data.table(cell_data)) {
  setDT(cell_data) # converts in place â€” no copy
}

# ---- Step 1: Build a flat, vectorized neighbor edge table -------------------
#
# Replaces build_neighbor_lookup entirely.
# rook_neighbors_unique is an nb object: a list of integer index vectors
# id_order is the vector of cell IDs in the same order as the nb object.

build_neighbor_edges <- function(id_order, neighbors) {
  # Expand the nb list into a flat edge list of (source_cell_id, neighbor_cell_id)
  n_neighbors <- vapply(neighbors, length, integer(1))
  source_idx  <- rep(seq_along(neighbors), times = n_neighbors)
  target_idx  <- unlist(neighbors, use.names = FALSE)

  data.table(
    source_id   = id_order[source_idx],
    neighbor_id = id_order[target_idx]
  )
}

edge_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)

# ---- Step 2: Vectorized neighbor-stat computation ---------------------------
#
# For each (source cell, year) we need max, min, mean of each variable
# across that cell's rook neighbors in the same year.
#
# Strategy:
#   1. Give every row in cell_data a row index.
#   2. Join edge_dt with cell_data twice:
#        - first to get the year of the source row
#        - then to get the neighbor's value in that same year
#   3. Group by source row index and aggregate.

# Ensure cell_data has a row-index column
cell_data[, .row_idx := .I]

# Minimal lookup tables (avoid copying the full data.table)
# Source side: row_idx -> (id, year)
source_key <- cell_data[, .(id, year, .row_idx)]
setkey(source_key, id)

compute_and_add_neighbor_features_vec <- function(cell_data, var_name, edge_dt,
                                                   source_key) {
  # Neighbor side: (id, year) -> value
  neighbor_val <- cell_data[, .(id, year, .val = get(var_name))]
  setkey(neighbor_val, id, year)

  # Join edges with source key to get (source .row_idx, neighbor_id, year)
  # edge_dt: source_id, neighbor_id
  work <- merge(edge_dt, source_key, by.x = "source_id", by.y = "id",
                allow.cartesian = TRUE)
  # work now has: source_id, neighbor_id, year, .row_idx

  # Join with neighbor values on (neighbor_id, year)
  setkey(work, neighbor_id, year)
  work <- neighbor_val[work, on = .(id = neighbor_id, year), nomatch = NA]
  # work now has: .val, id (=neighbor_id), year, source_id, .row_idx

  # Aggregate: group by .row_idx (the source row)
  stats <- work[!is.na(.val),
                .(nmax  = max(.val),
                  nmin  = min(.val),
                  nmean = mean(.val)),
                keyby = .row_idx]

  # Column names matching original pipeline
  col_max  <- paste0("n_max_",  var_name)
  col_min  <- paste0("n_min_",  var_name)
  col_mean <- paste0("n_mean_", var_name)

  # Assign by reference â€” no copy of cell_data
  cell_data[stats, (col_max)  := i.nmax,  on = ".row_idx"]
  cell_data[stats, (col_min)  := i.nmin,  on = ".row_idx"]
  cell_data[stats, (col_mean) := i.nmean, on = ".row_idx"]

  invisible(cell_data)
}

# ---- Step 3: Run for all five neighbor source variables ---------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message("Computing neighbor features for: ", var_name)
  compute_and_add_neighbor_features_vec(cell_data, var_name, edge_dt, source_key)
  gc()
}

# Clean up helper column
cell_data[, .row_idx := NULL]

# ---- Step 4: Chunked Random Forest prediction --------------------------------
#
# Works with both `randomForest` and `ranger` model objects.
# Preserves the trained model â€” no retraining.

chunked_rf_predict <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  preds  <- numeric(n)

  is_ranger <- inherits(model, "ranger")

  for (i in seq_along(chunks)) {
    idx <- chunks[[i]]
    chunk_data <- newdata[idx, ]

    if (is_ranger) {
      preds[idx] <- predict(model, data = chunk_data)$predictions
    } else {
      # randomForest package
      preds[idx] <- as.numeric(predict(model, newdata = chunk_data))
    }

    if (i %% 5 == 0) {
      message(sprintf("  Predicted chunk %d / %d  (%d rows so far)",
                       i, length(chunks), max(idx)))
      gc()
    }
  }

  preds
}

# Prepare the prediction matrix (only the predictor columns the model expects)
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else {
  # randomForest stores the variable names used during training
  pred_vars <- rownames(importance(rf_model))
}

# Subset to predictor columns; data.table subsetting is zero-copy for columns
pred_data <- cell_data[, ..pred_vars]

message("Starting chunked RF prediction over ", nrow(pred_data), " rows ...")
cell_data[, predicted_gdp := chunked_rf_predict(rf_model, pred_data,
                                                  chunk_size = 500000L)]
message("Prediction complete.")

# =============================================================================
# END OF OPTIMIZED PIPELINE
# =============================================================================
```

---

## 4. WHY THIS PRESERVES CORRECTNESS

| Requirement | How it is met |
|------------|---------------|
| **Trained RF model unchanged** | The model object (`rf_model`) is only passed to `predict()` â€” never modified or retrained. |
| **Original numerical estimand preserved** | The same `max`, `min`, `mean` aggregations are computed over the identical rook-neighbor sets and the identical variable values. Column names follow the same `n_max_*`, `n_min_*`, `n_mean_*` convention. `predict()` output is bit-identical because the same model and the same feature matrix are used. |
| **No data loss** | Rows with no valid neighbors receive `NA` for neighbor stats (same as original). The `merge`/`join` approach naturally handles this because those source rows simply have no grouped output and the `:=` join leaves them as `NA`. |

---

## 5. EXPECTED PERFORMANCE

| Stage | Before (est.) | After (est.) | Key change |
|-------|--------------|-------------|-----------|
| `build_neighbor_lookup` | ~10â€“20 hrs | **< 30 sec** | Vectorized `data.table` expansion replaces 6.46M-iteration R loop |
| `compute_neighbor_stats` (Ã—5 vars) | ~40â€“60 hrs | **5â€“15 min** | `data.table` keyed joins + grouped aggregation; no per-row `lapply` |
| Column binding | hours of GC | **~0 sec** | `:=` assign by reference |
| RF prediction | ~5â€“10 hrs | **15â€“45 min** | Chunked to cap memory; `ranger` multi-threaded if available |
| **Total** | **86+ hrs** | **~30â€“60 min** | |

All estimates assume the described 16 GB laptop. If the model is a `randomForest`-class object, the single largest remaining improvement would be to convert it once to a `ranger` object for multi-threaded prediction (a one-time conversion, not retraining), but the chunked approach above already works correctly with either package.
END_RESPONSE>>>
