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
run_id: Anthropic_F03_T07_R010
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 10
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_010.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering â€” all in interpreted R. |
| **Named-vector lookup via `idx_lookup[neighbor_keys]`** | R's named-vector subsetting is O(n) hashing per call; doing it 6.46M times is catastrophic. |
| **Redundant string construction** | `paste(id, year, sep="_")` is built once for the lookup table but then rebuilt inside every iteration for neighbor keys. |
| **Output is a list of 6.46M integer vectors** | ~6.46M list elements each holding ~4 neighbor indices. Memory allocation and GC pressure are enormous. |

**`compute_neighbor_stats`** amplifies the problem:

- Called 5 times (once per source variable), each time iterating over the same 6.46M-element list.
- Each iteration subsets a numeric vector, removes NAs, and computes `max/min/mean` â€” all in interpreted R.
- `do.call(rbind, result)` on a 6.46M-element list of length-3 vectors is itself slow and memory-hungry (builds a temporary list then copies into a matrix).

**Net effect on feature prep:** ~6.46M Ã— (string ops + lookup) + 5 Ã— 6.46M Ã— (subset + aggregate) â‰ˆ tens of billions of interpreted R operations â†’ the estimated 86+ hours.

### B. Random Forest Inference Bottleneck

With ~6.46M rows and ~110 predictors, a single `predict(rf_model, newdata)` call will:

- Internally copy `newdata` into its required format (often a full matrix copy: 6.46M Ã— 110 Ã— 8 bytes â‰ˆ 5.3 GB).
- Walk every tree for every row â€” CPU-bound but unavoidable.
- On 16 GB RAM, this leaves little headroom; GC thrashing and swap are likely.

Calling `predict()` inside any row-level or chunk-level loop (if that's happening) would multiply this cost.

### C. Object Copying / Memory

- R's copy-on-modify semantics mean that `cell_data <- compute_and_add_neighbor_features(cell_data, ...)` likely copies the entire ~5.7 GB data.frame on every assignment (5 times).
- Intermediate `do.call(rbind, ...)` creates multi-GB temporaries.

---

## 2. Optimization Strategy

| Layer | Strategy | Expected Speedup |
|---|---|---|
| **Data structure** | Convert `cell_data` to a `data.table`; use in-place `:=` assignment to eliminate copies. | ~5Ã— less memory, eliminates 5 full-frame copies |
| **Neighbor lookup** | Replace the per-row `lapply` + string lookup with a fully vectorized `data.table` equi-join. Build a flat edge table `(row_i, row_j)` once via merge, not per-row string matching. | ~100â€“500Ã— faster |
| **Neighbor stats** | Replace per-row `lapply` with a single grouped `data.table` aggregation over the flat edge table: `edges[, .(max, min, mean), by = row_i]`. | ~50â€“200Ã— faster |
| **Repeat for 5 vars** | Compute all 5 variables' neighbor stats in one pass over the edge table (or 5 fast grouped aggregations reusing the same edge table). | ~5Ã— fewer passes |
| **RF prediction** | Predict in chunks (~500K rows) to keep peak memory under control; avoid ever copying the full frame into `predict()`. Convert predictor columns to a matrix once. | Keeps within 16 GB; avoids swap |
| **Overall** | Target: feature prep in minutes, prediction in 10â€“30 min. Total < 1 hour. | ~100Ã— end-to-end |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Requirements: data.table (>= 1.14), ranger or randomForest (whichever was
#               used to train the saved model)
# =============================================================================

library(data.table)

# ---- 0. Load pre-existing objects -------------------------------------------
# cell_data          : data.frame / data.table with columns id, year, ntl, ec,
#                      pop_density, def, usd_est_n2, ... (6.46M rows)
# rf_model           : the already-trained Random Forest model (DO NOT retrain)
# id_order           : vector of cell IDs matching the nb object
# rook_neighbors_unique : spdep::nb list (length = # cells = 344,208)

# Convert to data.table in place (no copy if already a data.table)
setDT(cell_data)

# ---- 1. Build flat neighbor edge table (vectorised) -------------------------
#
# Instead of per-row string matching, we:
#   (a) expand the nb list into a two-column integer table of (cell_id, neighbor_cell_id),
#   (b) merge with cell_data's (id, year, row_index) to get (row_i, row_j) pairs
#       where row_i and row_j share the same year.
#
# This runs in seconds, not hours.

build_neighbor_edges <- function(cell_data, id_order, nb_list) {
  # --- a. Expand nb list into cell-ID pairs ----------------------------------
  n_cells <- length(nb_list)
  # Pre-allocate: total number of directed edges
  n_edges <- sum(lengths(nb_list))       # ~1.37M

  from_idx <- rep.int(seq_len(n_cells), lengths(nb_list))
  to_idx   <- unlist(nb_list, use.names = FALSE)

  # Map positional indices to actual cell IDs
  edges_cell <- data.table(
    from_id = id_order[from_idx],
    to_id   = id_order[to_idx]
  )
  rm(from_idx, to_idx)

  # --- b. Attach row numbers from cell_data ----------------------------------
  # Add a row-number column (needed for fast subsetting later)
  cell_data[, .row_i := .I]

  # Slim lookup: id -> year -> row number
  id_year_lookup <- cell_data[, .(id, year, .row_i)]

  # Merge: for every (from_id, to_id) edge, find all years where BOTH cells

  # have data.  Two keyed joins are much faster than per-row paste + lookup.
  setkey(id_year_lookup, id)

  # First join: attach year & row_i for the focal cell
  edges_full <- edges_cell[id_year_lookup,
    on = .(from_id = id),
    .(from_id, to_id, year = i.year, row_i = i..row_i),
    nomatch = NULL,
    allow.cartesian = TRUE
  ]

  # Second join: attach row_j for the neighbor cell in the same year
  setkey(id_year_lookup, id, year)
  setkey(edges_full, to_id, year)

  edges_full <- edges_full[id_year_lookup,
    on = .(to_id = id, year),
    .(row_i, row_j = i..row_i, year),
    nomatch = NULL
  ]

  # Clean up helper column
  cell_data[, .row_i := NULL]

  return(edges_full[, .(row_i, row_j)])
}

cat("Building neighbour edge table â€¦\n")
system.time({
  edge_dt <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
})
# edge_dt has columns: row_i (focal row), row_j (neighbour row)
# Rows â‰ˆ 1.37M edges Ã— 28 years â‰ˆ 38.5M (manageable)

cat(sprintf("Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))


# ---- 2. Compute neighbour statistics (vectorised) ---------------------------
#
# For each source variable we compute max, min, mean of neighbour values,
# grouped by the focal row.  One data.table grouped aggregation per variable.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbour features â€¦\n")
system.time({
  for (var_name in neighbor_source_vars) {

    # Pull the variable's values for all neighbour rows in one vectorised op
    edge_dt[, val := cell_data[[var_name]][row_j]]

    # Grouped aggregation â€” data.table does this in C, extremely fast
    agg <- edge_dt[!is.na(val),
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      keyby = row_i
    ]

    # Prepare NA-filled columns, then fill in aggregated values
    max_col  <- paste0("neighbor_max_",  var_name)
    min_col  <- paste0("neighbor_min_",  var_name)
    mean_col <- paste0("neighbor_mean_", var_name)

    # In-place assignment via := (no copy of cell_data)
    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    set(cell_data, i = agg$row_i, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$row_i, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$row_i, j = mean_col, value = agg$nb_mean)

    rm(agg)
  }
  edge_dt[, val := NULL]
})


# ---- 3. Chunked Random Forest prediction ------------------------------------
#
# Predicting 6.46M rows Ã— 110 features in one shot can exceed 16 GB.
# We split into chunks, predict each, and concatenate.  The trained model
# is loaded once and never modified.

# Identify the predictor columns the model expects.
# Works for both ranger and randomForest objects:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all needed columns exist
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

cat("Starting chunked prediction â€¦\n")
n_rows     <- nrow(cell_data)
chunk_size <- 500000L                    # ~440 MB per chunk matrix (110 cols)
n_chunks   <- ceiling(n_rows / chunk_size)

# Pre-allocate result vector
predictions <- numeric(n_rows)

system.time({
  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * chunk_size + 1L
    i_end   <- min(ch * chunk_size, n_rows)
    idx     <- i_start:i_end

    # Build a plain matrix for the chunk â€” avoids data.frame overhead inside
    # predict().  as.matrix on a data.table subset is efficient.
    chunk_mat <- as.matrix(cell_data[idx, ..pred_vars])

    if (inherits(rf_model, "ranger")) {
      preds <- predict(rf_model, data = chunk_mat)$predictions
    } else {
      preds <- predict(rf_model, newdata = chunk_mat)
    }

    predictions[i_start:i_end] <- preds
    rm(chunk_mat, preds)

    if (ch %% 2 == 0 || ch == n_chunks) {
      cat(sprintf("  chunk %d / %d done (rows %s â€“ %s)\n",
                  ch, n_chunks,
                  format(i_start, big.mark = ","),
                  format(i_end,   big.mark = ",")))
    }
  }
})

# Attach predictions to the data (in place)
cell_data[, predicted_gdp := predictions]
rm(predictions)

cat("Pipeline complete.\n")
```

---

## 4. Why This Preserves Correctness

| Requirement | How it is met |
|---|---|
| **Trained RF model untouched** | `rf_model` is only passed to `predict()`; no retraining or modification. |
| **Original numerical estimand preserved** | The neighbor statistics (max, min, mean) are computed with identical semantics (NA removal, same aggregation functions). The predictor matrix fed to `predict()` contains exactly the same columns in the same units. |
| **Same neighbor definition** | The edge table is expanded directly from `rook_neighbors_unique` and `id_order`, reproducing the same directed neighbor pairs. |

## 5. Expected Performance

| Stage | Original (est.) | Optimised (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~40â€“60 hours | **< 2 min** (vectorised join) |
| `compute_neighbor_stats` Ã— 5 | ~20â€“30 hours | **< 3 min** (grouped agg) |
| RF prediction (6.46M rows) | Depends on tree count | **10â€“30 min** (chunked, matrix input) |
| **Total** | **86+ hours** | **~15â€“35 min** |

Peak RAM stays under 16 GB because the edge table (~38.5M Ã— 2 int cols â‰ˆ 0.6 GB), the data.table (~5.7 GB), and one prediction chunk (~0.44 GB) coexist comfortably.
END_RESPONSE>>>
