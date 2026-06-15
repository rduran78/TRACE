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
run_id: Anthropic_F03_T10_R024
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 1.0
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 24
raw_output_file: Experiment_Output/Anthropic/Temp_1.0/output_F03_Anthropic_temp_10_rep_024.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, and named-vector lookup â€” all O(n) string operations repeated millions of times. |
| **`paste()`-keyed lookup** | Creating 6.46M string keys (`"id_year"`) and then probing them inside every iteration is extremely slow; R's named-vector lookup is O(n) per probe in the worst case. |
| **Redundant character coercions** | `as.character(data$id[i])` is called per row instead of vectorised once. |
| **Output: list of 6.46M integer vectors** | This structure alone consumes enormous memory (each list element has its own SEXP header â‰ˆ 56+ bytes overhead, times 6.46M â‰ˆ 350 MB just in overhead). |

**`compute_neighbor_stats`** is called 5 times (once per variable), each time doing:

- `lapply` over 6.46M elements, each subsetting `vals[idx]`, removing NAs, then computing `max/min/mean`. This is pure R-level looping â€” no vectorisation, no C-level aggregation.
- `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is notoriously slow (repeated memory allocation and copying).

### B. Random Forest Inference Bottleneck

With ~6.46M rows Ã— ~110 predictors, `predict.randomForest()` (or `predict.ranger()`) must:

- Copy the entire prediction data frame internally.
- Walk every tree for every row.

If the model was trained with `randomForest::randomForest`, the `predict()` method is single-threaded and unoptimised for large-n prediction. A `ranger` model would use all cores, but even then, passing a bloated `data.frame` with wrong column types or extra columns causes unnecessary copying.

**Estimated time breakdown** (of the ~86 h):

| Phase | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~30-40% |
| `compute_neighbor_stats` (Ã—5 vars) | ~25-35% |
| RF `predict()` | ~20-30% |
| Miscellaneous (I/O, GC, copying) | ~5-10% |

---

## 2. Optimization Strategy

### Feature Preparation

1. **Replace the string-keyed lookup with a `data.table` equi-join.** Build a `data.table` of `(id, year, row_idx)`, then join against an expanded neighbor-edge table `(id, neighbor_id)` crossed with years. This converts the entire `build_neighbor_lookup` into a single vectorised merge â€” no R-level loop at all.

2. **Replace per-row `lapply` aggregation with `data.table` grouped aggregation.** After the join, each original row has its neighbor rows linked. We group by the original row index and compute `max`, `min`, `mean` in one vectorised pass per variable, for all 6.46M groups simultaneously.

3. **Process all 5 neighbor variables in one pass** over the joined table instead of 5 separate `lapply` loops.

4. **Never materialise a 6.46M-element list.** Everything stays in columnar `data.table` form.

### Random Forest Inference

5. **If the model is a `randomForest` object, convert it to `ranger` format (or, more practically, use `ranger::predict` via a thin wrapper) or chunk the prediction.** Since the user says "preserve the trained model," we keep the model object but predict in chunks to control memory, and we ensure the input matrix is clean (no extra columns, correct types).

6. **Predict in batches** (~500Kâ€“1M rows) to avoid a single monolithic memory allocation, and to allow garbage collection between chunks.

7. **Strip the prediction input to only the columns the model needs**, as a `matrix` if the model supports it, to eliminate data-frame overhead.

---

## 3. Working R Code

```r
###############################################################################
# 0. Libraries
###############################################################################
library(data.table)
# library(ranger)        # only if the saved model is ranger; otherwise randomForest
# library(randomForest)  # whichever was used for training

###############################################################################
# 1. Load pre-existing objects (assumed already in environment or loaded here)
#    - cell_data        : data.frame / data.table with columns id, year, + predictors
#    - id_order         : integer vector of cell IDs matching the nb object
#    - rook_neighbors_unique : spdep nb object (list of integer index vectors)
#    - rf_model         : the pre-trained Random Forest model
###############################################################################

# Convert cell_data to data.table in-place (no copy if already data.table)
setDT(cell_data)

###############################################################################
# 2. Build edge list from the nb object â€” ONE-TIME, fully vectorised
###############################################################################
build_edge_dt <- function(id_order, nb_obj) {
  # nb_obj[[i]] contains integer indices into id_order for the neighbors of

  # id_order[i].  0L means no neighbors in spdep convention.
  n <- length(nb_obj)
  from_idx <- rep.int(seq_len(n), lengths(nb_obj))
  to_idx   <- unlist(nb_obj, use.names = FALSE)


  # Remove spdep's 0-coded "no neighbour" entries

  valid <- to_idx != 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_dt(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed edges)

cat("Edge table rows:", nrow(edge_dt), "\n")

###############################################################################
# 3. Compute ALL neighbor features in ONE pass via data.table join + group-by
###############################################################################
compute_all_neighbor_features <- function(cell_dt, edge_dt,
                                          source_vars) {
  # cell_dt must have columns: id, year, and every var in source_vars
  # Returns cell_dt (modified in-place) with new columns appended.

  # ---- 3a. Add a row-index column to cell_dt so we can map back ---------
  cell_dt[, .row_idx := .I]

  # ---- 3b. Build a keyed version for the join ---------------------------
  # We need: for each (id, year) row, find all neighbor rows that share

  # the same year and have neighbor_id == neighbor's id.
  #
  # Join path:
  #   cell_dt  -->  edge_dt  on id  -->  cell_dt again  on (neighbor_id, year)
  #
  # To avoid a massive intermediate table (6.46M Ã— avg_neighbors), we do a

  # single chained merge.

  # Slim table: only the columns we need for the neighbor lookup
  keep_cols <- c("id", "year", ".row_idx", source_vars)
  slim <- cell_dt[, ..keep_cols]

  # Key the slim table for the second join leg

  setkey(slim, id, year)

  # ---- 3c. Expand edges Ã— years ----------------------------------------
  # Instead of crossing 1.37M edges Ã— 28 years (38M rows), we let data.table

  # handle it via a merge:
  #
  #   Step 1: join slim to edge_dt on id  â†’  gives (row_idx, year, neighbor_id)
  #   Step 2: join that to slim again on (neighbor_id = id, year)

  # Step 1
  setkey(edge_dt, id)
  expanded <- edge_dt[slim, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded now has columns: id, neighbor_id, year, .row_idx, <source_vars>
  # The source_var values here are the FOCAL cell's values (not needed for

  # neighbor stats, but .row_idx is essential).
  # We only need .row_idx and year from the focal cell, plus neighbor_id.

  focal_info <- expanded[, .(.row_idx, year, neighbor_id)]
  rm(expanded); gc()

  # Step 2: bring in neighbor values
  setnames(focal_info, "neighbor_id", "id")
  setkey(focal_info, id, year)
  joined <- slim[focal_info, on = c("id", "year"), nomatch = NA]
  # 'joined' now has the neighbor's variable values, plus .row_idx pointing

  # back to the focal cell.  Rename to avoid confusion.
  # Columns from slim (the neighbor): id, year, .row_idx (neighbor's), <source_vars>
  # Columns from focal_info (i.): i..row_idx  â† this is the FOCAL row index
  setnames(joined, "i..row_idx", "focal_row_idx")
  # Drop the neighbor's own .row_idx â€” we don't need it

  joined[, .row_idx := NULL]

  rm(focal_info); gc()

  # ---- 3d. Grouped aggregation ------------------------------------------
  # For each focal_row_idx, compute max/min/mean of each source variable
  # across its neighbors.

  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))

  # Build the aggregation call programmatically
  # Using a simpler, robust approach:
  agg_list <- setNames(agg_exprs, agg_names)

  # data.table: aggregate
  agg <- joined[, {
    out <- vector("list", length(source_vars) * 3L)
    k <- 0L
    for (v in source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      n <- length(vals)
      k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else max(vals)
      k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else min(vals)
      k <- k + 1L; out[[k]] <- if (n == 0L) NA_real_ else mean(vals)
    }
    setattr(out, "names", agg_names)
    out
  }, by = focal_row_idx]

  rm(joined); gc()

  # ---- 3e. Merge aggregated stats back into cell_dt ---------------------
  setkey(agg, focal_row_idx)
  for (nm in agg_names) {
    set(cell_dt, i = agg$focal_row_idx, j = nm, value = agg[[nm]])
  }

  # Rows with no neighbors at all won't appear in agg â†’ they stay NA, which

  # matches the original behaviour.

  # Clean up the temporary column

  cell_dt[, .row_idx := NULL]

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features â€¦\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})
# Expected: minutes, not hours.

###############################################################################
# 4. Random Forest Prediction â€” batched, memory-efficient
###############################################################################
predict_rf_batched <- function(model, newdata, batch_size = 500000L) {
  # Determine the columns the model expects
  if (inherits(model, "ranger")) {
    # ranger stores variable names used in training
    model_vars <- model$forest$independent.variable.names
  } else if (inherits(model, "randomForest")) {
    # randomForest stores them as rownames of importance or via the formula
    model_vars <- rownames(model$importance)
    if (is.null(model_vars)) {
      # fallback: use the names stored in the forest's xlevels or from training
      model_vars <- names(model$forest$xlevels)
    }
  } else {
    stop("Unsupported model class: ", class(model)[1])
  }

  # Subset to only required columns â€” avoids copying unneeded data
  if (is.data.table(newdata)) {
    pred_input <- newdata[, ..model_vars]
  } else {
    pred_input <- newdata[, model_vars, drop = FALSE]
  }

  n <- nrow(pred_input)
  preds <- numeric(n)

  starts <- seq(1L, n, by = batch_size)

  cat(sprintf("Predicting %s rows in %d batches â€¦\n", format(n, big.mark = ","),
              length(starts)))

  for (i in seq_along(starts)) {
    s <- starts[i]
    e <- min(s + batch_size - 1L, n)
    batch <- pred_input[s:e, , drop = FALSE]

    if (inherits(model, "ranger")) {
      preds[s:e] <- predict(model, data = batch, num.threads = parallel::detectCores())$predictions
    } else {
      preds[s:e] <- predict(model, newdata = batch)
    }

    if (i %% 5 == 0 || i == length(starts)) {
      cat(sprintf("  batch %d/%d done (rows %sâ€“%s)\n",
                  i, length(starts),
                  format(s, big.mark = ","), format(e, big.mark = ",")))
    }
    gc()
  }

  preds
}

cat("Running Random Forest predictions â€¦\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_batched(rf_model, cell_data)]
})

###############################################################################
# 5. (Optional) If the original model is randomForest and is too slow,
#    convert it to ranger for multi-threaded predict. This preserves the
#    original numerical estimand because the trees are identical â€” we are
#    only changing the predict() dispatch, not retraining.
#
#    NOTE: A direct object conversion is non-trivial. A pragmatic alternative
#    is to keep the randomForest model and just batch as above. If speed is
#    still insufficient, one can serialise each tree's predictions in parallel
#    using parallel::mclapply (Unix) or future.apply.
###############################################################################

predict_rf_parallel <- function(model, newdata, n_cores = parallel::detectCores(),
                                batch_size = 500000L) {
  # Only for randomForest objects â€” parallelise across tree subsets
  if (!inherits(model, "randomForest"))
    stop("This helper is for randomForest objects only.")

  ntree <- model$ntree
  model_vars <- rownames(model$importance)
  if (is.data.table(newdata)) {
    pred_input <- as.data.frame(newdata[, ..model_vars])
  } else {
    pred_input <- newdata[, model_vars, drop = FALSE]
  }

  n <- nrow(pred_input)
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)

  cat(sprintf("Parallel predict: %d cores, %d batches\n", n_cores, length(starts)))

  for (b in seq_along(starts)) {
    s <- starts[b]
    e <- min(s + batch_size - 1L, n)
    batch <- pred_input[s:e, , drop = FALSE]

    # Split trees across cores
    tree_groups <- split(seq_len(ntree),
                         cut(seq_len(ntree), n_cores, labels = FALSE))

    # Predict subsets of trees in parallel, then average
    partial <- parallel::mclapply(tree_groups, function(trees) {
      predict(model, newdata = batch, predict.all = TRUE)$individual[, trees, drop = FALSE]
    }, mc.cores = n_cores)

    # Combine and average
    all_tree_preds <- do.call(cbind, partial)   # rows Ã— ntree
    preds[s:e] <- rowMeans(all_tree_preds)

    if (b %% 5 == 0 || b == length(starts)) {
      cat(sprintf("  batch %d/%d done\n", b, length(starts)))
    }
    rm(all_tree_preds, partial, batch); gc()
  }

  preds
}

# Usage (Unix/macOS only due to mclapply):
# cell_data[, predicted_gdp := predict_rf_parallel(rf_model, cell_data)]

###############################################################################
# 6. Summary of expected improvements
###############################################################################
# | Component                  | Before          | After (est.)      | Speedup |
# |----------------------------|-----------------|-------------------|---------|
# | build_neighbor_lookup      | ~25-35 h        | <5 min            | ~300Ã—   |
# | compute_neighbor_stats Ã—5  | ~20-30 h        | <5 min (all vars) | ~300Ã—   |
# | RF predict (single-thread) | ~15-25 h        | ~2-4 h (batched)  | ~5-8Ã—   |
# | RF predict (parallel)      |                 | ~0.5-1 h          | ~20-30Ã— |
# | Total                      | ~86 h           | ~1-4 h            | ~20-80Ã— |
###############################################################################

cat("Done.\n")
```

---

## Key Points

| Original Problem | Root Cause | Fix |
|---|---|---|
| `build_neighbor_lookup`: 6.46M R-level iterations with `paste`/named-vector lookups | String-keyed lookup in a loop; O(nÂ²) effective complexity | Single `data.table` equi-join on integer keys (`id`, `year`) â€” fully vectorised, C-level |
| `compute_neighbor_stats`: 5 Ã— 6.46M `lapply` iterations + `do.call(rbind, â€¦)` | R-level loop, per-element allocation, slow `rbind` | `data.table` grouped `[, .(max,min,mean), by=focal_row_idx]` â€” one pass for all 5 variables |
| RF `predict()`: single 6.46M-row call | Monolithic memory allocation; single-threaded for `randomForest` | Batched prediction (500K chunks); optional parallel tree-splitting; input trimmed to model columns only |
| Memory pressure (16 GB laptop) | Full data copied into predict; 6.46M-element lists | Columnar `data.table` throughout; batched predict with `gc()` between chunks; no intermediate lists |

The trained Random Forest model is never retrained or altered. The numerical predictions (the estimand) are preserved identically â€” we only change how efficiently the data is prepared and fed to `predict()`.
END_RESPONSE>>>
