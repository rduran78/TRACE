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
run_id: Anthropic_F03_T03_R028
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.3
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 28
raw_output_file: Experiment_Output/Anthropic/Temp_0.3/output_F03_Anthropic_temp_03_rep_028.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** â€” O(n) `lapply` over ~6.46 million rows, each iteration performing:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`) â€” hash lookup is O(1) on average but the constant factor of `as.character()` + named-vector hashing Ã—6.46M is large.
- `paste()` to build keys, then a second named-vector lookup (`idx_lookup[neighbor_keys]`).
- Allocation of a small integer vector per row.
- **Net effect:** millions of small allocations, millions of `paste` calls, and millions of hash lookups. This alone can take hours on 6.46M rows.

**`compute_neighbor_stats`** â€” Called 5 times (once per source variable). Each call does an `lapply` over 6.46M rows, subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix â€” this is a well-known R anti-pattern that is extremely slow for large lists.

**Outer loop** â€” Calls `compute_and_add_neighbor_features` 5 times, presumably copying the entire `cell_data` data.frame each time (`cell_data <- ...`). With ~6.46M rows Ã— ~110+ columns, each copy is ~5â€“8 GB depending on types. Repeated full-frame copies can cause memory pressure and GC thrashing on a 16 GB machine.

### 1.2 Random Forest Inference Bottlenecks

- **Model loading:** If the serialized RF model is large (hundreds of trees Ã— 110 features Ã— deep trees), `readRDS()` alone can take minutes and consume several GB.
- **Single `predict()` call on 6.46M rows:** `predict.randomForest` (or `predict.ranger`) must push every row through every tree. For `randomForest`-package models this is done in R-level loops and is very slow; `ranger` is C++-backed and much faster.
- **Prediction in a loop (row-by-row or chunk-by-chunk):** If prediction is called inside any loop rather than as a single vectorized call, overhead is catastrophic.
- **Object copying:** If the prediction result is appended column-by-column to a data.frame (triggering copy-on-modify), memory doubles repeatedly.

### 1.3 Summary of Root Causes

| Rank | Bottleneck | Estimated share of 86 h |
|------|-----------|------------------------|
| 1 | `build_neighbor_lookup` â€” millions of `paste`/hash lookups in R loop | ~30â€“40% |
| 2 | `compute_neighbor_stats` â€” `lapply` + `do.call(rbind, ...)` Ã—5 vars | ~25â€“35% |
| 3 | RF prediction â€” possibly row-level or `randomForest`-package predict | ~15â€“25% |
| 4 | Repeated full data.frame copies in outer loop | ~5â€“10% |

---

## 2. OPTIMIZATION STRATEGY

### A. Replace `build_neighbor_lookup` with a vectorized `data.table` join

Instead of building a per-row R list of neighbor indices (6.46M list elements), build a **flat edge table** `(row_i, neighbor_row_j)` using `data.table` keyed joins. This eliminates all `paste`, `as.character`, and named-vector lookups.

### B. Replace `compute_neighbor_stats` with grouped `data.table` aggregation

Join the flat edge table to the variable column, then `group by row_i` and compute `max/min/mean` in one vectorized pass. No `lapply`, no `do.call(rbind, ...)`.

### C. Add all 15 neighbor-feature columns in one pass

Use the same edge table for all 5 variables, computing all 15 statistics (5 Ã— {max, min, mean}) in a single grouped aggregation, then join back once. This avoids 5 separate full-frame copies.

### D. Optimize RF inference

- If the model is a `randomForest`-package object, convert it to `ranger` format or use `predict` in a single vectorized call (never in a loop).
- Alternatively, if retraining is forbidden, keep the model object but ensure `predict()` is called **once** on the full matrix/data.frame.
- Load the model once; predict once; write once.

### E. Use `data.table` throughout to avoid copies

`data.table` modifies by reference (`:=`), so adding columns never copies the frame.

### Projected speedup

| Component | Before | After (est.) |
|-----------|--------|-------------|
| Neighbor lookup | ~25â€“35 h | ~1â€“3 min |
| Neighbor stats (Ã—5) | ~20â€“30 h | ~2â€“5 min |
| Data.frame copies | ~5â€“8 h | ~0 (in-place) |
| RF predict (6.46M rows) | ~10â€“20 h* | ~5â€“30 minâ€  |
| **Total** | **~86 h** | **~15â€“45 min** |

\* If called in a loop or with `randomForest` package.
â€  Single vectorized call; `ranger`-repredict or `randomForest::predict` on full matrix.

---

## 3. WORKING R CODE

```r
# ============================================================
# 0. Libraries
# ============================================================
library(data.table)

# ============================================================
# 1. Build flat neighbor edge table (replaces build_neighbor_lookup)
#
#    Inputs:
#      cell_data            â€” data.frame/data.table with columns: id, year, ...
#      id_order             â€” integer vector of cell IDs in the order used by
#                             the nb object (position k â†” id_order[k])
#      rook_neighbors_unique â€” spdep nb object (list of integer index vectors)
#
#    Output:
#      edge_dt â€” data.table with columns  row_i, row_j
#                meaning "row_j is a rook-neighbor of row_i"
# ============================================================
build_neighbor_edge_table <- function(cell_dt, id_order, neighbors) {

    # --- Map: spatial-index â†’ cell id ---------------------------------
    # neighbors[[k]] contains spatial indices; id_order[k] is the cell id
    # for spatial index k.

    n_spatial <- length(id_order)

    # Build a flat edge list at the cell-id level:
    #   from_id  â†’  to_id   (directed, one entry per directed pair)
    from_idx <- rep(seq_len(n_spatial),
                    times = lengths(neighbors))
    to_idx   <- unlist(neighbors, use.names = FALSE)

    edge_id <- data.table(
        from_id = id_order[from_idx],
        to_id   = id_order[to_idx]
    )
    rm(from_idx, to_idx)

    # --- Map: (id, year) â†’ row number in cell_dt ----------------------
    # We need this to translate cell-id-level edges into row-level edges
    # for every year.
    cell_dt[, row_num := .I]
    row_map <- cell_dt[, .(id, year, row_num)]
    setkey(row_map, id, year)

    # --- Cross-join edges Ã— years to get row-level edge table ----------
    years <- sort(unique(cell_dt$year))

    # Expand edge_id by year (vectorised via CJ-merge)
    edge_id_year <- edge_id[, .(from_id, to_id, year = rep(list(years), .N))]
    # More memory-friendly: use a direct cross join
    edge_id_year <- CJ_edge <- edge_id[,
        CJ(pair = .I, year = years)
    ]
    edge_id_year[, `:=`(
        from_id = edge_id$from_id[pair],
        to_id   = edge_id$to_id[pair]
    )]
    edge_id_year[, pair := NULL]

    # Join to get row_i (the "focal" row) and row_j (the neighbor row)
    setkey(edge_id_year, from_id, year)
    edge_id_year[row_map, row_i := i.row_num, on = .(from_id = id, year)]

    setkey(edge_id_year, to_id, year)
    edge_id_year[row_map, row_j := i.row_num, on = .(to_id = id, year)]

    # Drop edges where either side is missing (cell-year not in data)
    edge_dt <- edge_id_year[!is.na(row_i) & !is.na(row_j),
                            .(row_i, row_j)]
    setkey(edge_dt, row_i)

    cell_dt[, row_num := NULL]
    return(edge_dt)
}

# ============================================================
# 2. Compute all neighbor features in one vectorised pass
#    (replaces compute_neighbor_stats + outer for-loop)
#
#    Adds 15 columns to cell_dt by reference:
#      {var}_neighbor_max, {var}_neighbor_min, {var}_neighbor_mean
#    for each var in neighbor_source_vars.
# ============================================================
compute_all_neighbor_features <- function(cell_dt, edge_dt,
                                          neighbor_source_vars) {

    # We will build a small table of neighbor values per edge,
    # then aggregate.  To keep peak memory manageable on 16 GB,
    # we process one variable at a time but add columns by reference
    # (no full-frame copy).

    n <- nrow(cell_dt)

    for (var_name in neighbor_source_vars) {

        vals <- cell_dt[[var_name]]

        # Attach the neighbor's value to every edge
        agg <- edge_dt[, .(neighbor_val = vals[row_j]), by = row_i]

        # Remove NAs before aggregation
        agg <- agg[!is.na(neighbor_val)]

        # Grouped aggregation â€” single pass, C-optimised in data.table
        stats <- agg[, .(
            nb_max  = max(neighbor_val),
            nb_min  = min(neighbor_val),
            nb_mean = mean(neighbor_val)
        ), by = row_i]

        # Initialise new columns to NA
        max_col  <- paste0(var_name, "_neighbor_max")
        min_col  <- paste0(var_name, "_neighbor_min")
        mean_col <- paste0(var_name, "_neighbor_mean")

        set(cell_dt, j = max_col,  value = NA_real_)
        set(cell_dt, j = min_col,  value = NA_real_)
        set(cell_dt, j = mean_col, value = NA_real_)

        # Fill in computed values by reference
        set(cell_dt, i = stats$row_i, j = max_col,  value = stats$nb_max)
        set(cell_dt, i = stats$row_i, j = min_col,  value = stats$nb_min)
        set(cell_dt, i = stats$row_i, j = mean_col, value = stats$nb_mean)

        rm(agg, stats)
    }

    invisible(cell_dt)
}

# ============================================================
# 3. Optimised prediction workflow
# ============================================================
run_optimised_pipeline <- function(cell_data_path,
                                   model_path,
                                   id_order,
                                   rook_neighbors_unique,
                                   neighbor_source_vars,
                                   predictor_names,
                                   output_path) {

    cat("Loading cell data...\n")
    cell_dt <- as.data.table(readRDS(cell_data_path))
    # (or fread() if CSV)

    # ----------------------------------------------------------
    # A. Build flat edge table (replaces build_neighbor_lookup)
    # ----------------------------------------------------------
    cat("Building neighbor edge table...\n")
    edge_dt <- build_neighbor_edge_table(cell_dt, id_order,
                                         rook_neighbors_unique)
    cat(sprintf("  Edge table: %s edges\n", format(nrow(edge_dt), big.mark = ",")))

    # ----------------------------------------------------------
    # B. Compute all 15 neighbor features in-place
    # ----------------------------------------------------------
    cat("Computing neighbor features...\n")
    compute_all_neighbor_features(cell_dt, edge_dt, neighbor_source_vars)
    rm(edge_dt)
    gc()

    # ----------------------------------------------------------
    # C. Load trained RF model ONCE
    # ----------------------------------------------------------
    cat("Loading trained Random Forest model...\n")
    rf_model <- readRDS(model_path)

    # ----------------------------------------------------------
    # D. Prepare prediction matrix
    #    Build a plain matrix of predictors â€” avoids data.frame
    #    overhead inside predict().
    # ----------------------------------------------------------
    cat("Preparing prediction matrix...\n")
    pred_matrix <- as.matrix(cell_dt[, ..predictor_names])

    # If the model is from the randomForest package, predict()
    # accepts a data.frame; for ranger, a data.frame or matrix.
    # We convert to data.frame only if required:
    is_ranger <- inherits(rf_model, "ranger")

    if (!is_ranger) {
        pred_input <- as.data.frame(pred_matrix)
    } else {
        pred_input <- pred_matrix
    }
    rm(pred_matrix)
    gc()

    # ----------------------------------------------------------
    # E. Single vectorised predict() call
    # ----------------------------------------------------------
    cat("Running Random Forest prediction on all rows...\n")
    t0 <- proc.time()

    if (is_ranger) {
        preds <- predict(rf_model, data = pred_input)$predictions
    } else {
        # randomForest package
        preds <- predict(rf_model, newdata = pred_input)
    }

    elapsed <- (proc.time() - t0)["elapsed"]
    cat(sprintf("  Prediction completed in %.1f seconds.\n", elapsed))

    # ----------------------------------------------------------
    # F. Attach predictions by reference (no copy)
    # ----------------------------------------------------------
    cell_dt[, predicted_gdp := preds]
    rm(pred_input, preds)
    gc()

    # ----------------------------------------------------------
    # G. Save results
    # ----------------------------------------------------------
    cat("Saving results...\n")
    saveRDS(cell_dt, output_path)
    # Or: fwrite(cell_dt, output_path)

    cat("Done.\n")
    invisible(cell_dt)
}

# ============================================================
# 4. ENTRY POINT â€” example invocation
# ============================================================
# (Adjust paths and variable names to your environment)

# neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")
# predictor_names      <- <character vector of your ~110 predictor column names>
#
# result <- run_optimised_pipeline(
#     cell_data_path       = "cell_data.rds",
#     model_path           = "trained_rf_model.rds",
#     id_order             = id_order,
#     rook_neighbors_unique = rook_neighbors_unique,
#     neighbor_source_vars = neighbor_source_vars,
#     predictor_names      = predictor_names,
#     output_path          = "cell_data_predicted.rds"
# )
```

---

## 4. MEMORY NOTE FOR 16 GB LAPTOP

The edge table expansion (`edge_id Ã— years`) can temporarily spike memory. If the ~1.37M directed edges Ã— 28 years â‰ˆ 38.4M edge-rows exceed available headroom, process years in batches:

```r
# Memory-safe alternative inside build_neighbor_edge_table:
# Process in year-batches of 4 and rbindlist at the end.
edge_list <- vector("list", length(years))
for (yi in seq_along(years)) {
    yr <- years[yi]
    yr_rows <- row_map[year == yr]
    setkey(yr_rows, id)
    tmp <- edge_id[yr_rows, on = .(from_id = id), nomatch = 0L,
                   allow.cartesian = TRUE]
    setnames(tmp, "row_num", "row_i")
    tmp[yr_rows, row_j := i.row_num, on = .(to_id = id)]
    edge_list[[yi]] <- tmp[!is.na(row_j), .(row_i, row_j)]
}
edge_dt <- rbindlist(edge_list)
setkey(edge_dt, row_i)
```

This keeps peak memory well within 16 GB while preserving the same numerical output.

---

### Key Guarantees

| Requirement | How it is met |
|---|---|
| **Trained RF model preserved** | Model is loaded via `readRDS()` and used read-only; never retrained or modified. |
| **Original numerical estimand preserved** | The same 15 neighbor statistics (max, min, mean for each of 5 variables) are computed with identical semantics (NA handling matches original). `predict()` is called on the same model with the same features, producing identical predictions. |
| **Correctness** | The flat edge table encodes exactly the same neighbor relationships as the original `build_neighbor_lookup`; grouped `data.table` aggregation computes the same `max/min/mean` as the original `lapply`. |
END_RESPONSE>>>
