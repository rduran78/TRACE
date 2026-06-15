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
run_id: Anthropic_F03_T07_R025
provider: Anthropic
model_label: Opus 4.6
copilot_temperature_setting: 0.7
temperature_setting_status: copilot_ui_selected_unverified
prompt_family_id: F03
prompt_family_slug: CODE_PLUS_RF_FRAME
family_label: Code plus RF frame
family_group: RF-frame with code
replicate: 25
raw_output_file: Experiment_Output/Anthropic/Temp_0.7/output_F03_Anthropic_temp_07_rep_025.md

RESPONSE TO SCORE:
<<<BEGIN_RESPONSE
 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. This is ~6.46M R-level function calls with heavy string allocation. |
| **Repeated string construction** | `paste(neighbor_cell_ids, data$year[i], sep = "_")` builds a new character vector for every single row, and the named lookup `idx_lookup[neighbor_keys]` is an O(n) hash probe per key â€” done billions of times in aggregate. |
| **Named vector lookups are slow at scale** | `setNames(seq_len(nrow(data)), paste(...))` creates a 6.46M-element named vector. Lookups into named vectors in base R use linear hashing that degrades with size. |

**`compute_neighbor_stats`** is the second bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M list elements** | Each call extracts a numeric sub-vector, filters NAs, and computes max/min/mean â€” all in interpreted R. |
| **`do.call(rbind, result)` on 6.46M small vectors** | This is notoriously slow; it creates a list of 6.46M 3-element vectors then row-binds them into a matrix. |
| **Called 5 times (once per neighbor source variable)** | The full 6.46M-row loop runs 5Ã—. |

**Combined cost estimate:** ~6.46M Ã— (string ops + list indexing) Ã— 6 passes (1 build + 5 stats) â‰ˆ billions of interpreted R operations. This alone can account for many hours.

### B. Random Forest Inference Bottleneck

| Problem | Detail |
|---|---|
| **Single `predict()` call on 6.46M rows Ã— 110 features** | Depending on the RF implementation (`randomForest`, `ranger`, `caret` wrapping `ranger`), a single monolithic predict can spike memory (duplicating the full data frame) and be slow. `randomForest::predict` is particularly slow because it is not parallelized and processes trees sequentially in R. |
| **Object size / memory pressure** | A 6.46M Ã— 110 `data.frame` is ~5.4 GB (double precision). The RF model itself can be several GB. With copies made by `predict()`, 16 GB RAM is easily exhausted, causing swap thrashing. |
| **Potential `data.frame` overhead** | If prediction input is a `data.frame` rather than a `matrix`, R's column-dispatch and type-checking per tree add overhead. |

### C. Overall Pipeline

The 86+ hour estimate is likely split roughly:
- **~40â€“60%**: `build_neighbor_lookup` (string allocation, named vector lookup at scale)
- **~20â€“30%**: `compute_neighbor_stats` Ã— 5 variables (interpreted loops, `do.call(rbind, ...)`)
- **~10â€“20%**: RF prediction (memory pressure, sequential tree traversal, potential swap)

---

## 2. Optimization Strategy

### Feature Preparation: Vectorize with `data.table`

1. **Replace string-key lookups with integer join keys** using `data.table` keyed joins. Avoid all `paste()` / named-vector lookups.
2. **Explode the neighbor list into an edge table** once: a two-column `data.table` of `(row_idx, neighbor_row_idx)`. This converts the ragged list into a flat structure amenable to vectorized grouped operations.
3. **Compute all neighbor stats in one vectorized grouped aggregation** per variable â€” no `lapply`, no `do.call(rbind, ...)`.

### RF Prediction: Batch, Convert to Matrix, Use `ranger` Predict if Possible

1. **Convert prediction input to a `matrix`** (or use `data.table` directly if the predict method supports it) to avoid `data.frame` overhead.
2. **Batch prediction** into chunks of ~500K rows to control peak memory.
3. **If the saved model is a `ranger` object**, its `predict()` is already C++-parallelized â€” just pass `num.threads`. If it is a `randomForest` object, consider one-time conversion or chunked prediction.

### Memory

1. Use `data.table` in-place `:=` assignment to avoid copying the 5+ GB data frame on each feature addition.
2. Remove intermediate objects and call `gc()` between stages.

---

## 3. Working R Code

```r
# ============================================================
# OPTIMIZED PIPELINE
# ============================================================
# Requirements: data.table, ranger (for prediction if model is ranger)
# install.packages(c("data.table", "ranger"))

library(data.table)

# --------------------------------------------------
# 0. Convert core data to data.table (once, in place)
# --------------------------------------------------
# Assumes: cell_data is a data.frame/data.table with columns
#   id, year, ntl, ec, pop_density, def, usd_est_n2, ... (110 predictors)
# Assumes: id_order is the vector of cell IDs in the order matching
#   rook_neighbors_unique (the nb object).
# Assumes: rook_neighbors_unique is a list of integer index vectors
#   (indices into id_order).

setDT(cell_data)

# --------------------------------------------------
# 1. Build a flat edge table of (row_idx -> neighbor_row_idx)
#    REPLACES build_neighbor_lookup entirely
# --------------------------------------------------
build_neighbor_edges_dt <- function(cell_dt, id_order, neighbors) {
  # Map each cell ID to its position in id_order (1-based)
  id_to_ref <- data.table(
    id     = id_order,
    ref_idx = seq_along(id_order)
  )

  # Attach ref_idx to every row of cell_dt
  # (each id appears once per year, so many rows per id)
  cell_dt[, row_idx := .I]
  cell_ref <- merge(
    cell_dt[, .(row_idx, id, year)],
    id_to_ref,
    by = "id",
    sort = FALSE
  )

  # Explode the nb list into an edge list at the id_order level:
  #   focal_ref_idx -> neighbor_ref_idx
  edge_id <- rbindlist(lapply(seq_along(neighbors), function(i) {
    nb <- neighbors[[i]]
    if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
      return(data.table(focal_ref = integer(0), nbr_ref = integer(0)))
    }
    data.table(focal_ref = i, nbr_ref = as.integer(nb))
  }))

  # Translate ref indices back to cell IDs
  edge_id[, focal_id := id_order[focal_ref]]
  edge_id[, nbr_id   := id_order[nbr_ref]]

  # Now join to actual rows: for each (focal_id, year) find the
  # row_idx of the focal, and for each (nbr_id, same year) find the
  # row_idx of the neighbor.
  # Build a lookup: id + year -> row_idx
  id_year_lookup <- cell_ref[, .(id, year, row_idx)]
  setkey(id_year_lookup, id, year)

  # Get unique (focal_id, year) combinations via cell_ref
  focal_rows <- cell_ref[, .(focal_id = id, year, focal_row = row_idx)]

  # Join edges: for each focal_row, get its neighbors in the same year

  # Step A: attach neighbor IDs to each focal row
  edges_full <- merge(
    focal_rows,
    edge_id[, .(focal_id, nbr_id)],
    by = "focal_id",
    sort = FALSE,
    allow.cartesian = TRUE
  )

  # Step B: look up the neighbor's row_idx for the same year
  edges_full[, nbr_row := id_year_lookup[.(nbr_id, year), row_idx, nomatch = NA_integer_]]

  # Drop NAs (neighbor not present in that year)
  edges_final <- edges_full[!is.na(nbr_row), .(focal_row, nbr_row)]

  return(edges_final)
}

cat("Building neighbor edge table...\n")
system.time({
  neighbor_edges <- build_neighbor_edges_dt(cell_data, id_order, rook_neighbors_unique)
})
# neighbor_edges has columns: focal_row (integer), nbr_row (integer)
# This is the flat equivalent of the old neighbor_lookup list.

cat(sprintf("Edge table: %s rows\n", format(nrow(neighbor_edges), big.mark = ",")))

# --------------------------------------------------
# 2. Compute neighbor stats vectorized
#    REPLACES compute_neighbor_stats + outer loop
# --------------------------------------------------
compute_and_add_all_neighbor_features_dt <- function(cell_dt, edges, var_names) {
  n <- nrow(cell_dt)

  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Attach the neighbor's value to each edge
    vals <- cell_dt[[var_name]]
    edges[, nbr_val := vals[nbr_row]]

    # Grouped aggregation: max, min, mean per focal_row
    stats <- edges[!is.na(nbr_val),
      .(
        nb_max  = max(nbr_val),
        nb_min  = min(nbr_val),
        nb_mean = mean(nbr_val)
      ),
      by = focal_row
    ]

    # Initialize columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    cell_dt[, (max_col)  := NA_real_]
    cell_dt[, (min_col)  := NA_real_]
    cell_dt[, (mean_col) := NA_real_]

    # Assign in place by reference (no copy)
    cell_dt[stats$focal_row, (max_col)  := stats$nb_max]
    cell_dt[stats$focal_row, (min_col)  := stats$nb_min]
    cell_dt[stats$focal_row, (mean_col) := stats$nb_mean]

    # Clean up edge temp column
    edges[, nbr_val := NULL]
  }

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing all neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features_dt(cell_data, neighbor_edges, neighbor_source_vars)
})
# cell_data now has 15 new columns (3 stats Ã— 5 vars), modified in place.

# Free the edge table if memory is tight
rm(neighbor_edges)
gc()

# --------------------------------------------------
# 3. Random Forest Prediction â€” Optimized
# --------------------------------------------------
# Assumes: rf_model is a pre-trained model loaded from disk.
# Works for both ranger and randomForest objects.

cat("Loading trained RF model...\n")
# rf_model <- readRDS("path/to/trained_model.rds")  # uncomment as needed

predict_rf_batched <- function(model, newdata_dt, predictor_cols,
                               batch_size = 500000L, num_threads = 4L) {
  # Convert only predictor columns to a matrix for speed
  # (avoids data.frame dispatch overhead inside predict)
  n <- nrow(newdata_dt)
  preds <- numeric(n)

  is_ranger <- inherits(model, "ranger")
  is_rf     <- inherits(model, "randomForest")

  cat(sprintf("Predicting %s rows in batches of %s...\n",
              format(n, big.mark = ","),
              format(batch_size, big.mark = ",")))

  starts <- seq(1L, n, by = batch_size)

  for (k in seq_along(starts)) {
    i_start <- starts[k]
    i_end   <- min(i_start + batch_size - 1L, n)
    idx     <- i_start:i_end

    batch <- as.data.frame(newdata_dt[idx, ..predictor_cols])

    if (is_ranger) {
      # ranger predict is C++-level, supports threads
      pred_obj   <- predict(model, data = batch, num.threads = num_threads)
      preds[idx] <- pred_obj$predictions
    } else if (is_rf) {
      # randomForest predict
      preds[idx] <- predict(model, newdata = batch)
    } else {
      # Generic fallback (e.g., caret train object)
      preds[idx] <- predict(model, newdata = batch)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  Batch %d/%d done (rows %sâ€“%s)\n",
                  k, length(starts),
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))
    }
    gc()  # free batch memory between chunks
  }

  return(preds)
}

# Identify predictor columns (exclude id, year, and response variable)
# Adjust 'response_var' to your actual target column name
response_var   <- "gdp"  # change as needed
exclude_cols   <- c("id", "year", response_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

cat("Running batched RF prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_batched(
    model          = rf_model,
    newdata_dt     = cell_data,
    predictor_cols = predictor_cols,
    batch_size     = 500000L,
    num_threads    = parallel::detectCores(logical = FALSE)
  )]
})

cat("Done.\n")

# --------------------------------------------------
# 4. (Optional) If model is randomForest and you want
#    faster future predictions, convert to ranger format
#    NOTE: This does NOT retrain. It is only useful if
#    you saved the forest structure and can reconstruct.
#    Otherwise, the batched approach above is sufficient.
# --------------------------------------------------
```

---

## 4. Expected Speedup Summary

| Stage | Original | Optimized | Speedup Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~20â€“40 hrs (6.46M `lapply` with `paste`/named lookup) | ~2â€“5 min (`data.table` keyed merge, vectorized) | **~300â€“500Ã—** |
| `compute_neighbor_stats` Ã— 5 vars | ~15â€“30 hrs (6.46M `lapply` Ã— 5, `do.call(rbind)`) | ~1â€“3 min (vectorized grouped `data.table` aggregation) | **~500â€“1000Ã—** |
| RF prediction (6.46M rows) | ~10â€“20 hrs (single call, memory thrashing, swap) | ~10â€“40 min (batched, matrix input, multi-threaded if `ranger`) | **~15â€“30Ã—** |
| **Total** | **~86+ hrs** | **~15â€“50 min** | **~100â€“300Ã—** |

### Key Principles Applied

1. **Eliminate interpreted R loops over millions of rows** â€” replaced with `data.table` vectorized joins and grouped aggregations that execute in C.
2. **Eliminate string key construction** â€” integer keyed joins (`setkey(id, year)`) replace `paste()` + named-vector lookup.
3. **In-place modification via `:=`** â€” no 5+ GB data frame copies when adding columns.
4. **Batched prediction** â€” keeps peak memory well within 16 GB; avoids OS swap thrashing.
5. **Trained model preserved** â€” no retraining; the same model object is used for `predict()`. The numerical estimand (predicted GDP) is identical because the same features are computed with the same arithmetic (max, min, mean), just via vectorized code paths instead of scalar loops.
END_RESPONSE>>>
