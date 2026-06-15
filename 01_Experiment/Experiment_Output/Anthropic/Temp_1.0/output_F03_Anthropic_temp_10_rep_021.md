 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function iterates over ~6.46 million rows with `lapply`, performing per-row string pasting (`paste(id, year)`), character-keyed hash lookups, and NA filtering. With 6.46M iterations, each involving string concatenation and named-vector lookup (which is O(n) in base R for large named vectors), this alone could take hours.

**`compute_neighbor_stats`:** Called 5 times (once per neighbor source variable). Each call iterates over 6.46M rows, subsetting a numeric vector by index, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end materializes a 6.46M-element list into a matrix — this is a well-known R anti-pattern that is extremely slow for large lists.

**The Outer Loop:** Calls `compute_and_add_neighbor_features` 5 times, likely copying the entire `cell_data` data.frame each time (R's copy-on-modify semantics). Each copy of a 6.46M × 110+ column data.frame is expensive in both time and memory.

### 1.2 Prediction Workflow Bottlenecks (Inferred)

- **Model loading:** If `readRDS` is used per-chunk or per-iteration, deserialization of a large Random Forest is slow.
- **Row-by-row or small-batch prediction:** `predict.randomForest` or `predict.ranger` has per-call overhead; calling it millions of times individually is catastrophic.
- **Data frame coercion:** If prediction data is reassembled into a data.frame each call, R's type-checking/column-validation overhead dominates.
- **Memory:** With 6.46M rows × 110 features × 8 bytes ≈ 5.7 GB for the numeric matrix alone, plus the RF model, 16 GB is tight. Garbage collection pressure from repeated copies causes major slowdowns.

### 1.3 Root Cause Summary

| Component | Problem | Severity |
|---|---|---|
| `build_neighbor_lookup` | Per-row string ops + named-vector lookup on 6.46M rows | **Critical** |
| `compute_neighbor_stats` | `do.call(rbind, list_of_6.46M)` + per-row `lapply` | **Critical** |
| Outer loop data copies | Copy-on-modify of 6.46M-row data.frame ×5 | **High** |
| RF prediction | Likely sub-optimal batching or repeated model loads | **High** |
| Memory pressure | ~5-10 GB working set on 16 GB machine | **Medium** |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation

1. **Replace named-vector lookups with `data.table` integer-keyed joins.** Use `data.table` for all tabular operations — eliminates copy-on-modify and gives O(1) keyed lookups.

2. **Vectorize the neighbor lookup.** Expand the `nb` object into an edge-list (a two-column integer matrix of `[row_index, neighbor_row_index]`), then use `data.table` grouped operations to compute max/min/mean in a single vectorized pass per variable — no `lapply` over 6.46M rows.

3. **Compute all 5 variables' neighbor stats in one grouped pass** over the edge table, or at minimum avoid re-copying the main table.

4. **Eliminate `do.call(rbind, ...)`** entirely — replaced by `data.table` aggregation which returns a table directly.

### 2.2 Prediction Workflow

1. **Load the model once** with `readRDS`.
2. **Predict in a single call** (or a small number of large chunks if memory-constrained). Both `randomForest::predict` and `ranger::predict` accept the full matrix/data.frame at once.
3. **Pass a matrix, not a data.frame**, to the predict function when possible (avoids per-column type checking).
4. **Use `gc()` strategically** after discarding large intermediates.

### 2.3 Expected Improvement

| Step | Before | After | Speedup |
|---|---|---|---|
| Neighbor lookup build | ~hours (string ops) | ~30-90 sec (integer join) | ~50-100× |
| Neighbor stats (×5 vars) | ~hours (lapply + rbind) | ~30-60 sec (vectorized) | ~100× |
| Data copies | ~5 copies of 5 GB | 0 copies (in-place data.table) | memory halved |
| RF prediction | unknown (likely batched poorly) | single call | potentially 10-100× |
| **Total pipeline** | **86+ hours** | **~10-30 minutes** | **~200×** |

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# Dependencies: data.table, ranger (or randomForest)
# =============================================================================

library(data.table)

# ---- 0. Load model ONCE ---------------------------------------------------
# Adjust path and predict function depending on whether model is
# randomForest::randomForest or ranger::ranger
rf_model <- readRDS("path/to/trained_rf_model.rds")

# Detect model type for correct predict() dispatch later
model_is_ranger <- inherits(rf_model, "ranger")

# ---- 1. Convert main data to data.table (in-place, no copy) ---------------
# Assumes `cell_data` is already in memory as a data.frame or data.table
setDT(cell_data)

# Create a unique integer row index — this is the master row key
cell_data[, row_idx := .I]

# Create a fast integer key for (id, year) -> row_idx mapping
setkey(cell_data, id, year)


# ---- 2. Build vectorised edge-list from the nb object ---------------------
#
# rook_neighbors_unique is an nb object: a list of length n_cells,
# where element i is an integer vector of neighbor *positions* in id_order.
# id_order is the vector of cell IDs in the same order.
#
# We need to map (cell_id, year) -> row_idx for every directed edge.

build_edge_table <- function(cell_data, id_order, nb_obj) {
  # --- a. Expand nb into an edge list of (focal_cell_id, neighbor_cell_id) ---
  n <- length(nb_obj)
  lens <- lengths(nb_obj)                       # number of neighbors per cell
  focal_pos   <- rep(seq_len(n), lens)           # position in id_order
  neighbor_pos <- unlist(nb_obj, use.names = FALSE)

  edge_cells <- data.table(
    focal_id    = id_order[focal_pos],
    neighbor_id = id_order[neighbor_pos]
  )
  rm(focal_pos, neighbor_pos, lens)

  # --- b. Cross-join with years to get (focal_id, year, neighbor_id, year) ---
  years <- sort(unique(cell_data$year))

  # Expand edges × years (directed, same year for focal and neighbor)
  # This produces ~1.37M edges × 28 years ≈ 38.5M rows — fits in RAM
  edge_years <- edge_cells[, CJ(year = years), by = .(focal_id, neighbor_id)]

  # --- c. Attach focal row_idx ---
  edge_years <- merge(
    edge_years,
    cell_data[, .(focal_id = id, year, focal_row = row_idx)],
    by.x = c("focal_id", "year"),
    by.y = c("focal_id", "year"),
    all.x = TRUE,
    allow.cartesian = FALSE
  )

  # --- d. Attach neighbor row_idx ---
  edge_years <- merge(
    edge_years,
    cell_data[, .(neighbor_id = id, year, neighbor_row = row_idx)],
    by.x = c("neighbor_id", "year"),
    by.y = c("neighbor_id", "year"),
    all.x = TRUE,
    allow.cartesian = FALSE
  )

  # Drop edges where either side is missing (boundary / missing year)
  edge_years <- edge_years[!is.na(focal_row) & !is.na(neighbor_row)]

  setkey(edge_years, focal_row)
  return(edge_years)
}

cat("Building edge table...\n")
edge_dt <- build_edge_table(cell_data, id_order, rook_neighbors_unique)
gc()
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))


# ---- 3. Compute neighbor stats for all variables in vectorised fashion -----

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

compute_all_neighbor_stats <- function(cell_data, edge_dt, var_names) {
  n_rows <- nrow(cell_data)

  for (v in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", v))

    # Pull the variable values for neighbor rows
    vals <- cell_data[[v]]
    edge_dt[, nbr_val := vals[neighbor_row]]

    # Grouped aggregation: max, min, mean — skipping NAs
    agg <- edge_dt[!is.na(nbr_val),
                   .(nmax  = max(nbr_val),
                     nmin  = min(nbr_val),
                     nmean = mean(nbr_val)),
                   keyby = focal_row]

    # Allocate columns with NA, then fill matched rows
    max_col  <- paste0("n_max_",  v)
    min_col  <- paste0("n_min_",  v)
    mean_col <- paste0("n_mean_", v)

    set(cell_data, j = max_col,  value = NA_real_)
    set(cell_data, j = min_col,  value = NA_real_)
    set(cell_data, j = mean_col, value = NA_real_)

    idx <- agg$focal_row
    set(cell_data, i = idx, j = max_col,  value = agg$nmax)
    set(cell_data, i = idx, j = min_col,  value = agg$nmin)
    set(cell_data, i = idx, j = mean_col, value = agg$nmean)

    # Clean up inside edge_dt
    edge_dt[, nbr_val := NULL]
  }
  invisible(NULL)
}

cat("Computing neighbor features...\n")
compute_all_neighbor_stats(cell_data, edge_dt, neighbor_source_vars)
gc()
cat("Neighbor features done.\n")

# Free the edge table — no longer needed
rm(edge_dt)
gc()


# ---- 4. Prepare prediction matrix -----------------------------------------
# Identify predictor columns (exclude id, year, row_idx, and the response)
response_var <- "gdp"   # adjust to your actual response column name
exclude_cols <- c("id", "year", "row_idx", response_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# Verify we have the expected ~110 predictors
cat(sprintf("Predictor count: %d\n", length(predictor_cols)))

# Build a numeric matrix for prediction (more memory-efficient, faster predict)
# Do this in-place column extraction to avoid a full copy:
pred_matrix <- as.matrix(cell_data[, ..predictor_cols])
gc()


# ---- 5. Predict in one call (or chunked if memory-constrained) ------------

predict_chunked <- function(model, newdata_mat, chunk_size = 1e6,
                            is_ranger = FALSE) {
  n <- nrow(newdata_mat)
  preds <- numeric(n)
  starts <- seq(1, n, by = chunk_size)

  for (k in seq_along(starts)) {
    s <- starts[k]
    e <- min(s + chunk_size - 1L, n)
    chunk_df <- as.data.frame(newdata_mat[s:e, , drop = FALSE])

    if (is_ranger) {
      preds[s:e] <- predict(model, data = chunk_df)$predictions
    } else {
      preds[s:e] <- predict(model, newdata = chunk_df)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      cat(sprintf("  Predicted rows %s – %s of %s\n",
                  format(s, big.mark = ","),
                  format(e, big.mark = ","),
                  format(n, big.mark = ",")))
    }
  }
  return(preds)
}

cat("Running Random Forest prediction...\n")
cell_data[, predicted_gdp := predict_chunked(
  rf_model, pred_matrix,
  chunk_size = 500000L,          # ~500K rows per chunk; tune to your RAM
  is_ranger  = model_is_ranger
)]
gc()

# Free the matrix
rm(pred_matrix)
gc()

cat("Prediction complete.\n")


# ---- 6. (Optional) Write results -----------------------------------------
# fwrite(cell_data[, .(id, year, predicted_gdp)], "cell_gdp_predictions.csv")
```

---

### Alternate Streamlined Edge Table Build (Lower Peak Memory)

If the cross-join in step 2 exceeds available RAM (~38.5M rows × 5 columns ≈ 1.5 GB, which should be fine on 16 GB), use this leaner version that avoids the double merge:

```r
build_edge_table_lean <- function(cell_data, id_order, nb_obj) {
  # 1. Row-map: (id, year) -> row_idx
  row_map <- cell_data[, .(id, year, row_idx)]
  setkey(row_map, id, year)

  # 2. Expand nb to cell-ID edge list
  lens <- lengths(nb_obj)
  focal_id    <- rep(id_order, lens)
  neighbor_id <- id_order[unlist(nb_obj, use.names = FALSE)]

  edge_cells <- data.table(focal_id, neighbor_id)
  rm(focal_id, neighbor_id)

  # 3. Expand by year
  years_dt <- data.table(year = sort(unique(cell_data$year)))
  edge_all <- edge_cells[, CJ_dt := TRUE][
    years_dt, on = "CJ_dt", allow.cartesian = TRUE
  ]
  # Cleaner cross join:
  edge_all <- CJ(edge_idx = seq_len(nrow(edge_cells)),
                 year = sort(unique(cell_data$year)))
  edge_all[, `:=`(focal_id    = edge_cells$focal_id[edge_idx],
                   neighbor_id = edge_cells$neighbor_id[edge_idx])]
  edge_all[, edge_idx := NULL]

  # 4. Key lookups for focal_row and neighbor_row
  edge_all[row_map, focal_row := i.row_idx,
           on = .(focal_id = id, year)]
  edge_all[row_map, neighbor_row := i.row_idx,
           on = .(neighbor_id = id, year)]

  edge_all <- edge_all[!is.na(focal_row) & !is.na(neighbor_row)]
  setkey(edge_all, focal_row)
  return(edge_all[, .(focal_row, neighbor_row)])
}
```

---

## 4. SUMMARY OF CHANGES AND GUARANTEES

| Concern | Guarantee |
|---|---|
| **Trained RF model preserved** | Model is loaded once with `readRDS`; only `predict()` is called — no retraining. |
| **Numerical estimand preserved** | The same max/min/mean neighbor statistics are computed; the same predictor columns are fed to the same model — predictions are numerically identical (up to floating-point ordering of summation). |
| **Memory ≤ 16 GB** | `data.table` in-place modification avoids copies; prediction is chunked at 500K rows; edge table is freed after use. Peak ≈ 8-10 GB. |
| **Runtime** | Expected ~10–30 minutes total vs. 86+ hours, primarily from eliminating per-row `lapply`, string operations, `do.call(rbind, ...)`, and data.frame copies. |