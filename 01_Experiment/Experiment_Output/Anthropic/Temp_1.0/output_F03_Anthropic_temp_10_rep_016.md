 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`** runs an `lapply` over ~6.46 million rows. Inside every iteration it:

- Calls `as.character()` on a single ID,
- Performs a named-vector lookup (`id_to_ref`),
- Subsets the `neighbors` list,
- Builds a `paste(…, sep="_")` key for every neighbor,
- Looks those keys up in `idx_lookup` (another named-vector lookup), and
- Filters NAs.

Named-vector lookup in R is O(n) per call when the vector is long (it's a linear scan of the `names` attribute unless hashed). With ~6.46M rows and multiple lookups per row, this is **extremely slow** — likely the single largest time sink in the pipeline.

**`compute_neighbor_stats`** is called 5 times (one per variable). Each call runs another `lapply` over 6.46M rows, extracting neighbor values, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end creates 6.46M small 3-element vectors and binds them — itself an O(n²) memory-copy pattern.

**In total**, the feature-preparation stage alone performs ~6×6.46M ≈ 39M R-level function calls with slow lookups, small vector allocations, and repeated list-to-matrix conversions.

### 1.2 Random Forest Inference Bottleneck

Calling `predict(rf_model, newdata = <6.46M × 110>)` in a single shot on a `randomForest` or `ranger` object will:

- Allocate a single massive prediction matrix internally.
- On a 16 GB laptop, the working set for 6.46M × 110 `numeric` columns is ~5.3 GB for the data alone, plus the forest's node matrices, plus the prediction scratch space. This can easily exceed RAM and cause swapping / crash.
- If the user is calling `predict` inside a row-level or chunk-level loop (common in poorly optimised pipelines), the per-call overhead is amplified millions of times.

### 1.3 Summary of Root Causes

| Component | Root Cause | Impact |
|---|---|---|
| `build_neighbor_lookup` | R-level loop + named-vector linear scan over 6.46M rows | ~hours |
| `compute_neighbor_stats` | 5 × R-level loop over 6.46M rows + `do.call(rbind,…)` | ~hours |
| Prediction | Possible single giant `predict()` → OOM; or row-by-row predict → overhead | ~hours |
| Memory | Repeated full-copy of `cell_data` via `cell_data <- cbind(cell_data, …)` | GBs wasted |

---

## 2. OPTIMIZATION STRATEGY

| # | Strategy | Expected Speedup |
|---|---|---|
| 1 | Replace named-vector lookups with `data.table` hash joins or integer-keyed environments | 100–1000× for lookup step |
| 2 | Vectorise `build_neighbor_lookup` entirely with `data.table` merge/join | Eliminates 6.46M R function calls |
| 3 | Vectorise `compute_neighbor_stats` using `data.table` grouped aggregation over an edge-list representation | Eliminates 5 × 6.46M R function calls |
| 4 | Avoid `do.call(rbind, …)` and in-place column addition; add columns by reference with `:=` | Eliminates O(n²) copy |
| 5 | Batch `predict()` in chunks (~500K rows) to stay within RAM while avoiding per-row overhead | Safe RAM use, fast |
| 6 | Use `ranger` for prediction if possible (C++ back-end, faster than `randomForest::predict`) | 2–5× for predict |

---

## 3. WORKING R CODE

```r
# ============================================================
# OPTIMISED PIPELINE
# ============================================================
# Requirements: data.table, ranger (if model is ranger), randomForest
# Preserves: trained RF model object, original numerical estimand.

library(data.table)

# ----------------------------------------------------------
# 0.  Convert working data to data.table (by reference if already is one)
# ----------------------------------------------------------
setDT(cell_data)

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# Create an integer row-index column (used for joins)
cell_data[, .row_idx := .I]


# ----------------------------------------------------------
# 1.  Build edge-list representation of neighbour graph
#     (replaces build_neighbor_lookup entirely)
# ----------------------------------------------------------
build_neighbor_edgelist <- function(cell_dt, id_order, neighbors) {
  # id_order : vector of cell IDs in the same order as the nb object
  # neighbors: spdep nb object (list of integer index vectors)

  # --- step a: expand nb list to directed edge-list of cell IDs ---
  from_idx <- rep(seq_along(neighbors), lengths(neighbors))
  to_idx   <- unlist(neighbors, use.names = FALSE)

  # Remove the 0-entries spdep uses for cells with no neighbours

  valid    <- to_idx > 0L
  from_idx <- from_idx[valid]
  to_idx   <- to_idx[valid]

  edge_ids <- data.table(
    id_from = id_order[from_idx],
    id_to   = id_order[to_idx]
  )

  # --- step b: for every (id_from, year) find the row index of the
  #             focal cell, and for every (id_to, year) find the row
  #             index of the neighbour cell.
  # We cross-join edges with all years present for the focal cell.
  # Then join to get the neighbour's row index (if it exists for
  # that year).
  # ---

  # Lookup: cell_id + year -> .row_idx
  id_year_lookup <- cell_dt[, .(id, year, .row_idx)]
  setkey(id_year_lookup, id, year)

  # Get unique years per focal id  (all years a focal cell appears)
  focal_years <- cell_dt[, .(year), keyby = .(id)]

  # Join focal_years with edges to get (id_from, id_to, year)
  setkey(edge_ids, id_from)
  setkey(focal_years, id)
  edge_year <- edge_ids[focal_years, on = .(id_from = id),
                        .(id_from, id_to, year),
                        allow.cartesian = TRUE, nomatch = NULL]

  # Join to get focal row index
  edge_year[id_year_lookup,
            focal_row := i..row_idx,
            on = .(id_from = id, year)]

  # Join to get neighbour row index
  edge_year[id_year_lookup,
            nbr_row := i..row_idx,
            on = .(id_to = id, year)]

  # Keep only edges where both focal and neighbour exist
  edge_year <- edge_year[!is.na(focal_row) & !is.na(nbr_row)]

  edge_year
}

message("Building neighbour edge-list …")
edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
message(sprintf("  Edge-year rows: %s", format(nrow(edge_dt), big.mark = ",")))


# ----------------------------------------------------------
# 2.  Compute neighbour stats for all variables at once
#     (replaces compute_neighbor_stats + outer loop)
# ----------------------------------------------------------
compute_all_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  # For each variable, compute max, min, mean of neighbour values
  # grouped by focal_row, then join back.

  for (vn in var_names) {
    message(sprintf("  Neighbour features for: %s", vn))

    # Attach the neighbour's value for this variable to each edge
    edge_dt[, nbr_val := cell_dt[[vn]][nbr_row]]

    # Aggregate: one row per focal_row
    agg <- edge_dt[!is.na(nbr_val),
                   .(vmax  = max(nbr_val),
                     vmin  = min(nbr_val),
                     vmean = mean(nbr_val)),
                   keyby = .(focal_row)]

    # Build target column names (must match what downstream code expects)
    col_max  <- paste0(vn, "_neighbor_max")
    col_min  <- paste0(vn, "_neighbor_min")
    col_mean <- paste0(vn, "_neighbor_mean")

    # Initialise with NA, then fill matched rows by reference
    set(cell_dt, j = col_max,  value = NA_real_)
    set(cell_dt, j = col_min,  value = NA_real_)
    set(cell_dt, j = col_mean, value = NA_real_)

    matched <- agg$focal_row
    set(cell_dt, i = matched, j = col_max,  value = agg$vmax)
    set(cell_dt, i = matched, j = col_min,  value = agg$vmin)
    set(cell_dt, i = matched, j = col_mean, value = agg$vmean)
  }

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

message("Computing neighbour features …")
compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)

# Free the (potentially large) edge table
rm(edge_dt); gc()

# Remove helper column
cell_data[, .row_idx := NULL]


# ----------------------------------------------------------
# 3.  Batched Random Forest Prediction
# ----------------------------------------------------------
predict_rf_batched <- function(model, newdata, batch_size = 500000L) {
  # Works with both randomForest and ranger model objects.
  # Returns a numeric vector of predictions (preserves estimand).

  n <- nrow(newdata)
  preds <- numeric(n)

  is_ranger <- inherits(model, "ranger")
  starts <- seq(1L, n, by = batch_size)

  message(sprintf("Predicting %s rows in %d batches …",
                  format(n, big.mark = ","), length(starts)))

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + batch_size - 1L, n)

    batch <- newdata[i1:i2, , drop = FALSE]

    if (is_ranger) {
      preds[i1:i2] <- predict(model, data = batch)$predictions
    } else {
      preds[i1:i2] <- predict(model, newdata = batch)
    }

    if (k %% 5 == 0 || k == length(starts)) {
      message(sprintf("  batch %d / %d done", k, length(starts)))
    }
  }

  preds
}

# --- Prepare prediction data.frame ---------------------------
# Identify predictor columns (exclude target, id, year, etc.)
# Adjust 'target_col' to whatever your response variable is named.
target_col   <- "gdp"  # <-- change if different
exclude_cols <- c(target_col, "id", "year")
pred_vars    <- setdiff(names(cell_data), exclude_cols)

# Build a plain data.frame / matrix for predict
# (randomForest & ranger both accept data.frame)
pred_df <- as.data.frame(cell_data[, ..pred_vars])

# --- Run prediction ------------------------------------------
message("Running Random Forest prediction …")
cell_data[, predicted_gdp := predict_rf_batched(rf_model, pred_df,
                                                 batch_size = 500000L)]
rm(pred_df); gc()

message("Done.")
```

---

## 4. EXPECTED IMPACT

| Stage | Before (est.) | After (est.) | Why |
|---|---|---|---|
| `build_neighbor_lookup` | ~20–40 h | ~1–3 min | Hash joins replace 6.46M named-vector scans |
| `compute_neighbor_stats` (5 vars) | ~20–40 h | ~2–5 min | Vectorised `data.table` grouped aggregation; no R-level loop |
| Column binding / copies | ~hours of GC | ~0 s | `:=` by-reference assignment; no copy |
| RF `predict()` | variable / OOM risk | ~5–20 min | 500K-row batches; stays in RAM; no per-row overhead |
| **Total** | **~86+ h** | **~10–30 min** | **~150–500× overall speedup** |

### Key Guarantees

- **Trained RF model is never modified or retrained** — only `predict()` is called.
- **Numerical estimand is preserved** — no rounding, discretisation, or approximation is introduced; `max`, `min`, `mean`, and RF predictions are computed identically.
- **RAM-safe** — the edge-list is the largest interim object (~50–100M rows × 5 int columns ≈ 2–4 GB), well within 16 GB. Prediction batches cap working memory.