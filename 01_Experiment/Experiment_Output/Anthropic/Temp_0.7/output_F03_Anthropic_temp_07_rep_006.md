 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Why it hurts |
|---|---|
| **Row-level `lapply` over 6.46 M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering — all interpreted R. |
| **`paste(id, year)` string keys + named-vector lookup** | For 6.46 M rows × ~4 neighbors each ≈ 25 M string constructions and hash lookups. String hashing is slow and memory-hungry. |
| **Output is a 6.46 M–element list of integer vectors** | ~200–400 MB of list overhead alone; every element is a separately allocated SEXP. |

**`compute_neighbor_stats`** compounds the problem:

| Problem | Why it hurts |
|---|---|
| **Another `lapply` over 6.46 M rows** | Each call subsets a numeric vector, removes NAs, computes three summary stats — 5 times (once per variable). That is ~32 M interpreted iterations total. |
| **`do.call(rbind, result)` on a 6.46 M–element list** | Materialises a huge list-of-vectors into a matrix — known to be O(n²) for large lists in base R. |

**Combined cost estimate for feature prep alone: tens of hours on the stated hardware.**

### B. Prediction / Inference Bottlenecks

| Problem | Why it hurts |
|---|---|
| **Predicting 6.46 M rows at once with `predict.randomForest` (or `ranger::predict`)** | `randomForest::predict` builds an N × ntree matrix internally. With 6.46 M rows and (say) 500 trees, that is a ~24 GB dense matrix — exceeds 16 GB RAM, triggers swapping or crashes. |
| **Model loaded from disk every run (if applicable)** | A large RF model can be 1–4 GB; repeated `readRDS` is wasteful. |
| **No chunked prediction** | Even with `ranger`, a single monolithic predict call peaks at high memory. |
| **Potential `data.frame` copies** | Adding columns one-at-a-time with `cell_data$new_col <- ...` can trigger full-frame copies under R's copy-on-modify semantics. |

### C. Summary of Root Causes (ordered by impact)

1. **`build_neighbor_lookup`**: millions of string operations in interpreted R.
2. **`compute_neighbor_stats`**: millions of per-row `lapply` calls, repeated 5×.
3. **`do.call(rbind, ...)` on million-element lists**.
4. **Monolithic RF prediction** on 6.46 M rows exceeds RAM.
5. **Repeated column-binding** triggers copy-on-modify.

---

## 2. Optimization Strategy

### Feature Preparation → Vectorised `data.table` Joins

Replace the entire lookup + stats pipeline with:

1. **Build a flat edge table** (`cell_id → neighbor_id`) from the `nb` object — one-time, vectorised.
2. **Join the edge table to the panel data** by `(neighbor_id, year)` using `data.table` keyed joins — O(n log n), fully vectorised in C.
3. **Group-by aggregate** `(cell_id, year)` to compute `max`, `min`, `mean` — single pass per variable, fully vectorised.

This eliminates all `lapply`, all `paste`-key lookups, and all `do.call(rbind, ...)`.

### Prediction → Chunked Inference

Split the 6.46 M rows into chunks (~500 K each) and call `predict()` per chunk, then `rbind` the results. This keeps peak memory well within 16 GB.

### Additional

- Use `data.table::set()` or pre-allocate columns to avoid copy-on-modify.
- Load the model once with `readRDS` and reuse the in-memory object.
- If the model is `randomForest`, consider converting to `ranger` format (or simply use the model as-is with chunked predict).

**Expected speedup: from 86+ hours → roughly 10–30 minutes** (feature prep drops from hours to minutes; prediction from hours to minutes).

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMISED PIPELINE — Feature Preparation + Chunked RF Prediction
# =============================================================================
# Requirements: data.table (>= 1.14), ranger or randomForest (whichever was
# used to train the model).
# The trained model object and the original numerical estimand are preserved
# exactly — we only change how features are built and how predict() is called.
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
  # id_order maps position → cell_id.
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

    # --- a. Build a slim lookup: neighbor_id + year → value ---
    # Columns: neighbor_id, year, value
    val_dt <- cell_dt[, .(neighbor_id = id, year, value = get(var))]
    setkey(val_dt, neighbor_id, year)

    # --- b. Join edge table to panel to get (cell_id, year) per edge,
    #         then join to val_dt to get the neighbor's value. ---
    #
    # Start from edge_dt, add the year dimension by joining to the
    # focal cell's years.  Because every cell_id appears for every year
    # it has data, we join on cell_id → id to pull in the year column.

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

## 4. Why This Works — Performance Summary

| Component | Before (base R) | After (data.table) | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` | ~6.46 M `paste` + hash lookups | Single vectorised edge table build | ~100–500× |
| `compute_neighbor_stats` (×5 vars) | ~32 M `lapply` iterations + `do.call(rbind)` | 5 keyed joins + group-by aggregations | ~50–200× |
| RF prediction (6.46 M rows) | Single call, ~24 GB peak → swap/crash | 13 chunks × 500 K rows, ~2–3 GB peak | Fits in 16 GB; avoids swap |
| Column addition | Copy-on-modify each time | `data.table` set-by-reference | No copies |

**Estimated total wall-clock time: 10–30 minutes** on the described 16 GB laptop, down from 86+ hours.

### Key Guarantees

- **Trained RF model is never retrained** — only `predict()` is called.
- **Numerical estimand is preserved exactly** — the same features are computed (max, min, mean of the same neighbor values), and the same model produces the same predictions. No approximation is introduced.