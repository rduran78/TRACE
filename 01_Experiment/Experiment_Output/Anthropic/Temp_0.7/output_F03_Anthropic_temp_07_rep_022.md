 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### Primary Bottlenecks

**A. `build_neighbor_lookup` — O(n) `lapply` with per-row string operations**

The function iterates over ~6.46 million rows, and for every single row it:
1. Converts `data$id[i]` to character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Pastes each neighbor cell ID with the year to create string keys.
4. Looks up those keys in `idx_lookup` (another named character vector).

Named-vector lookup in R is **hash-based but has significant per-call overhead** when done millions of times inside `lapply`. The `paste()` calls create millions of temporary character vectors. The result: this single function likely takes **hours** on 6.46M rows.

**B. `compute_neighbor_stats` — repeated per-variable full-data sweeps**

Called 5 times (once per neighbor source variable). Each call does an `lapply` over 6.46M rows, subsetting a numeric vector by index vectors and computing `max/min/mean`. The `lapply` + `do.call(rbind, ...)` pattern on 6.46M list elements is extremely slow — `do.call(rbind, ...)` on a list of millions of small vectors is a known R anti-pattern (quadratic memory allocation).

**C. Object copying in the outer loop**

`cell_data <- compute_and_add_neighbor_features(cell_data, ...)` likely triggers full-copy of the ~6.46M × 110-column data.frame on each iteration (R's copy-on-modify semantics). With 5 variables, that's 5 full copies of a multi-GB object.

**D. Random Forest prediction**

Predicting 6.46M rows × 110 features through a Random Forest (especially one with many trees) is inherently expensive. If `predict()` is called row-by-row or in small batches rather than as a single vectorized call, the overhead multiplies enormously. Model loading from disk (if done repeatedly) also adds cost.

### Summary of Time Sinks (estimated share of 86+ hours)

| Component | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~30-40% | Per-row string ops, named-vector lookup ×6.46M |
| `compute_neighbor_stats` (×5) | ~25-35% | `lapply` + `do.call(rbind,...)` on millions of rows |
| Data.frame copying (outer loop) | ~10-15% | Copy-on-modify, repeated column binding |
| RF prediction | ~15-25% | Large matrix, possibly suboptimal call pattern |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Replace row-level R loops with vectorized / `data.table` operations

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Pre-explode the `nb` object into a `data.table` edge list; merge with data by `(neighbor_id, year)` to get row indices — fully vectorized, no per-row `paste`/lookup | 50–200× |
| `compute_neighbor_stats` | Group-by aggregation on the edge-list `data.table` using `[, .(max, min, mean), by = source_row]` | 50–100× |
| `do.call(rbind, ...)` | Eliminated entirely — results come from `data.table` aggregation | 10–50× |
| Data.frame copy | Use `data.table` with `:=` (modify in place) — zero copies | 5× per iteration |
| RF prediction | Single `predict()` call on the full `data.table`/matrix; load model once | Ensures no unnecessary overhead |

**Target runtime: ~5–20 minutes** (down from 86+ hours).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Requirements: data.table, ranger (or randomForest — adapts to either)
# Preserves: trained RF model object, original numerical estimand
# =============================================================================

library(data.table)

# ---- 0. Load model once ------------------------------------------------------
# Adjust path/object name to your setup.
# rf_model <- readRDS("path/to/trained_rf_model.rds")
# The model is assumed to already be in memory as `rf_model`.

# ---- 1. Convert working data to data.table (in-place efficiency) -------------
# Assumes `cell_data` is your ~6.46M-row data.frame/data.table with columns
#   id, year, ntl, ec, pop_density, def, usd_est_n2, ... (110 predictors)
# Assumes `id_order` is the vector mapping position in the nb object to cell id.
# Assumes `rook_neighbors_unique` is the spdep::nb list.

if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place — no copy
}

# ---- 2. Build a vectorized edge list from the nb object ----------------------
# This replaces `build_neighbor_lookup` entirely.

build_edge_list <- function(id_order, nb_obj) {
  # Explode the nb list into (source_position, neighbor_position) pairs
  n <- length(nb_obj)
  source_pos <- rep(seq_len(n), lengths(nb_obj))
  neighbor_pos <- unlist(nb_obj)

  # Remove the spdep convention where 0 means "no neighbors"
  valid <- neighbor_pos != 0L
  source_pos <- source_pos[valid]
  neighbor_pos <- neighbor_pos[valid]

  # Map positions back to cell IDs
  data.table(
    source_id   = id_order[source_pos],
    neighbor_id = id_order[neighbor_pos]
  )
}

cat("Building edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- 3. Compute all neighbor features in one vectorized pass -----------------
# Strategy:
#   - Join edge_dt with cell_data on (neighbor_id, year) to pull neighbor values.
#   - Group by (source_id, year) to compute max/min/mean.
#   - Join results back to cell_data using := (in-place, no copy).

compute_all_neighbor_features <- function(dt, edge_dt, var_names) {
  # Ensure keys for fast joins
  # We need a row-reference column so we can join back
  dt[, .row_idx := .I]

  # Slim table for joining: only id, year, and the source variables
  # Plus the row index for final join-back
  cols_needed <- c("id", "year", var_names, ".row_idx")
  slim <- dt[, ..cols_needed]

  # --- Step A: Expand edges × years -------------------------------------------
  # For each (source_id, neighbor_id) edge, we need every year present in the data.
  # Instead of a cross-join (expensive), we join edge_dt to the data directly.

  # Create the lookup: for each (id, year) → values of the source vars
  # Key the slim table on (id, year) for fast join
  setkey(slim, id, year)

  # Join: for each edge, pull the neighbor's variable values for every year
  # edge_dt has (source_id, neighbor_id)
  # We want: for each row in cell_data identified by (source_id, year),
  #          find all neighbors, look up their values at the same year.

  # Step A1: Get (source_id, year) pairs from the data
  source_years <- slim[, .(source_id = id, year, .row_idx)]

  # Step A2: Cross edge list with years via join on source_id
  #   Result: (source_id, year, neighbor_id, .row_idx)
  cat("  Joining edges with year dimension...\n")
  setkey(edge_dt, source_id)
  setkey(source_years, source_id)
  expanded <- edge_dt[source_years, on = "source_id", allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded now has columns: source_id, neighbor_id, year, .row_idx
  # .row_idx refers to the source row in cell_data

  # Step A3: Look up neighbor values by (neighbor_id, year)
  cat("  Looking up neighbor values...\n")
  setnames(slim, "id", "neighbor_id")  # rename for join
  # Keep only the var columns + key columns in the right-side table
  neighbor_vals <- slim[, c("neighbor_id", "year", var_names), with = FALSE]
  setkey(neighbor_vals, neighbor_id, year)
  setkey(expanded, neighbor_id, year)

  merged <- neighbor_vals[expanded, on = c("neighbor_id", "year"), nomatch = NA]
  # merged has: neighbor_id, year, <var_names>, source_id, .row_idx

  # Step A4: Aggregate by source row
  cat("  Computing neighbor statistics...\n")
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(var_names, function(v) {
    vn <- as.name(v)
    list(
      bquote(max(.(vn), na.rm = TRUE)),
      bquote(min(.(vn), na.rm = TRUE)),
      bquote(mean(.(vn), na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(var_names, function(v) {
    paste0("n_", v, c("_max", "_min", "_mean"))
  }))
  names(agg_exprs) <- agg_names

  # Suppress -Inf/Inf warnings from max/min on empty sets
  agg_result <- suppressWarnings(
    merged[, lapply(agg_exprs, eval, envir = .SD), by = .row_idx]
  )

  # Replace Inf/-Inf with NA (from groups where all neighbor values were NA)
  for (col_name in agg_names) {
    set(agg_result, which(is.infinite(agg_result[[col_name]])), col_name, NA_real_)
  }

  # Step A5: Join back to cell_data by .row_idx (in place)
  cat("  Joining neighbor features back to main table...\n")
  setkey(agg_result, .row_idx)
  setkey(dt, .row_idx)

  for (col_name in agg_names) {
    dt[agg_result, (col_name) := get(paste0("i.", col_name)), on = ".row_idx"]
  }

  # Clean up
  dt[, .row_idx := NULL]
  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorized)...\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})
# cell_data now has 15 new columns: n_{var}_{max,min,mean} for each of the 5 vars
# All added in-place via := — no copies of the 6.46M-row table.

cat(sprintf("  cell_data now has %d columns\n", ncol(cell_data)))

# ---- 4. Random Forest prediction (single vectorized call) --------------------

predict_gdp <- function(dt, model, predictor_cols = NULL, batch_size = 500000L) {
  # Determine predictor columns
  if (is.null(predictor_cols)) {
    # Attempt to extract from model
    if (inherits(model, "ranger")) {
      predictor_cols <- model$forest$independent.variable.names
    } else if (inherits(model, "randomForest")) {
      # randomForest stores variable names used in training
      predictor_cols <- rownames(model$importance)
    } else {
      stop("Cannot auto-detect predictor columns. Please supply `predictor_cols`.")
    }
  }

  # Verify all predictor columns exist
  missing_cols <- setdiff(predictor_cols, names(dt))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing predictor columns in data: %s",
                 paste(missing_cols, collapse = ", ")))
  }

  n <- nrow(dt)
  cat(sprintf("Predicting %s rows with %d predictors...\n",
              format(n, big.mark = ","), length(predictor_cols)))

  # For very large datasets, predict in batches to manage memory peaks
  # while still using vectorized predict() within each batch.
  if (n <= batch_size) {
    # Single call — most efficient
    if (inherits(model, "ranger")) {
      preds <- predict(model, data = dt[, ..predictor_cols])$predictions
    } else {
      preds <- predict(model, newdata = dt[, ..predictor_cols])
    }
  } else {
    # Batch prediction to avoid memory spikes on 16 GB laptop
    n_batches <- ceiling(n / batch_size)
    cat(sprintf("  Using %d batches of ~%s rows\n",
                n_batches, format(batch_size, big.mark = ",")))
    preds <- numeric(n)
    for (b in seq_len(n_batches)) {
      start_idx <- (b - 1L) * batch_size + 1L
      end_idx   <- min(b * batch_size, n)
      batch_data <- dt[start_idx:end_idx, ..predictor_cols]

      if (inherits(model, "ranger")) {
        preds[start_idx:end_idx] <- predict(model, data = batch_data)$predictions
      } else {
        preds[start_idx:end_idx] <- predict(model, newdata = batch_data)
      }

      if (b %% 5 == 0 || b == n_batches) {
        cat(sprintf("  Batch %d/%d complete\n", b, n_batches))
      }
    }
  }

  return(preds)
}

cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_gdp(cell_data, rf_model)]
})

cat("Pipeline complete.\n")
cat(sprintf("  Rows predicted: %s\n", format(nrow(cell_data), big.mark = ",")))
cat(sprintf("  Output column:  'predicted_gdp'\n"))

# ---- 5. (Optional) Memory cleanup -------------------------------------------
rm(edge_dt)
gc()
```

---

## 4. WHY THIS WORKS — KEY CHANGES SUMMARIZED

| Original | Optimized | Mechanism |
|---|---|---|
| `build_neighbor_lookup`: `lapply` over 6.46M rows with `paste`, named-vector lookup | `build_edge_list`: vectorized `rep`/`unlist` on the `nb` object → `data.table` with ~1.37M edges | Eliminates per-row string operations entirely |
| `compute_neighbor_stats`: `lapply` over 6.46M rows + `do.call(rbind, 6.46M-element list)` × 5 variables | Single `data.table` equi-join + grouped aggregation for all 5 variables simultaneously | `data.table` parallelized C-level grouping; no R-level row loop |
| `cell_data <- cbind(cell_data, ...)` × 5 | `cell_data[, (cols) := values]` in place | Zero-copy column addition via `:=` |
| Potentially row-by-row or repeated `predict()` calls | Single (or large-batch) `predict()` call on the full feature matrix | Amortizes tree-traversal setup; vectorized C/C++ path in ranger/randomForest |
| Model potentially reloaded per iteration | Model loaded once, held in memory | Eliminates repeated deserialization |

**Expected runtime on a 16 GB laptop: ~5–20 minutes** (depending on the number of RF trees and their depth), down from the estimated 86+ hours. The trained model and original numerical estimand are fully preserved — no retraining, no transformation of the target variable.