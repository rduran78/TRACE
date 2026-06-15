 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector)
- NA filtering

With 6.46M iterations, each doing string concatenation and multiple named-vector lookups against a 6.46M-element vector, this is extremely slow — likely **hours** on its own.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, subsetting a numeric vector and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-element vectors is also costly (repeated memory allocation). This is called 5 times (once per neighbor source variable), so ~32.3M iterations total.

**`do.call(rbind, ...)` on millions of small vectors** is a classic R anti-pattern — it creates millions of tiny row matrices and binds them one by one.

### 1.2 Prediction-Workflow Bottlenecks

Although the prediction code isn't shown, common issues with Random Forest inference on 6.46M rows × 110 features include:

- **Loading the model from disk repeatedly** or holding redundant copies in memory.
- **Predicting all 6.46M rows in a single call** can exceed RAM if the forest is large (many trees × deep nodes). Conversely, **row-by-row prediction** is catastrophically slow.
- **Object copying:** R's copy-on-modify semantics mean that adding columns to a large data.frame (`cell_data$new_col <- ...`) can trigger full copies of the entire data.frame. With ~6.46M rows × 110+ columns, each copy is ~5–7 GB.
- **Data type overhead:** If `cell_data` is a `data.frame` rather than a `data.table`, column additions and subsetting are much slower.

### 1.3 Summary of Root Causes

| Bottleneck | Estimated Impact |
|---|---|
| `build_neighbor_lookup`: 6.46M string-paste + named-vector lookups | ~hours |
| `compute_neighbor_stats`: 6.46M `lapply` iterations × 5 vars, plus `do.call(rbind, ...)` | ~hours |
| Copy-on-modify from repeated `cell_data$new_col <-` assignments | ~tens of minutes per copy event |
| RF prediction on 6.46M rows (if done naively) | ~minutes to hours depending on forest size |

---

## 2. OPTIMIZATION STRATEGY

### 2.1 Feature Preparation — Vectorize via `data.table` and Sparse Matrix Multiplication

The entire neighbor-stats computation (max, min, mean of neighbor values) can be recast as **sparse-matrix operations** or **`data.table` grouped joins**, eliminating all `lapply` loops.

**Key insight:** Build a sparse adjacency matrix `W` of dimension (6.46M × 6.46M) where `W[i,j] = 1` if row `j` is a rook neighbor of row `i` in the same year. Then:
- Neighbor mean = `(W %*% vals) / (W %*% ones)` (sparse matrix–vector multiply)
- Neighbor max/min require a grouped approach (sparse matrix multiply doesn't directly give max/min), but can be done efficiently with `data.table` joins.

**Practical approach chosen:** Instead of a full 6.46M × 6.46M sparse matrix, we build an **edge list** (from-row, to-row) and use `data.table` grouped aggregation. This is memory-efficient and fast.

### 2.2 Eliminate Object Copying

Convert `cell_data` to a `data.table` and use `:=` (modify-in-place) for all column additions. This eliminates multi-GB copies.

### 2.3 RF Prediction — Batched Prediction

- Load the model **once**.
- Predict in **batches** (e.g., 500K rows) to control peak memory.
- Ensure the prediction input is a clean matrix or data.frame with only the required columns (no extra columns that waste memory).

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================

library(data.table)
library(Matrix)
library(randomForest) # or ranger — adjust predict() call accordingly

# ---- 0. Convert to data.table (in-place, no copy) --------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# ---- 1. Build neighbor edge list (vectorized, replaces build_neighbor_lookup)
build_neighbor_edgelist <- function(dt, id_order, neighbors) {
  # Map each cell id to its position in id_order
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Build a data.table of (from_id, to_id) from the nb object
  # neighbors[[i]] gives the neighbor indices in id_order for id_order[i]
  from_ref <- rep(seq_along(neighbors), lengths(neighbors))
  to_ref   <- unlist(neighbors, use.names = FALSE)

  # Remove 0-entries (spdep uses 0 for "no neighbors")
  valid <- to_ref > 0L
  from_ref <- from_ref[valid]
  to_ref   <- to_ref[valid]

  edge_cells <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )

  # Now cross-join with years present in the data to get (from_row, to_row)
  # First, create a row-index lookup: (id, year) -> row index
  dt[, .row_idx := .I]

  # Merge edges with row indices for each year
  # For "from" side:
  lookup <- dt[, .(id, year, .row_idx)]

  # from side
  setnames(lookup, c("id", "year", ".row_idx"), c("from_id", "year", "from_row"))
  edges <- merge(edge_cells, lookup, by = "from_id", allow.cartesian = TRUE)

  # to side
  lookup_to <- dt[, .(to_id = id, year, to_row = .row_idx)]
  edges <- merge(edges, lookup_to, by = c("to_id", "year"))

  # Return only the integer row indices
  edges[, .(from_row = as.integer(from_row), to_row = as.integer(to_row))]
}

cat("Building neighbor edge list...\n")
system.time({
  edge_dt <- build_neighbor_edgelist(cell_data, id_order, rook_neighbors_unique)
})
cat(sprintf("Edge list: %s edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- 2. Compute neighbor stats efficiently (replaces compute_neighbor_stats)
compute_and_add_all_neighbor_features <- function(dt, edge_dt, var_names) {
  for (var_name in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", var_name))

    # Extract the variable values for the "to" (neighbor) rows
    edge_dt[, val := dt[[var_name]][to_row]]

    # Grouped aggregation by from_row
    stats <- edge_dt[!is.na(val),
      .(
        nb_max  = max(val),
        nb_min  = min(val),
        nb_mean = mean(val)
      ),
      by = from_row
    ]

    # Initialize new columns with NA
    max_col  <- paste0(var_name, "_nb_max")
    min_col  <- paste0(var_name, "_nb_min")
    mean_col <- paste0(var_name, "_nb_mean")

    dt[, (max_col)  := NA_real_]
    dt[, (min_col)  := NA_real_]
    dt[, (mean_col) := NA_real_]

    # Assign in-place using row indices
    set(dt, i = stats$from_row, j = max_col,  value = stats$nb_max)
    set(dt, i = stats$from_row, j = min_col,  value = stats$nb_min)
    set(dt, i = stats$from_row, j = mean_col, value = stats$nb_mean)
  }

  # Clean up temporary column in edge_dt
  edge_dt[, val := NULL]

  invisible(dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_and_add_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# Clean up the temporary row-index column
cell_data[, .row_idx := NULL]

# ---- 3. Random Forest Prediction (batched, memory-efficient) ----------------

# Load model ONCE
cat("Loading RF model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")

# Identify the exact predictor columns the model expects
# For randomForest:
if (inherits(rf_model, "randomForest")) {
  pred_vars <- rownames(rf_model$importance)
} else if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else {
  stop("Unsupported model class. Adjust predictor extraction accordingly.")
}

# Verify all required predictors are present
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop(paste("Missing predictor columns:", paste(missing_vars, collapse = ", ")))
}

# Batched prediction to control memory
predict_batched <- function(model, dt, pred_vars, batch_size = 500000L) {
  n <- nrow(dt)
  predictions <- numeric(n)

  n_batches <- ceiling(n / batch_size)
  cat(sprintf("Predicting %s rows in %d batches of up to %s...\n",
              format(n, big.mark = ","), n_batches,
              format(batch_size, big.mark = ",")))

  for (b in seq_len(n_batches)) {
    start_idx <- (b - 1L) * batch_size + 1L
    end_idx   <- min(b * batch_size, n)

    # Extract only the needed columns for this batch (minimizes memory)
    batch_data <- dt[start_idx:end_idx, ..pred_vars]

    if (inherits(model, "ranger")) {
      preds <- predict(model, data = batch_data)$predictions
    } else {
      # randomForest
      preds <- predict(model, newdata = batch_data)
    }

    predictions[start_idx:end_idx] <- preds

    if (b %% 5 == 0 || b == n_batches) {
      cat(sprintf("  Batch %d/%d complete (rows %s-%s)\n",
                  b, n_batches,
                  format(start_idx, big.mark = ","),
                  format(end_idx, big.mark = ",")))
    }

    # Free batch memory
    rm(batch_data, preds)
    if (b %% 10 == 0) gc(verbose = FALSE)
  }

  predictions
}

cat("Running predictions...\n")
system.time({
  cell_data[, predicted_gdp := predict_batched(rf_model, cell_data, pred_vars)]
})

cat("Done.\n")
```

---

## 4. EXPECTED PERFORMANCE GAINS

| Component | Original | Optimized | Speedup |
|---|---|---|---|
| `build_neighbor_lookup` (6.46M `lapply` + string ops) | ~10–30 hours | ~2–5 min (vectorized merge) | **~200–400×** |
| `compute_neighbor_stats` (5 vars × 6.46M `lapply` + `do.call(rbind)`) | ~40–50 hours | ~1–3 min (`data.table` grouped agg) | **~1000×** |
| Column assignment (copy-on-modify) | ~5–10 GB copies × multiple times | In-place via `:=` / `set()` | **eliminates copies** |
| RF prediction | Varies | Batched, single model load | **controlled memory, same speed** |
| **Total estimated** | **86+ hours** | **~10–30 minutes** | **~200×** |

### Why This Preserves Correctness

- **Same numerical estimand:** The `max`, `min`, and `mean` computations are identical — just computed via `data.table` grouped aggregation instead of `lapply`. The values are the same floating-point results.
- **Trained RF model is untouched:** It is loaded from disk read-only and used only for `predict()`. No retraining occurs.
- **Edge list faithfully reproduces the neighbor lookup:** Every `(from_row, to_row)` pair corresponds to the same rook-neighbor relationship in the same year as the original `build_neighbor_lookup` produced.