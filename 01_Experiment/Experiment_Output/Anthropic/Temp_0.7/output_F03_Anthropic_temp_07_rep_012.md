 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first major bottleneck:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering — all inside an interpreted R loop. |
| **Repeated `paste` key construction** | `paste(neighbor_cell_ids, data$year[i], sep="_")` is called ~6.46M times, each producing a small character vector that must be matched against a named vector of 6.46M elements. Named-vector lookup (`idx_lookup[neighbor_keys]`) is O(n) hashing per call, repeated millions of times. |
| **Character-based join logic** | The entire neighbor resolution is string-based (`paste` → named lookup). This is orders of magnitude slower than integer-indexed joins. |

**`compute_neighbor_stats`** is the second major bottleneck:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements** | Each call extracts a small integer vector, subsets a numeric vector, removes NAs, and computes three summary statistics — all in interpreted R. |
| **`do.call(rbind, result)` on 6.46M small vectors** | This builds a list of 6.46M length-3 vectors and then row-binds them. This is a classic R anti-pattern that is extremely slow and memory-hungry. |
| **Called 5 times** | The outer loop repeats this for each of the 5 neighbor source variables. |

**Memory pressure**: With 6.46M rows × 110 columns, the data frame alone is ~5–6 GB. The neighbor lookup list (6.46M elements, each a small integer vector) adds another ~1–2 GB. Repeated `do.call(rbind, ...)` on lists of millions of elements creates massive transient allocations, likely triggering garbage collection storms and possible swap on a 16 GB machine.

### B. Random Forest Inference Bottlenecks

| Problem | Detail |
|---|---|
| **Single-call `predict()` on 6.46M rows** | Depending on the RF implementation (`randomForest`, `ranger`, etc.), predicting 6.46M rows with 110 features and hundreds of trees can exhaust RAM (the `randomForest` package is particularly memory-hungry at prediction time). |
| **Model loading** | If the serialized model is large (hundreds of MB), `readRDS()` is a one-time cost but can be significant. |
| **Data frame copying** | R's copy-on-modify semantics mean that adding columns to `cell_data` inside a loop (`cell_data <- compute_and_add_neighbor_features(...)`) may trigger full copies of the 5–6 GB frame on each iteration. |

### C. Overall Runtime Decomposition (estimated)

| Phase | Estimated share of 86+ hrs |
|---|---|
| `build_neighbor_lookup` | ~30–40% |
| `compute_neighbor_stats` (×5 vars) | ~30–40% |
| RF `predict()` | ~10–20% |
| Data I/O, model load, overhead | ~5–10% |

---

## 2. Optimization Strategy

### Principle: Replace interpreted R loops and string operations with vectorized `data.table` joins and grouped aggregations.

| Current | Optimized |
|---|---|
| `build_neighbor_lookup`: `lapply` over 6.46M rows, `paste`-based named lookup | **Eliminate entirely.** Build an edge-list `data.table` of (id, year, neighbor_id) and merge directly with the data on (neighbor_id, year) — no per-row loop, no string keys. |
| `compute_neighbor_stats`: `lapply` over 6.46M rows, `do.call(rbind, ...)` | **Replace with `data.table` grouped aggregation** on the edge-list: `edges[data, on=...][, .(max, min, mean), by=.(id, year)]`. One vectorized pass per variable. |
| `cell_data <- ...` in loop (copy-on-modify) | **Use `data.table` set-by-reference** (`:=`), zero copies. |
| RF `predict()` on full 6.46M rows | **Batch prediction** in chunks (~500K rows) to control peak RAM; use `ranger::predict` if possible (much faster and more memory-efficient than `randomForest::predict`). |

**Expected speedup**: From 86+ hours to approximately **15–45 minutes** depending on hardware, with peak RAM kept under 16 GB.

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger"))
#   (ranger is recommended; randomForest fallback is supported)

library(data.table)

# ---- Configuration ----------------------------------------------------------
BATCH_SIZE <- 500000L # rows per RF prediction batch (tune to RAM)

# ---- Step 0: Load pre-trained model and data --------------------------------
# Assumes: rf_model  <- readRDS("path/to/trained_model.rds")
#          cell_data <- as a data.frame / data.table with columns: id, year, ...
#          id_order  <- integer vector of cell IDs in the order used by spdep::nb
#          rook_neighbors_unique <- spdep::nb object (list of integer index vectors)

# Convert to data.table in-place (no copy if already data.table)
setDT(cell_data)

# ---- Step 1: Build edge list (replaces build_neighbor_lookup) ---------------
# This is fully vectorized — no per-row loop.

build_edge_list <- function(id_order, nb_object) {
  # nb_object[[i]] contains integer indices into id_order for the neighbors
  # of id_order[i]. Index 0 means "no neighbors" in spdep convention.

  n <- length(nb_object)

  # Number of neighbors per cell
  n_neighbors <- vapply(nb_object, function(x) {
    sum(x > 0L)
  }, integer(1))

  total_edges <- sum(n_neighbors)

  # Pre-allocate vectors
  from_id <- integer(total_edges)
  to_id   <- integer(total_edges)

  pos <- 1L
  for (i in seq_len(n)) {
    nb <- nb_object[[i]]
    nb <- nb[nb > 0L]
    len <- length(nb)
    if (len > 0L) {
      idx <- pos:(pos + len - 1L)
      from_id[idx] <- id_order[i]
      to_id[idx]   <- id_order[nb]
      pos <- pos + len
    }
  }

  data.table(id = from_id, neighbor_id = to_id)
}

cat("Building edge list...\n")
edge_dt <- build_edge_list(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s directed edges\n", format(nrow(edge_dt), big.mark = ",")))

# ---- Step 2: Compute neighbor features (replaces compute_neighbor_stats) ----
# Fully vectorized data.table grouped aggregation. Adds columns by reference.

compute_and_add_neighbor_features_dt <- function(cell_dt, edge_dt, var_name) {
  # Subset the data to just the join keys and the variable of interest
  # to minimize the memory footprint of the join.
  lookup <- cell_dt[, .(neighbor_id = id, year, value = get(var_name))]

  # Join edges with the lookup to get neighbor values
  # edge_dt has (id, neighbor_id); we add year from cell_dt for the focal cell,

  # then look up the neighbor's value in that same year.

  # First, add year to edges by joining with focal cell's (id, year)
  focal_keys <- cell_dt[, .(id, year)]

  # Cross edges with all years for each focal cell
  # edge_dt has unique (id, neighbor_id) pairs (spatial, time-invariant)
  # focal_keys has (id, year) — one row per cell-year
  # We need (id, year, neighbor_id) for every cell-year and its neighbors.

  edges_with_year <- edge_dt[focal_keys, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # Result: (id, neighbor_id, year)

  # Now join to get the neighbor's value in that year
  edges_with_year[lookup, on = .(neighbor_id, year), neighbor_val := i.value]

  # Compute grouped statistics
  stats <- edges_with_year[
    !is.na(neighbor_val),
    .(
      nb_max  = max(neighbor_val),
      nb_min  = min(neighbor_val),
      nb_mean = mean(neighbor_val)
    ),
    by = .(id, year)
  ]

  # Rename columns to match original naming convention
  max_col  <- paste0("nb_max_", var_name)
  min_col  <- paste0("nb_min_", var_name)
  mean_col <- paste0("nb_mean_", var_name)
  setnames(stats, c("nb_max", "nb_min", "nb_mean"), c(max_col, min_col, mean_col))

  # Remove old columns if they exist (in case of re-run)
  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Merge back by reference
  cell_dt[stats, on = .(id, year), (c(max_col, min_col, mean_col)) := mget(c(
    paste0("i.", max_col), paste0("i.", min_col), paste0("i.", mean_col)
  ))]

  # Clean up large intermediate objects
  rm(edges_with_year, stats, lookup)
  gc()

  invisible(cell_dt)
}

# ---- Step 2b: Memory-optimized variant for tight RAM situations -------------
# Processes one variable at a time but streams the year-expansion in chunks.

compute_and_add_neighbor_features_lowmem <- function(cell_dt, edge_dt, var_name) {
  max_col  <- paste0("nb_max_",  var_name)
  min_col  <- paste0("nb_min_",  var_name)
  mean_col <- paste0("nb_mean_", var_name)

  # Remove old columns if they exist

  for (col in c(max_col, min_col, mean_col)) {
    if (col %in% names(cell_dt)) cell_dt[, (col) := NULL]
  }

  # Initialize result columns with NA
  cell_dt[, (max_col)  := NA_real_]
  cell_dt[, (min_col)  := NA_real_]
  cell_dt[, (mean_col) := NA_real_]

  years <- sort(unique(cell_dt$year))

  # Create a keyed version for fast subsetting
  setkey(cell_dt, year)

  for (yr in years) {
    # Subset to this year
    yr_data <- cell_dt[.(yr), .(id, value = get(var_name))]

    # Join edges with this year's values for the neighbor
    yr_edges <- edge_dt[yr_data, on = .(neighbor_id = id), nomatch = NULL, allow.cartesian = FALSE]
    # yr_edges now has: id (focal), neighbor_id, value (neighbor's value)

    # Aggregate
    yr_stats <- yr_edges[
      !is.na(value),
      .(nb_max = max(value), nb_min = min(value), nb_mean = mean(value)),
      by = .(id)
    ]
    yr_stats[, year := yr]

    # Update cell_dt by reference
    cell_dt[yr_stats, on = .(id, year),
            `:=`(
              (max_col)  = i.nb_max,
              (min_col)  = i.nb_min,
              (mean_col) = i.nb_mean
            )]
  }

  setkey(cell_dt, NULL) # remove key
  gc()
  invisible(cell_dt)
}

# ---- Step 3: Run neighbor feature computation --------------------------------

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))
  t0 <- proc.time()

  # Use the low-memory variant on a 16 GB laptop
  compute_and_add_neighbor_features_lowmem(cell_dt = cell_data, edge_dt = edge_dt, var_name = var_name)

  elapsed <- (proc.time() - t0)[["elapsed"]]
  cat(sprintf("    Done in %.1f seconds\n", elapsed))
}

rm(edge_dt)
gc()

# ---- Step 4: Random Forest prediction in batches ----------------------------

cat("Running Random Forest prediction...\n")

# Determine which predict function to use
is_ranger <- inherits(rf_model, "ranger")

# Get the feature names the model expects
if (is_ranger) {
  feature_names <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used during training
  feature_names <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features are present
missing_feats <- setdiff(feature_names, names(cell_data))
if (length(missing_feats) > 0) {
  stop("Missing features in cell_data: ", paste(missing_feats, collapse = ", "))
}

# Prepare prediction input (only needed columns, as a data.frame for predict())
pred_input <- as.data.frame(cell_data[, ..feature_names])

n_rows <- nrow(pred_input)
n_batches <- ceiling(n_rows / BATCH_SIZE)

# Pre-allocate prediction vector
predictions <- numeric(n_rows)

cat(sprintf("  Total rows: %s | Batch size: %s | Batches: %d\n",
            format(n_rows, big.mark = ","),
            format(BATCH_SIZE, big.mark = ","),
            n_batches))

for (b in seq_len(n_batches)) {
  start_idx <- (b - 1L) * BATCH_SIZE + 1L
  end_idx   <- min(b * BATCH_SIZE, n_rows)

  batch_data <- pred_input[start_idx:end_idx, , drop = FALSE]

  if (is_ranger) {
    batch_pred <- predict(rf_model, data = batch_data)$predictions
  } else {
    batch_pred <- predict(rf_model, newdata = batch_data)
  }

  predictions[start_idx:end_idx] <- batch_pred

  if (b %% 5 == 0 || b == n_batches) {
    cat(sprintf("    Batch %d/%d complete (rows %s-%s)\n",
                b, n_batches,
                format(start_idx, big.mark = ","),
                format(end_idx, big.mark = ",")))
  }
}

rm(pred_input, batch_data, batch_pred)
gc()

# ---- Step 5: Attach predictions to cell_data --------------------------------
cell_data[, predicted_gdp := predictions]

cat("Pipeline complete.\n")
cat(sprintf("  Predictions: %s values, range [%.4f, %.4f]\n",
            format(length(predictions), big.mark = ","),
            min(predictions, na.rm = TRUE),
            max(predictions, na.rm = TRUE)))
```

---

## Summary of Changes and Expected Impact

| Component | Original | Optimized | Expected Speedup |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with `paste`/named-vector lookup | Vectorized `data.table` edge-list built once (~1.37M edges) | **~200–500×** |
| **Neighbor stats** | `lapply` + `do.call(rbind, ...)` per variable × 5 | `data.table` grouped join + aggregation per year per variable | **~100–300×** |
| **Data mutation** | `cell_data <- ...` (copy-on-modify, ~5 GB copies) | `:=` set-by-reference (zero copies) | **Eliminates ~25–30 GB transient allocation** |
| **RF prediction** | Presumably single `predict()` call on 6.46M rows | Batched (500K rows/batch), supports `ranger` and `randomForest` | **Controls peak RAM; ~2–5× faster with `ranger`** |
| **Overall runtime** | **86+ hours** | **~15–45 minutes** (feature prep ~5–15 min, RF predict ~10–30 min) | **~100–300×** |

The trained Random Forest model is loaded and used as-is (`predict()` only), and the numerical estimand (predicted GDP) is preserved identically — no retraining, no transformation of the output.