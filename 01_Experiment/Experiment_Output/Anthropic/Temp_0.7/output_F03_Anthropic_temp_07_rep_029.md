 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### Primary Bottlenecks (in order of severity)

**A. `build_neighbor_lookup` — O(n²)-scale character hashing and per-row `lapply`**

With ~6.46 million rows, the function:
- Creates a named character vector `idx_lookup` of length 6.46M keyed by `paste(id, year)` strings.
- Then calls `lapply` over every row (6.46M iterations), each time doing `paste()`, character name-matching into `idx_lookup`, and subsetting. R named-vector lookup is O(n) in the worst case per probe because R rehashes internally, and the sheer number of string allocations (`paste` on vectors inside a per-row lambda) is extremely expensive.
- **Estimated cost:** billions of character comparisons; this alone can take tens of hours.

**B. `compute_neighbor_stats` — repeated per-variable full-data scans**

Called 5 times (once per neighbor source variable), each time doing `lapply` over 6.46M rows, extracting neighbor values by index, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also very slow (repeated memory allocation).

**C. Excessive object copying (`cell_data` mutation in a loop)**

Each `compute_and_add_neighbor_features` call likely adds columns to `cell_data` (a data.frame). In base R, adding columns to a data.frame triggers a full copy (~6.46M × 110+ columns). Over 5 iterations this means ~5 full copies of a multi-GB object.

**D. Random Forest prediction on 6.46M rows × 110 features**

`predict.randomForest` (or `predict.ranger`) on 6.46M rows is memory-intensive. If using the `randomForest` package, it stores predictions in a dense matrix and can be slow. If the model is large (many trees, deep), prediction alone can take hours and may exceed 16 GB RAM if done naively.

**E. Model loading**

If the serialized RF model is large (>1 GB), `readRDS` time is non-trivial but is a one-time cost and not the main issue.

---

## 2. Optimization Strategy

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| `build_neighbor_lookup` | Replace character-key lookup with integer-key lookup using `data.table` keyed joins. Build a single `(id, year) → row_index` hash table via `data.table`, then vectorized join. | **100–500×** |
| `compute_neighbor_stats` | Pre-build a flat edge-list (CSR-style) of `(row_i, neighbor_row_j)`, then use `data.table` grouped aggregation or vectorized C-level operations to compute all stats in one pass per variable. | **50–200×** |
| Object copying | Use `data.table` with `:=` (in-place column addition). Zero copies. | **5–10×** |
| RF prediction | Use `ranger` (C++ backend) if possible; if model is `randomForest`-class, convert or predict in chunks to control peak memory. Predict in batches of ~500K rows. | **2–10×** |
| Overall | Eliminate all per-row `lapply` in R; everything becomes vectorized or `data.table`-grouped. | **~86h → <1h** |

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE — preserves trained RF model and numerical estimand
# =============================================================================

library(data.table)

# ---- 0. Convert cell_data to data.table (once) -----------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# ---- 1. FAST NEIGHBOR LOOKUP: build a flat edge-list -----------------------
#
# Instead of a list-of-vectors (one per row), we build a two-column data.table
# (row_i, neighbor_row_j) that can be joined and aggregated vectorially.

build_neighbor_edgelist_dt <- function(dt, id_order, neighbors_nb) {
  # Map cell id -> position in id_order (integer)
  id_to_ref <- setNames(seq_along(id_order), as.character(id_order))

  # Map (id, year) -> row index via data.table keyed join
  dt[, .row_idx := .I]
  idx_dt <- dt[, .(id, year, .row_idx)]
  setkey(idx_dt, id, year)

  # Unique cell ids in the data
  unique_ids_in_data <- unique(dt$id)

  # Build the edge list: for each cell id, get its neighbor cell ids

  # Then cross with all years that cell appears in.
  # This is the key insight: neighbor relationships are spatial (id-level),
  # but we need them at the (id, year) level.

  # Step 1: Build spatial edge list (cell_id -> neighbor_cell_id)
  message("Building spatial edge list...")
  edge_list <- rbindlist(lapply(seq_along(id_order), function(ref_idx) {
    nb_indices <- neighbors_nb[[ref_idx]]
    if (length(nb_indices) == 0L) return(NULL)
    data.table(
      id = id_order[ref_idx],
      neighbor_id = id_order[nb_indices]
    )
  }))

  if (nrow(edge_list) == 0L) {
    warning("No neighbor edges found.")
    return(data.table(row_i = integer(0), neighbor_row_j = integer(0)))
  }

  # Step 2: Expand to (id, year, neighbor_id) by joining with years per cell
  message("Expanding to cell-year level...")
  years_per_id <- dt[, .(year = unique(year)), by = id]
  setkey(years_per_id, id)
  setkey(edge_list, id)

  # Join: each spatial edge gets all years of the focal cell
  edge_year <- edge_list[years_per_id, on = "id", allow.cartesian = TRUE, nomatch = 0L]
  # edge_year now has columns: id, neighbor_id, year

  # Step 3: Map focal (id, year) -> row_i
  edge_year_merged <- merge(
    edge_year,
    idx_dt,
    by = c("id", "year"),
    all.x = TRUE
  )
  setnames(edge_year_merged, ".row_idx", "row_i")

  # Step 4: Map neighbor (neighbor_id, year) -> neighbor_row_j
  neighbor_idx <- idx_dt[, .(neighbor_id = id, year, neighbor_row_j = .row_idx)]
  setkey(neighbor_idx, neighbor_id, year)

  edge_final <- merge(
    edge_year_merged,
    neighbor_idx,
    by = c("neighbor_id", "year"),
    all.x = FALSE  # inner join: drop neighbors not present in that year
  )

  # Clean up helper column
  dt[, .row_idx := NULL]

  message(sprintf("Edge list: %s edges for %s cell-year rows.",
                  format(nrow(edge_final), big.mark = ","),
                  format(nrow(dt), big.mark = ",")))

  edge_final[, .(row_i, neighbor_row_j)]
}

# ---- 2. FAST NEIGHBOR STATS: vectorized grouped aggregation -----------------

compute_neighbor_stats_dt <- function(dt, edge_dt, var_name) {
  # Extract the variable values for all neighbor rows
  vals <- dt[[var_name]]

  # Build a working table: row_i + neighbor value
  work <- edge_dt[, .(row_i, nval = vals[neighbor_row_j])]

  # Remove NAs

  work <- work[!is.na(nval)]

  # Grouped aggregation (vectorized C-level in data.table)
  agg <- work[, .(
    nb_max  = max(nval),
    nb_min  = min(nval),
    nb_mean = mean(nval)
  ), by = row_i]

  # Create full-length result aligned to all rows
  n <- nrow(dt)
  out_max  <- rep(NA_real_, n)
  out_min  <- rep(NA_real_, n)
  out_mean <- rep(NA_real_, n)

  out_max[agg$row_i]  <- agg$nb_max
  out_min[agg$row_i]  <- agg$nb_min
  out_mean[agg$row_i] <- agg$nb_mean

  # Add columns in-place (no copy)
  set(dt, j = paste0("nb_max_",  var_name), value = out_max)
  set(dt, j = paste0("nb_min_",  var_name), value = out_min)
  set(dt, j = paste0("nb_mean_", var_name), value = out_mean)

  invisible(NULL)
}

# ---- 3. MAIN PIPELINE ------------------------------------------------------

message("=== Step 1: Build neighbor edge list ===")
edge_dt <- build_neighbor_edgelist_dt(cell_data, id_order, rook_neighbors_unique)

message("=== Step 2: Compute neighbor features ===")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

for (var_name in neighbor_source_vars) {
  message(sprintf("  Processing: %s", var_name))
  compute_neighbor_stats_dt(cell_data, edge_dt, var_name)
}

# Free the edge list after feature engineering
rm(edge_dt); gc()

message("=== Step 3: Load trained Random Forest model ===")
# Adjust path as needed
rf_model <- readRDS("trained_rf_model.rds")

message("=== Step 4: Predict in batches ===")

# Determine feature columns (must match training)
# Assumes the model knows its feature names; adjust as needed.
if (inherits(rf_model, "ranger")) {
  feature_cols <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores the original call or we can inspect
  # Use the names from the training data or a saved vector
  if (exists("rf_feature_names")) {
    feature_cols <- rf_feature_names
  } else {
    # Fallback: try to extract from the model
    feature_cols <- rownames(rf_model$importance)
  }
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all features exist
missing_cols <- setdiff(feature_cols, names(cell_data))
if (length(missing_cols) > 0) {
  stop("Missing feature columns: ", paste(missing_cols, collapse = ", "))
}

# Batch prediction to control peak memory (~500K rows per batch)
batch_size <- 500000L
n_rows     <- nrow(cell_data)
n_batches  <- ceiling(n_rows / batch_size)
preds      <- numeric(n_rows)

message(sprintf("Predicting %s rows in %d batches of up to %s...",
                format(n_rows, big.mark = ","), n_batches,
                format(batch_size, big.mark = ",")))

for (b in seq_len(n_batches)) {
  idx_start <- (b - 1L) * batch_size + 1L
  idx_end   <- min(b * batch_size, n_rows)
  batch_idx <- idx_start:idx_end

  # Extract batch as data.frame (RF predict methods expect data.frame)
  batch_df <- as.data.frame(cell_data[batch_idx, ..feature_cols])

  if (inherits(rf_model, "ranger")) {
    preds[batch_idx] <- predict(rf_model, data = batch_df, num.threads = 4L)$predictions
  } else {
    preds[batch_idx] <- predict(rf_model, newdata = batch_df)
  }

  if (b %% 2 == 0 || b == n_batches) {
    message(sprintf("  Batch %d/%d done (rows %s–%s)",
                    b, n_batches,
                    format(idx_start, big.mark = ","),
                    format(idx_end, big.mark = ",")))
  }
  rm(batch_df); gc(verbose = FALSE)
}

# Store predictions in-place
set(cell_data, j = "predicted_gdp", value = preds)
rm(preds); gc()

message("=== Pipeline complete ===")
```

---

## 4. Summary of Changes and Why They Work

| Original Problem | Fix Applied | Mechanism |
|---|---|---|
| `paste(id, year)` string keys in a 6.46M named vector, probed per-row in `lapply` | `data.table` keyed integer join to build a flat edge-list once | Eliminates ~6.46M × `paste` calls and O(n) named-vector lookups; `data.table` uses radix-based hashing at C level |
| Per-row `lapply` in `compute_neighbor_stats` (6.46M × 5 vars = 32.3M R function calls) | Single `data.table` grouped `[, .(max, min, mean), by = row_i]` per variable | Vectorized C-level grouping; no R-level loop |
| `do.call(rbind, list_of_6.46M_vectors)` | Eliminated entirely; results written via `set()` | Avoids constructing and binding a 6.46M-element list |
| `cell_data` copied 5× when adding columns to a `data.frame` in a loop | `data.table::set()` adds columns **in-place** with zero copies | Saves ~5 × multi-GB copy operations |
| Full 6.46M-row prediction in one call (potential OOM on 16 GB) | Batched prediction (500K rows/batch) with `gc()` between batches | Peak memory stays well within 16 GB |
| `randomForest::predict` is single-threaded | If model is `ranger`, uses `num.threads = 4L`; if `randomForest`, batching still helps | `ranger` predict is parallelized in C++ |

**Estimated runtime:** Under 1 hour total on a 16 GB laptop (edge-list build ~5–10 min, neighbor stats ~5–10 min, RF prediction ~15–30 min depending on model size), down from the original 86+ hours. The trained model is never retrained, and all numerical outputs are preserved identically.