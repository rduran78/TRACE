 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys for every neighbor of every row
- Named-vector lookup into `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector)
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations × ~4 neighbors each ≈ 25.8M string-paste + hash-lookup operations on a 6.46M-entry named vector. This alone can take many hours.

**`compute_neighbor_stats`:** Calls `lapply` over 6.46M rows, subsetting a numeric vector, removing NAs, and computing max/min/mean. The `do.call(rbind, result)` at the end binds 6.46M 3-element vectors into a matrix — this is a known R anti-pattern that is extremely slow for millions of elements.

**Outer loop:** Calls `compute_neighbor_stats` 5 times (once per variable), so the 6.46M-row lapply + rbind penalty is paid 5 times. Additionally, `compute_and_add_neighbor_features` likely copies the entire `cell_data` data.frame on each assignment (`cell_data <- ...`), which for ~6.46M × 110+ columns is a multi-GB copy each time.

### 1.2 Prediction-Workflow Bottlenecks

- **Model loading:** If the serialized Random Forest is large (hundreds of MB to several GB for 110 predictors on millions of rows), `readRDS` and deserialization is a one-time but significant cost.
- **Prediction in a loop:** If `predict()` is called row-by-row or in small batches rather than on the full matrix at once, overhead per call dominates.
- **Object copying:** R's copy-on-modify semantics mean that modifying `cell_data` inside a loop (adding columns) triggers full data.frame copies.
- **Memory pressure:** 6.46M rows × 110 columns × 8 bytes ≈ 5.7 GB just for the numeric matrix. The Random Forest object, neighbor lookup list, and intermediate copies can easily exceed 16 GB, causing swap thrashing.

### 1.3 Root-Cause Summary

| Bottleneck | Estimated Share | Cause |
|---|---|---|
| `build_neighbor_lookup` (string ops on 6.46M rows) | ~30% | `paste`/named-vector lookups in R loop |
| `compute_neighbor_stats` (lapply + do.call rbind) | ~25% | Per-row R-level loop, slow rbind |
| Data.frame copies in outer loop | ~15% | Copy-on-modify, 5 iterations |
| Prediction loop (if row/batch-wise) | ~20% | R-level predict overhead per call |
| Memory thrashing / GC | ~10% | >16 GB working set on 16 GB machine |

---

## 2. OPTIMIZATION STRATEGY

### Principle: Vectorize, use `data.table`, pre-build integer index matrices, compute all neighbor stats in one pass, predict in one call.

| Problem | Solution |
|---|---|
| String-key lookups in `build_neighbor_lookup` | Build an integer matrix mapping each row to its neighbor-row indices using `data.table` equi-joins — no strings, no named vectors |
| Per-row `lapply` in `compute_neighbor_stats` | Use the integer index matrix to do column-wise vectorized aggregation via `data.table` grouping or matrix indexing with `rowMeans`/`pmin`/`pmax` on a pre-extracted neighbor-value matrix |
| `do.call(rbind, ...)` on 6.46M elements | Pre-allocate matrix; or avoid entirely with vectorized path |
| Data.frame copy on each variable | Use `data.table` set-by-reference (`:=`) — zero copies |
| Prediction loop | Single `predict(model, newdata)` call on full matrix |
| Memory pressure | Convert to matrix for predict; drop intermediate objects; `gc()` strategically; process neighbor stats for all 5 variables in one pass over the index structure |

**Expected speedup:** From ~86+ hours to ~10–30 minutes.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE — Feature Preparation + Random Forest Prediction
# =============================================================================
# Requirements: data.table, ranger (or randomForest — works with both)
# The trained RF model object is assumed to be on disk as "rf_model.rds".
# cell_data is assumed to be a data.frame/data.table with columns:
#   id, year, ntl, ec, pop_density, def, usd_est_n2, ... (all predictors)
# id_order: integer vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)
# =============================================================================

library(data.table)

# ---- Step 0: Convert cell_data to data.table in place --------------------
setDT(cell_data)

# ---- Step 1: Build neighbor lookup as an integer-indexed edge list --------
#
# Goal: for every row in cell_data, find the row indices of its rook neighbors
# in the same year.  We avoid all string operations.
#
# Sub-step 1a: Map cell id -> position in id_order (integer vector, not named)

build_neighbor_edgelist <- function(id_order, neighbors) {
  # neighbors is an nb object: neighbors[[i]] gives integer indices into

  # id_order for the neighbors of id_order[i].
  # We expand this into a two-column data.table: (cell_id, neighbor_cell_id)
  n <- length(id_order)
  from <- rep.int(id_order, lengths(neighbors))
  to   <- id_order[unlist(neighbors, use.names = FALSE)]
  data.table(cell_id = from, neighbor_id = to)
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
cat(sprintf("  Edge list: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# Sub-step 1b: Add row-index to cell_data and join
#   We need to map (cell_id, year) -> row_index for both the focal cell and
#   its neighbors.

cell_data[, row_idx := .I]

# Create a keyed lookup: (id, year) -> row_idx
lookup <- cell_data[, .(cell_id = id, year, row_idx)]
setkey(lookup, cell_id, year)

# Sub-step 1c: For each edge (cell_id, neighbor_id), cross with all years
#   of the focal cell, then find the neighbor's row in the same year.
#
#   But expanding 1.37M edges × 28 years = 38.4M rows is feasible and fast.
#
#   More memory-efficient: join edges onto the focal cell's (id, year, row_idx),
#   then join the neighbor side.

cat("Joining edges with years...\n")

# Focal side: every (cell_id, year, focal_row_idx) that exists in the data
focal <- cell_data[, .(cell_id = id, year, focal_row = row_idx)]
setkey(focal, cell_id)

# Join: for each focal row, attach its neighbor cell IDs
# edge_dt is keyed on cell_id
setkey(edge_dt, cell_id)
expanded <- edge_dt[focal, on = "cell_id", allow.cartesian = TRUE, nomatch = 0L]
# expanded now has columns: cell_id, neighbor_id, year, focal_row

# Now find the neighbor's row index in the same year
setkey(expanded, neighbor_id, year)
setkey(lookup, cell_id, year)
expanded <- lookup[expanded, on = c(cell_id = "neighbor_id", "year"), nomatch = 0L]
# After this join, 'row_idx' is the neighbor's row index, 'focal_row' is the focal row index

neighbor_map <- expanded[, .(focal_row, neighbor_row = row_idx)]

# Clean up large intermediates
rm(focal, expanded, lookup, edge_dt)
gc()

cat(sprintf("  Neighbor map: %s pairs\n", format(nrow(neighbor_map), big.mark = ",")))

# ---- Step 2: Compute neighbor stats (max, min, mean) for all variables ----
#
# Strategy: group neighbor_map by focal_row, extract neighbor values from the
# column, and compute stats — all vectorized inside data.table.

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor statistics...\n")

for (var_name in neighbor_source_vars) {
  cat(sprintf("  Processing: %s\n", var_name))

  # Extract the variable as a plain numeric vector (fast column access)
  vals <- cell_data[[var_name]]

  # Attach neighbor values to the map
  neighbor_map[, nval := vals[neighbor_row]]

  # Compute grouped stats — this is the core vectorized aggregation
  stats <- neighbor_map[!is.na(nval),
    .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ),
    by = focal_row
  ]

  # Assign back to cell_data by reference (zero-copy)
  max_col  <- paste0("neighbor_max_",  var_name)
  min_col  <- paste0("neighbor_min_",  var_name)
  mean_col <- paste0("neighbor_mean_", var_name)

  # Initialize with NA
  set(cell_data, j = max_col,  value = NA_real_)
  set(cell_data, j = min_col,  value = NA_real_)
  set(cell_data, j = mean_col, value = NA_real_)

  # Fill in computed values
  set(cell_data, i = stats$focal_row, j = max_col,  value = stats$nb_max)
  set(cell_data, i = stats$focal_row, j = min_col,  value = stats$nb_min)
  set(cell_data, i = stats$focal_row, j = mean_col, value = stats$nb_mean)

  rm(stats)
}

# Clean up the temporary column
neighbor_map[, nval := NULL]

cat("Neighbor features complete.\n")
gc()

# ---- Step 3: Load the trained Random Forest model ------------------------

cat("Loading trained Random Forest model...\n")
rf_model <- readRDS("rf_model.rds")

# ---- Step 4: Predict in a single vectorized call -------------------------
#
# Identify the predictor columns the model expects.
# For ranger:
if (inherits(rf_model, "ranger")) {
  pred_vars <- rf_model$forest$independent.variable.names
} else if (inherits(rf_model, "randomForest")) {
  # randomForest stores variable names used during training
  pred_vars <- rownames(rf_model$importance)
} else {
  stop("Unsupported model class: ", class(rf_model)[1])
}

# Verify all predictor columns exist
missing_vars <- setdiff(pred_vars, names(cell_data))
if (length(missing_vars) > 0) {
  stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
}

# Extract predictor matrix — convert only needed columns to avoid memory bloat
cat("Preparing prediction matrix...\n")
pred_data <- cell_data[, ..pred_vars]

# Predict — single call, no loop
cat("Running prediction on all rows...\n")

if (inherits(rf_model, "ranger")) {
  # ranger::predict returns a list; predictions in $predictions
  pred_result <- predict(rf_model, data = pred_data)
  cell_data[, predicted_gdp := pred_result$predictions]
} else {
  # randomForest::predict returns a vector directly
  cell_data[, predicted_gdp := predict(rf_model, newdata = pred_data)]
}

rm(pred_data)
gc()

cat("Prediction complete.\n")

# ---- Step 5 (optional): If memory is tight, batch prediction -------------
#
# If the single predict() call causes an out-of-memory error on a 16 GB
# laptop, use the following batched version instead of Step 4's predict block:

run_batched_prediction <- function(model, dt, pred_vars, batch_size = 500000L) {
  n <- nrow(dt)
  preds <- numeric(n)
  starts <- seq(1L, n, by = batch_size)

  for (k in seq_along(starts)) {
    i1 <- starts[k]
    i2 <- min(i1 + batch_size - 1L, n)
    cat(sprintf("  Batch %d/%d: rows %s–%s\n", k, length(starts),
                format(i1, big.mark = ","), format(i2, big.mark = ",")))

    batch <- dt[i1:i2, ..pred_vars]

    if (inherits(model, "ranger")) {
      preds[i1:i2] <- predict(model, data = batch)$predictions
    } else {
      preds[i1:i2] <- predict(model, newdata = batch)
    }

    rm(batch)
    if (k %% 3 == 0) gc()
  }
  preds
}

# Uncomment the following to use batched prediction instead:
# cell_data[, predicted_gdp := run_batched_prediction(rf_model, cell_data, pred_vars)]

# ---- Step 6: Output / save -----------------------------------------------
cat(sprintf("Rows predicted: %s\n", format(sum(!is.na(cell_data$predicted_gdp)), big.mark = ",")))

# Save results
# fwrite(cell_data, "cell_data_with_predictions.csv")
# or
# saveRDS(cell_data, "cell_data_with_predictions.rds")
```

---

## 4. WHY THIS IS FASTER — ITEMIZED

| Original | Optimized | Speedup Factor |
|---|---|---|
| `paste` + named-vector lookup on 6.46M rows in `lapply` | Integer equi-join via `data.table` (radix-sorted merge) | ~100–500× |
| `lapply` over 6.46M rows for neighbor stats | `data.table` grouped aggregation (`[, .(max, min, mean), by = focal_row]`) — internally parallelized C code | ~50–200× |
| `do.call(rbind, list_of_6.46M_vectors)` | Eliminated entirely; stats written directly via `set()` | ∞ (removed) |
| 5 full data.frame copies (`cell_data <- ...`) | Zero copies — `:=` and `set()` modify in place | ~5× memory, ~10× time |
| Prediction in a loop (assumed) | Single `predict()` call on full matrix (or large batches) | ~100–1000× |
| ~86+ hours estimated | ~10–30 minutes expected | ~170–500× overall |

**Key invariants preserved:**
- The trained Random Forest model is loaded from disk and never retrained.
- The original numerical estimand (predicted GDP) is identical — the same features are computed with the same arithmetic (max, min, mean of non-NA neighbor values), and the same `predict()` function is called on the same model.
- All 15 neighbor-derived features (3 stats × 5 variables) are computed identically to the original logic, including NA handling.