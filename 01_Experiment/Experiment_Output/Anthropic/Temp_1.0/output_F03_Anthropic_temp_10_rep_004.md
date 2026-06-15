 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis

There are two distinct bottleneck zones in this pipeline:

### Bottleneck 1: Feature Preparation (`build_neighbor_lookup` + `compute_neighbor_stats`)

1. **`build_neighbor_lookup` creates ~6.46 million R lists via `lapply`**, each performing character key construction (`paste`), hash-table lookups (`idx_lookup[neighbor_keys]`), and NA filtering. With ~1.37M neighbor relationships spread across 28 years, the total number of string-match operations is enormous. The per-row overhead of string allocation, named-vector lookup (linear or hash), and `as.integer` coercion accumulates to hours.

2. **`compute_neighbor_stats` uses `lapply` over 6.46M elements**, each subsetting a numeric vector, removing NAs, and computing `max/min/mean`. The `do.call(rbind, result)` at the end creates a 6.46M-row matrix from a list of 6.46M 3-element vectors — this is an O(n²) memory-copy pattern in R. This function is called **5 times** (once per neighbor source variable), multiplying the cost.

3. **String-keyed lookups are the core anti-pattern.** The entire neighbor lookup is mediated by `paste(..., sep="_")` keys matched against a named vector. This is orders of magnitude slower than integer-indexed operations.

### Bottleneck 2: Random Forest Prediction

1. **Predicting 6.46M rows with ~110 features through a Random Forest is inherently expensive.** If done in a single `predict()` call, memory for the prediction matrix (~5.3 GB for a dense numeric matrix) can exceed available RAM, triggering swapping.

2. **If prediction is done row-by-row or in a tight R loop**, the overhead per call to `predict.randomForest` or `predict.ranger` (model object copying, data.frame coercion, factor-level checking) dominates. Even chunked prediction needs appropriately sized chunks.

3. **Model loading**: If the serialized RF model is large (hundreds of MB to multiple GB), repeated `readRDS` calls or unnecessary copies will thrash memory.

---

## Optimization Strategy

| Component | Problem | Solution |
|---|---|---|
| Neighbor lookup | String keys, per-row `lapply` | Replace with pure integer arithmetic: `(id_index - 1) * n_years + year_index`. Pre-expand the spatial `nb` object to cell-year integer row indices in vectorized form using `data.table`. |
| Neighbor stats | 6.46M-element `lapply` + `do.call(rbind,...)` | Vectorized grouped aggregation via `data.table`: explode neighbor pairs into an edge table, join variable values, then `group by` source row to compute max/min/mean in C-level code. |
| Feature binding | Repeated `cell_data <- ...` column addition (copy-on-modify) | Use `data.table` set-by-reference (`:=`) to add columns in place — zero copies. |
| RF prediction | Single giant call or row-level loop | Chunk prediction into ~500K-row batches; pre-allocate output vector; use `ranger` or `predict(..., num.threads)` if available. |
| Memory | Multiple large intermediate objects | Reuse edge table across variables; `rm()` + `gc()` intermediaries; never duplicate the model object. |

**Expected speedup:** From 86+ hours to roughly **15–45 minutes** depending on model size and disk I/O.

---

## Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE
# =============================================================================
# Prerequisites:
#   - data.table, ranger (or randomForest)
#   - cell_data: data.frame/data.table with columns: id, year, ntl, ec,
#                pop_density, def, usd_est_n2, ... (all predictor columns)
#   - id_order: integer vector of unique cell IDs in the order matching
#               rook_neighbors_unique
#   - rook_neighbors_unique: nb object (list of integer index vectors into id_order)
#   - rf_model: pre-trained Random Forest model (already loaded or to be loaded once)
# =============================================================================

library(data.table)

# ---- 0. Load model ONCE, keep in memory ----
# rf_model <- readRDS("path/to/trained_rf_model.rds")   # do this once at top

# ---- 1. Convert cell_data to data.table (by reference if possible) ----
if (!is.data.table(cell_data)) {
  setDT(cell_data)  # converts in place, no copy
}

# ---- 2. Build integer-indexed neighbor edge table (ONCE) ----
# This replaces build_neighbor_lookup entirely.

build_neighbor_edge_table <- function(cell_data, id_order, nb_obj) {
  # Map each unique cell ID to its position in id_order
  n_ids <- length(id_order)

  # Build spatial edge list: source_spatial_idx -> neighbor_spatial_idx
  # nb_obj[[i]] gives the integer indices (into id_order) of neighbors of
  # id_order[i]. We expand this into a two-column integer matrix.
  src_idx <- rep(
    seq_len(n_ids),
    times = lengths(nb_obj)
  )
  dst_idx <- unlist(nb_obj, use.names = FALSE)

  # Now translate spatial indices to actual cell IDs
  src_id <- id_order[src_idx]
  dst_id <- id_order[dst_idx]

  # Create edge table at the spatial level
  spatial_edges <- data.table(src_id = src_id, dst_id = dst_id)

  # Create a row-index lookup in cell_data: (id, year) -> row number

  cell_data[, .row_idx := .I]

  # Cross-join spatial edges with years present in data:
  # For every (src_id, dst_id) pair, we need every year where BOTH exist.
  # Efficient approach: join edges to cell_data twice (on src and dst).

  # Keyed lookup tables
  src_lookup <- cell_data[, .(src_row = .row_idx, year = year, src_id = id)]
  setkey(src_lookup, src_id, year)

  dst_lookup <- cell_data[, .(dst_row = .row_idx, year = year, dst_id = id)]
  setkey(dst_lookup, dst_id, year)

  # Join: for each spatial edge, find all years where src exists
  setkey(spatial_edges, src_id)
  edges_with_year <- src_lookup[spatial_edges, on = .(src_id), nomatch = 0L,
                                 allow.cartesian = TRUE]
  # edges_with_year now has: src_id, year, src_row, dst_id

  # Now join to find dst_row for the same year
  setkey(edges_with_year, dst_id, year)
  full_edges <- dst_lookup[edges_with_year, on = .(dst_id, year), nomatch = 0L]
  # full_edges has: dst_id, year, dst_row, src_id, src_row

  # Clean up temporary column
  cell_data[, .row_idx := NULL]

  # Return minimal edge table: src_row, dst_row (integer row indices into cell_data)
  full_edges[, .(src_row = src_row, dst_row = dst_row)]
}

cat("Building neighbor edge table...\n")
system.time({
  edge_table <- build_neighbor_edge_table(cell_data, id_order, rook_neighbors_unique)
})
# edge_table: each row says "cell_data row src_row has neighbor at cell_data row dst_row"
cat(sprintf("Edge table: %s rows\n", format(nrow(edge_table), big.mark = ",")))


# ---- 3. Vectorized neighbor feature computation (replaces compute_neighbor_stats) ----

add_neighbor_features_vectorized <- function(cell_data, edge_table, var_names) {
  # For each variable, pull the neighbor values via edge_table,
  # then group by src_row to compute max, min, mean.
  # All done in data.table C-level grouped operations.

  for (vname in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", vname))

    # Extract neighbor values via integer indexing (extremely fast)
    edge_table[, val := cell_data[[vname]][dst_row]]

    # Grouped aggregation
    stats <- edge_table[!is.na(val),
                        .(nmax  = max(val),
                          nmin  = min(val),
                          nmean = mean(val)),
                        by = src_row]

    # Pre-fill with NA
    col_max  <- paste0("n_max_",  vname)
    col_min  <- paste0("n_min_",  vname)
    col_mean <- paste0("n_mean_", vname)

    set(cell_data, j = col_max,  value = NA_real_)
    set(cell_data, j = col_min,  value = NA_real_)
    set(cell_data, j = col_mean, value = NA_real_)

    # Fill computed values by reference (no copy)
    set(cell_data, i = stats$src_row, j = col_max,  value = stats$nmax)
    set(cell_data, i = stats$src_row, j = col_min,  value = stats$nmin)
    set(cell_data, i = stats$src_row, j = col_mean, value = stats$nmean)
  }

  # Clean up temp column in edge_table
  edge_table[, val := NULL]

  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  add_neighbor_features_vectorized(cell_data, edge_table, neighbor_source_vars)
})

# Free edge table memory now that features are computed
rm(edge_table)
gc()


# ---- 4. Chunked Random Forest Prediction ----

predict_rf_chunked <- function(model, newdata, chunk_size = 500000L,
                               num_threads = NULL) {
  # newdata: data.table with all predictor columns
  # Returns: numeric vector of predictions, same length as nrow(newdata)

  n <- nrow(newdata)
  predictions <- numeric(n)   # pre-allocate full output vector

  # Determine which columns the model needs
  # For ranger: model$forest$independent.variable.names

  # For randomForest: names(model$forest$xlevels) or colnames(model$forest$xlevels)
  # We pass the full data and let predict() select the right columns.

  n_chunks <- ceiling(n / chunk_size)
  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))

  for (i in seq_len(n_chunks)) {
    start_idx <- (i - 1L) * chunk_size + 1L
    end_idx   <- min(i * chunk_size, n)

    chunk <- newdata[start_idx:end_idx]

    # Detect model type and predict accordingly
    if (inherits(model, "ranger")) {
      if (!is.null(num_threads)) {
        pred <- predict(model, data = chunk, num.threads = num_threads)$predictions
      } else {
        pred <- predict(model, data = chunk)$predictions
      }
    } else if (inherits(model, "randomForest")) {
      pred <- as.numeric(predict(model, newdata = chunk))
    } else {
      # Generic fallback
      pred <- as.numeric(predict(model, newdata = chunk))
    }

    predictions[start_idx:end_idx] <- pred

    if (i %% 5 == 0 || i == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %s-%s)\n",
                  i, n_chunks,
                  format(start_idx, big.mark = ","),
                  format(end_idx, big.mark = ",")))
    }
  }

  predictions
}

# --- Run prediction ---
cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(
    model      = rf_model,
    newdata    = cell_data,
    chunk_size = 500000L,
    num_threads = parallel::detectCores() - 1L   # ignored if not ranger
  )]
})

cat("Done.\n")
```

---

## Summary of Changes and Why They Work

| Original | Optimized | Why |
|---|---|---|
| `paste(id, year)` string keys for 6.46M rows | Integer row-index edge table built once via `data.table` joins | Eliminates ~billions of string allocations and hash lookups |
| `lapply` over 6.46M rows in `build_neighbor_lookup` | Vectorized `rep`/`unlist` + `data.table` keyed join | C-level vectorized operations, no R-level loop overhead |
| `lapply` + `do.call(rbind, ...)` in `compute_neighbor_stats` (called 5×) | `data.table` grouped `[, .(max, min, mean), by=src_row]` (called 5×) | Grouped aggregation in C via GForce; no list creation or row-binding |
| `cell_data <- cbind(...)` (copy-on-modify, 15 times for 5 vars × 3 stats) | `set(cell_data, j=..., value=...)` by reference | Zero-copy column addition; no 6.46M-row data.frame duplication |
| Single monolithic `predict()` on 6.46M rows (potential OOM) | Chunked prediction in 500K-row batches with pre-allocated output | Keeps peak memory under control; no intermediate concatenation |
| Possible repeated `readRDS` of model | Load model once, pass by reference | R does not copy large objects on read-access (copy-on-modify semantics) |

**Numerical equivalence is preserved**: the same max/min/mean aggregations are computed on exactly the same neighbor sets, and the trained RF model is used as-is with `predict()` — no retraining, no rounding, no approximation.