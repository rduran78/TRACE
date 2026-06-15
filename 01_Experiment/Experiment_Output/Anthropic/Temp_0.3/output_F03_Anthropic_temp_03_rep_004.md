 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the first critical bottleneck:

1. **Row-level `lapply` over 6.46M rows.** Each iteration performs character coercion (`as.character`), string pasting (`paste(..., sep="_")`), named-vector lookups (`id_to_ref[...]`, `idx_lookup[...]`), and NA filtering. This is ~6.46M R-level function calls with repeated string allocation.
2. **Named-vector lookups are O(n) hash lookups on character keys.** With 6.46M keys in `idx_lookup`, each lookup is not trivial, and it is done millions of times inside the loop.
3. **String construction inside the loop.** `paste(neighbor_cell_ids, data$year[i], sep = "_")` creates new character vectors on every iteration — enormous GC pressure.

**`compute_neighbor_stats`** is the second bottleneck:

1. **Another `lapply` over 6.46M rows**, each extracting a variable-length subset of a numeric vector, removing NAs, and computing three summary statistics.
2. **`do.call(rbind, result)` on a 6.46M-element list of 3-element vectors** — this is a well-known R anti-pattern that creates a massive intermediate list and then binds row-by-row.

**Outer loop** calls `compute_and_add_neighbor_features` 5 times, so the above costs are multiplied by 5.

### B. Random Forest Inference Bottleneck

With ~6.46M rows × ~110 predictors, a single `predict()` call on a `ranger` or `randomForest` object is itself expensive:

1. **Memory:** The prediction matrix alone is ~6.46M × 110 × 8 bytes ≈ 5.3 GB as a dense numeric matrix. On a 16 GB laptop this risks swapping.
2. **Object copying:** If `cell_data` is a `data.frame`, converting to the matrix required by `predict()` triggers a full copy. If columns are added iteratively (the outer loop), each `cell_data$new_col <- ...` may trigger a copy-on-modify of the entire data.frame.
3. **Single-call prediction on 6.46M rows** may itself take a long time depending on tree count and depth; chunked prediction can improve cache locality and reduce peak memory.

### C. Estimated Time Breakdown (86+ hours)

| Phase | Estimated share |
|---|---|
| `build_neighbor_lookup` (string ops, 6.46M iterations) | ~30–40% |
| `compute_neighbor_stats` × 5 vars (lapply + rbind) | ~30–40% |
| RF `predict()` + data conversion | ~15–25% |
| Repeated data.frame copy-on-modify | ~5–10% |

---

## 2. OPTIMIZATION STRATEGY

| Problem | Solution |
|---|---|
| String-key lookups in `build_neighbor_lookup` | Replace with integer-indexed lookup via `data.table` join on `(id, year)` → row index. Build a flat edge list (from_row, to_row) once. |
| Row-level `lapply` in `compute_neighbor_stats` | Replace with vectorized grouped aggregation on the edge list using `data.table`. |
| `do.call(rbind, ...)` on millions of rows | Eliminated — `data.table` returns a single table directly. |
| 5× repeated loop | Single grouped aggregation per variable, all vectorized. |
| `data.frame` copy-on-modify | Use `data.table` with `:=` (in-place column addition). |
| RF prediction memory spike | Predict in chunks (~500K rows) to cap peak memory. |
| RF model loading | Load once, reuse; use `ranger` if possible (faster predict). |

**Expected speedup:** From 86+ hours to roughly **10–30 minutes** depending on hardware, dominated by the RF prediction step.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED PIPELINE: Feature Preparation + Chunked RF Prediction
# =============================================================================
# Requirements: data.table, ranger (or randomForest)
# Preserves: trained RF model object, original numerical estimand

library(data.table)

# ---- 0. LOAD DATA -----------------------------------------------------------
# Assume:
#   cell_data          : data.frame/data.table with columns id, year, ntl, ec,
#                        pop_density, def, usd_est_n2, ... (all predictor cols)
#   id_order           : integer vector of cell IDs in the order matching
#                        rook_neighbors_unique
#   rook_neighbors_unique : spdep nb object (list of integer index vectors)
#   rf_model           : trained model object (ranger or randomForest)

# Convert to data.table in place (no copy if already data.table)
setDT(cell_data)

# ---- 1. BUILD FLAT EDGE LIST (replaces build_neighbor_lookup) ----------------
build_edge_list <- function(cell_data, id_order, neighbors) {
  # Map each cell id to its position in id_order
  n_ids <- length(id_order)

  # Build a data.table mapping (id, year) -> row number in cell_data
  cell_data[, .row_idx := .I]
  row_lookup <- cell_data[, .(id, year, .row_idx)]
  setkey(row_lookup, id, year)

  # Expand the nb object into a flat edge list: (from_id_pos, to_id_pos)
  from_pos <- rep(seq_len(n_ids), lengths(neighbors))
  to_pos   <- unlist(neighbors, use.names = FALSE)

  # Convert positions to actual cell IDs
  from_id <- id_order[from_pos]
  to_id   <- id_order[to_pos]

  edges <- data.table(from_id = from_id, to_id = to_id)

  # Get unique years
  years <- sort(unique(cell_data$year))

  # Cross-join edges with years: each edge exists in every year
  # This gives us (from_id, to_id, year) — the full directed neighbor-year list
  edge_year <- CJ_dt(edges, years)

  # Join to get from_row and to_row
  # from_row: the row in cell_data for (from_id, year)
  # to_row:   the row in cell_data for (to_id, year)
  setnames(row_lookup, c("id", "year", ".row_idx"),
           c("from_id", "year", "from_row"))
  setkey(row_lookup, from_id, year)
  edge_year <- row_lookup[edge_year, on = .(from_id, year), nomatch = 0L]

  setnames(row_lookup, c("from_id", "year", "from_row"),
           c("to_id", "year", "to_row"))
  setkey(row_lookup, to_id, year)
  edge_year <- row_lookup[edge_year, on = .(to_id, year), nomatch = 0L]

  # Restore row_lookup names for safety
  setnames(row_lookup, c("to_id", "year", "to_row"),
           c("id", "year", ".row_idx"))

  # Clean up helper column
  cell_data[, .row_idx := NULL]

  # Return only what we need: from_row, to_row (integer vectors)
  edge_year[, .(from_row, to_row)]
}

# Helper: cross join a data.table of edges with a vector of years
CJ_dt <- function(edges, years) {
  years_dt <- data.table(year = years)
  # keyed cross join
  edges[, .cj_key := 1L]
  years_dt[, .cj_key := 1L]
  result <- edges[years_dt, on = ".cj_key", allow.cartesian = TRUE]
  result[, .cj_key := NULL]
  result
}

cat("Building edge list...\n")
edge_list <- build_edge_list(cell_data, id_order, rook_neighbors_unique)
# edge_list has columns: from_row, to_row
# Meaning: for the cell-year at row `from_row`, row `to_row` is a neighbor.

cat(sprintf("Edge list: %s rows\n", format(nrow(edge_list), big.mark = ",")))

# ---- 2. VECTORIZED NEIGHBOR STATS (replaces compute_neighbor_stats) ----------
compute_and_add_all_neighbor_features <- function(cell_data, edge_list,
                                                   var_names) {
  # For each variable, we need: neighbor_max, neighbor_min, neighbor_mean
  # Strategy: attach the neighbor's value to each edge, then group by from_row.

  for (vn in var_names) {
    cat(sprintf("  Computing neighbor stats for: %s\n", vn))

    # Extract the variable values for the "to" (neighbor) rows
    vals <- cell_data[[vn]]
    edge_list[, nval := vals[to_row]]

    # Remove edges where the neighbor value is NA
    valid <- edge_list[!is.na(nval)]

    # Grouped aggregation — one pass
    agg <- valid[, .(
      nb_max  = max(nval),
      nb_min  = min(nval),
      nb_mean = mean(nval)
    ), by = from_row]

    # Initialize columns with NA
    max_col  <- paste0(vn, "_neighbor_max")
    min_col  <- paste0(vn, "_neighbor_min")
    mean_col <- paste0(vn, "_neighbor_mean")

    cell_data[, (max_col)  := NA_real_]
    cell_data[, (min_col)  := NA_real_]
    cell_data[, (mean_col) := NA_real_]

    # Assign aggregated values by row index (in place, no copy)
    set(cell_data, i = agg$from_row, j = max_col,  value = agg$nb_max)
    set(cell_data, i = agg$from_row, j = min_col,  value = agg$nb_min)
    set(cell_data, i = agg$from_row, j = mean_col, value = agg$nb_mean)
  }

  # Clean up temp column from edge_list
  edge_list[, nval := NULL]

  invisible(cell_data)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
compute_and_add_all_neighbor_features(cell_data, edge_list, neighbor_source_vars)
cat("Neighbor features complete.\n")

# Free the edge list — it can be large
rm(edge_list)
gc()

# ---- 3. CHUNKED RANDOM FOREST PREDICTION ------------------------------------
# This preserves the trained model and the original numerical estimand.

chunked_predict_rf <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)

  # Identify predictor columns expected by the model
  # Works for both ranger and randomForest objects
  if (inherits(model, "ranger")) {
    pred_vars <- model$forest$independent.variable.names
  } else if (inherits(model, "randomForest")) {
    pred_vars <- rownames(model$importance)
  } else {
    stop("Unsupported model class: ", class(model)[1])
  }

  # Validate that all required columns exist
  missing_vars <- setdiff(pred_vars, names(newdata))
  if (length(missing_vars) > 0) {
    stop("Missing predictor columns: ", paste(missing_vars, collapse = ", "))
  }

  cat(sprintf("Predicting %s rows in %d chunks of up to %s...\n",
              format(n, big.mark = ","), n_chunks,
              format(chunk_size, big.mark = ",")))

  for (k in seq_len(n_chunks)) {
    i_start <- (k - 1L) * chunk_size + 1L
    i_end   <- min(k * chunk_size, n)

    # Extract only the predictor columns for this chunk — minimizes memory
    chunk_dt <- newdata[i_start:i_end, ..pred_vars]

    if (inherits(model, "ranger")) {
      chunk_pred <- predict(model, data = chunk_dt)$predictions
    } else {
      # randomForest expects a matrix or data.frame
      chunk_pred <- predict(model, newdata = as.data.frame(chunk_dt))
    }

    preds[i_start:i_end] <- chunk_pred

    if (k %% 5 == 0 || k == n_chunks) {
      cat(sprintf("  Chunk %d/%d done (rows %s–%s)\n",
                  k, n_chunks,
                  format(i_start, big.mark = ","),
                  format(i_end, big.mark = ",")))
    }

    # Free chunk memory
    rm(chunk_dt, chunk_pred)
    if (k %% 10 == 0) gc()
  }

  preds
}

# ---- 4. RUN PREDICTION ------------------------------------------------------
cat("Loading trained RF model...\n")
# rf_model <- readRDS("path/to/trained_rf_model.rds")  # load once

cat("Running prediction...\n")
cell_data[, predicted_gdp := chunked_predict_rf(rf_model, cell_data,
                                                 chunk_size = 500000L)]

cat("Pipeline complete.\n")
```

---

## 4. SUMMARY OF CHANGES AND EXPECTED IMPACT

| Component | Original | Optimized | Speedup Factor |
|---|---|---|---|
| **Neighbor lookup** | `lapply` over 6.46M rows with string keys | Flat integer edge list via `data.table` join | ~50–200× |
| **Neighbor stats** | `lapply` + `do.call(rbind, ...)` per variable | `data.table` grouped `[, .(max, min, mean), by=from_row]` | ~100–500× |
| **Column addition** | `data.frame$col <- ...` (copy-on-modify × 15 cols) | `data.table` `:=` and `set()` (in-place, zero-copy) | ~5–10× |
| **RF prediction** | Single `predict()` on 6.46M rows (peak ~10+ GB) | Chunked 500K-row prediction (peak ~1–2 GB per chunk) | Fits in 16 GB RAM; avoids swapping |
| **Overall estimated time** | **86+ hours** | **~10–30 minutes** | **~200–500×** |

### Key Guarantees

- **Trained RF model is never modified or retrained** — it is loaded once and used read-only in `predict()`.
- **Original numerical estimand is preserved** — the same `predict()` function is called with the same features; chunking does not alter predictions (concatenation of chunk outputs is identical to a single-call output).
- **Memory stays within 16 GB** — the edge list cross-join is the largest intermediate object (~1.37M edges × 28 years × 2 int columns ≈ ~600 MB), and prediction chunks cap at ~500K × 110 × 8 bytes ≈ 420 MB each.