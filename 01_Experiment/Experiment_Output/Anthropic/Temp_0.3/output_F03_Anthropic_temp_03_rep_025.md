 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering — all interpreted R. At ~6.46M iterations this alone can take hours. |
| **Named-vector lookup (`idx_lookup[neighbor_keys]`)** | `idx_lookup` is a named character vector of length 6.46M. Subsetting a named vector is O(n) per call in base R (linear scan or hash with overhead). Called millions of times. |
| **Repeated `paste` / `as.character` inside the loop** | String allocation inside a hot loop creates massive GC pressure. |
| **Output is a list of 6.46M integer vectors** | ~6.46M list elements, each a small integer vector → enormous memory overhead from R's SEXP headers (~56 bytes per vector + 40 bytes per list slot). |

**`compute_neighbor_stats`** compounds the problem:

| Problem | Detail |
|---|---|
| **`lapply` over 6.46M elements again, once per variable** | 5 variables × 6.46M = ~32.3M R-level function calls. |
| **`do.call(rbind, result)` on 6.46M single-row matrices** | This is notoriously slow — it must allocate and copy incrementally. |
| **Repeated NA filtering inside the loop** | `vals[idx]` then `neighbor_vals[!is.na(...)]` for every row. |

### B. Random Forest Inference Bottleneck

With ~6.46M rows × 110 predictors, a single `predict()` call on a `ranger` or `randomForest` model will:

- Attempt to allocate a prediction matrix of ~6.46M × 110 ≈ 5.4 GB (double precision), likely exceeding 16 GB RAM when combined with the model and data.
- If using `randomForest::predict`, it converts to a dense matrix internally and is single-threaded.
- If using `ranger::predict`, it is multi-threaded but still needs the full matrix in memory.

### C. Object Copying / Memory

Adding columns to a `data.frame` inside a `for` loop (`cell_data <- compute_and_add_neighbor_features(...)`) triggers full-copy semantics in base R. With 6.46M rows × 110+ columns, each copy is ~5+ GB. Five iterations = ~25 GB of transient allocation on a 16 GB machine → swap thrashing.

### Summary: Where the 86+ Hours Go

| Phase | Estimated share |
|---|---|
| `build_neighbor_lookup` (string ops, named-vector lookup) | ~35–45% |
| `compute_neighbor_stats` × 5 vars (lapply + rbind) | ~25–35% |
| `data.frame` column-addition copies | ~10–15% |
| `predict()` on 6.46M rows (if single-threaded / memory-bound) | ~10–20% |

---

## 2. Optimization Strategy

### Principle: Replace interpreted R loops with vectorized / `data.table` operations, and chunk the prediction.

| Bottleneck | Strategy |
|---|---|
| `build_neighbor_lookup` | Build a **flat `data.table`** of `(row_idx, neighbor_row_idx)` pairs using vectorized joins — no `lapply`, no `paste` inside a loop. |
| `compute_neighbor_stats` | **Group-by aggregation** on the flat edge table joined to the value column — one `data.table` operation per variable, fully vectorized in C. |
| Column addition copies | Use `data.table` **set-by-reference** (`:=`) — zero copies. |
| Prediction memory | **Chunk prediction** into batches of ~500K rows; optionally use `ranger` with `num.threads`. |
| Overall memory | Keep one `data.table` throughout; avoid materializing the 6.46M-element list. |

Expected speedup: **~100–300×** for feature preparation (hours → minutes), plus manageable memory for prediction.

---

## 3. Working R Code

```r
# =============================================================================
# 0. LIBRARIES
# =============================================================================
library(data.table)
# Use ranger for multi-threaded prediction if the model is a ranger object.
# library(ranger)

# =============================================================================
# 1. CONVERT CORE DATA TO data.table (once, in-place)
# =============================================================================
setDT(cell_data)  # converts in-place, no copy

# Ensure key columns exist
stopifnot(all(c("id", "year") %in% names(cell_data)))

# =============================================================================
# 2. BUILD FLAT NEIGHBOR EDGE TABLE (vectorized, replaces build_neighbor_lookup)
# =============================================================================
build_neighbor_edges <- function(cell_dt, id_order, nb_list) {
  # --- Map each cell id to its position in id_order --------------------------
  # id_order is the vector of cell IDs in the order matching nb_list indices.
  n_ids   <- length(id_order)
  id_to_ref <- data.table(id = id_order, ref_idx = seq_len(n_ids))

  # --- Expand nb_list into a flat (ref_idx, neighbor_ref_idx) table ----------
  # nb_list[[i]] contains integer indices into id_order for the neighbors of

  # id_order[i].
  lengths_vec <- lengths(nb_list)                       # integer vector
  from_ref    <- rep.int(seq_len(n_ids), lengths_vec)   # vectorized repeat
  to_ref      <- unlist(nb_list, use.names = FALSE)     # flat neighbor refs

  edge_ids <- data.table(
    from_id = id_order[from_ref],
    to_id   = id_order[to_ref]
  )
  rm(from_ref, to_ref, lengths_vec)

  # --- Build row-index lookup: (id, year) -> row position --------------------
  cell_dt[, .row_idx := .I]
  row_lookup <- cell_dt[, .(id, year, .row_idx)]

  # --- Cross-join edges with years (only years present in data) --------------
  # For every edge (from_id, to_id) we need every year that the *from* cell

  # has data for.  Because the panel is balanced (all cells × all years),
  # we can simply cross-join with the unique years.
  unique_years <- sort(unique(cell_dt$year))

  # Expand: every directed edge × every year
  # Use CJ inside a merge chain to stay memory-efficient.
  # Step A: attach row_idx for the "from" cell (the focal cell)
  edges_with_year <- edge_ids[
    , .(to_id = to_id), by = from_id
  ][
    CJ(from_id = unique(edge_ids$from_id), year = unique_years),
    on = "from_id",
    allow.cartesian = TRUE,
    nomatch = NULL
  ]

  # Step B: attach focal-cell row index
  setnames(row_lookup, c("id", "year", ".row_idx"),
           c("from_id", "year", "focal_row"))
  edges_with_year <- row_lookup[edges_with_year, on = .(from_id, year), nomatch = NULL]

  # Step C: attach neighbor-cell row index
  setnames(row_lookup, c("from_id", "year", "focal_row"),
           c("to_id", "year2", "neighbor_row"))
  edges_with_year[, year2 := year]
  edges_with_year <- row_lookup[edges_with_year,
                                on = .(to_id, year2 = year2),
                                nomatch = NULL]

  # Clean up
  edges_with_year[, year2 := NULL]
  setnames(row_lookup, c("to_id", "year2", "neighbor_row"),
           c("id", "year", ".row_idx"))  # restore
  cell_dt[, .row_idx := NULL]

  # Result: data.table with columns  focal_row | neighbor_row

  edges_with_year[, .(focal_row, neighbor_row)]
}

cat("Building flat neighbor edge table …\n")
system.time({
  edge_dt <- build_neighbor_edges(cell_data, id_order, rook_neighbors_unique)
})
# edge_dt has ~1.37M edges × 28 years ≈ 38.5M rows (manageable)

cat(sprintf("Edge table: %s rows\n", format(nrow(edge_dt), big.mark = ",")))

# =============================================================================
# 3. VECTORIZED NEIGHBOR STATS (replaces compute_neighbor_stats + outer loop)
# =============================================================================
add_neighbor_features <- function(cell_dt, edge_dt, var_names) {
  for (vn in var_names) {
    cat(sprintf("  Computing neighbor stats for '%s' …\n", vn))

    # Attach the neighbor's value to every edge row
    vals <- cell_dt[[vn]]
    edge_dt[, nval := vals[neighbor_row]]

    # Aggregate per focal row — max, min, mean — dropping NAs
    agg <- edge_dt[!is.na(nval),
                   .(vmax  = max(nval),
                     vmin  = min(nval),
                     vmean = mean(nval)),
                   by = focal_row]

    # Initialise new columns with NA
    max_col  <- paste0("neighbor_max_",  vn)
    min_col  <- paste0("neighbor_min_",  vn)
    mean_col <- paste0("neighbor_mean_", vn)

    set(cell_dt, j = max_col,  value = NA_real_)
    set(cell_dt, j = min_col,  value = NA_real_)
    set(cell_dt, j = mean_col, value = NA_real_)

    # Write aggregated values by reference (no copy of cell_dt)
    set(cell_dt, i = agg$focal_row, j = max_col,  value = agg$vmax)
    set(cell_dt, i = agg$focal_row, j = min_col,  value = agg$vmin)
    set(cell_dt, i = agg$focal_row, j = mean_col, value = agg$vmean)
  }
  edge_dt[, nval := NULL]  # clean temp column
  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features …\n")
system.time({
  add_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# Free the edge table — no longer needed
rm(edge_dt); gc()

# =============================================================================
# 4. CHUNKED RANDOM FOREST PREDICTION (memory-safe on 16 GB)
# =============================================================================
# Assumes:
#   rf_model   — the pre-trained model object (ranger or randomForest)
#   pred_vars  — character vector of the ~110 predictor column names

predict_chunked <- function(model, dt, pred_vars,
                            chunk_size = 500000L,
                            is_ranger  = inherits(model, "ranger")) {
  n <- nrow(dt)
  n_chunks <- ceiling(n / chunk_size)
  preds <- numeric(n)

  cat(sprintf("Predicting %s rows in %d chunks …\n",
              format(n, big.mark = ","), n_chunks))

  for (k in seq_len(n_chunks)) {
    i_start <- (k - 1L) * chunk_size + 1L
    i_end   <- min(k * chunk_size, n)
    idx     <- i_start:i_end

    # Extract only the predictor columns for this chunk (minimises peak RAM)
    chunk_df <- as.data.frame(dt[idx, ..pred_vars])

    if (is_ranger) {
      preds[idx] <- predict(model, data = chunk_df,
                             num.threads = parallel::detectCores())$predictions
    } else {
      # randomForest package
      preds[idx] <- predict(model, newdata = chunk_df)
    }

    if (k %% 5 == 0 || k == n_chunks) {
      cat(sprintf("  chunk %d / %d done\n", k, n_chunks))
    }
  }
  preds
}

cat("Running Random Forest prediction …\n")
system.time({
  cell_data[, predicted_gdp := predict_chunked(rf_model, cell_data, pred_vars)]
})

cat("Done.\n")
```

---

## 4. Why This Preserves Correctness

| Requirement | Guarantee |
|---|---|
| **Trained RF model unchanged** | `predict()` is read-only; the model object is never modified. |
| **Numerical estimand preserved** | The neighbor features are computed with the identical `max`, `min`, `mean` aggregations, excluding `NA`s, matching the original `compute_neighbor_stats` logic exactly. The same predictor columns are fed to the same model, producing identical predictions (up to floating-point associativity of `mean`, which is negligible). |

## 5. Expected Performance

| Phase | Before | After (estimated) |
|---|---|---|
| Neighbor lookup construction | ~20–30 hours | **2–5 minutes** (vectorized joins) |
| Neighbor stats (5 vars) | ~30–40 hours | **3–8 minutes** (`data.table` grouped agg) |
| Column addition | ~10 hours (copies) | **< 1 second** (by-reference `:=` / `set`) |
| RF prediction (6.46M rows) | ~10–15 hours | **15–45 minutes** (chunked, multi-threaded `ranger`) |
| **Total** | **~86+ hours** | **~25–60 minutes** |

The dominant gains come from eliminating the 6.46M-iteration interpreted R loops and the repeated full-copy `data.frame` modifications. Chunked prediction keeps peak memory well within 16 GB.