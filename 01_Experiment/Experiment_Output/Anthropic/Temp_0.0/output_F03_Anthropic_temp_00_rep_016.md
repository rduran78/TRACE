 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. DIAGNOSIS

### 1.1 Feature-Preparation Bottlenecks

**`build_neighbor_lookup`:** This function calls `lapply` over ~6.46 million rows, and inside each iteration it performs:
- Character coercion and named-vector lookup (`id_to_ref[as.character(...)]`)
- `paste` to build keys
- Named-vector lookup on `idx_lookup` (a named character vector of length 6.46M — each lookup is O(n) hash probe on a very large vector)
- NA filtering and `as.integer` coercion

**Cost:** ~6.46M iterations × multiple string allocations and hash lookups = extremely slow. This alone could take hours.

**`compute_neighbor_stats`:** Another `lapply` over 6.46M rows, each extracting a variable-length subset of a numeric vector, removing NAs, and computing three summary statistics. The `do.call(rbind, result)` at the end binds 6.46M small vectors into a matrix — this is a known R anti-pattern that is very slow and memory-hungry.

**Outer loop:** This is called 5 times (once per neighbor source variable), so `compute_neighbor_stats` runs 5 × 6.46M = ~32.3M iterations total.

### 1.2 Prediction Workflow Bottlenecks (Inferred)

Although the prediction code isn't shown, common problems at this scale include:
- **Loading the model from disk on every call** instead of once.
- **Predicting row-by-row** instead of in a single batch `predict(model, newdata)`.
- **Unnecessary `data.frame` copies** (R's copy-on-modify semantics) when adding columns inside loops.
- **Memory pressure:** 6.46M rows × 110 columns of doubles ≈ 5.4 GB. Combined with a Random Forest object (often 1–4 GB) and intermediate copies, 16 GB RAM is tight, causing swapping.

---

## 2. OPTIMIZATION STRATEGY

| Area | Problem | Solution |
|---|---|---|
| Neighbor lookup | Per-row string pasting and named-vector lookup | Vectorize entirely with `data.table` keyed joins — no `lapply`, no `paste` per row |
| Neighbor stats | 6.46M `lapply` iterations + `do.call(rbind, ...)` | Expand neighbor pairs into a long table, compute grouped `max/min/mean` via `data.table` |
| Column addition | Repeated `cell_data <- cbind(...)` copies the whole data.frame | Use `data.table` `:=` (modify in place, zero copies) |
| Prediction | Possibly row-by-row or chunked sub-optimally | Single batch `predict()` call; if memory-constrained, chunk in ~500K blocks |
| Model loading | Possibly reloaded repeatedly | Load once with `readRDS`, keep in memory |
| Memory | ~5.4 GB data + model + intermediates > 16 GB | Use `data.table` (lower overhead), `gc()` between stages, predict in chunks |

**Expected speedup:** From 86+ hours to roughly 10–30 minutes for feature preparation, plus prediction time depending on the forest size.

---

## 3. WORKING R CODE

```r
# =============================================================================
# OPTIMIZED CELL-LEVEL GDP PREDICTION PIPELINE
# =============================================================================
# Prerequisites: install.packages(c("data.table", "ranger")) or whatever RF
# package was used for training. The code below is generic to both
# randomForest::predict and ranger::predict.

library(data.table)

# ---- 0. Load model ONCE ------------------------------------------------------
rf_model <- readRDS("path/to/trained_rf_model.rds")  # load once, reuse

# ---- 1. Convert cell_data to data.table in place ----------------------------
#    Assumes cell_data is a data.frame/data.table with columns: id, year, and
#    all predictor columns.
setDT(cell_data)

# ---- 2. Build neighbor edge list (vectorised, no lapply) ---------------------
build_neighbor_edgelist <- function(id_order, nb_object) {
  # nb_object: spdep nb list — nb_object[[i]] gives integer indices of
  # neighbors of the i-th element in id_order.
  #
  # Returns a data.table with columns: id, neighbor_id
  # where both refer to the original cell IDs (not positional indices).

  # Pre-allocate vectors
  n <- length(nb_object)
  lengths_vec <- lengths(nb_object)
  total_edges <- sum(lengths_vec)

  from_idx <- rep.int(seq_len(n), lengths_vec)
  to_idx   <- unlist(nb_object, use.names = FALSE)

  data.table(
    id          = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

cat("Building neighbor edge list...\n")
edge_dt <- build_neighbor_edgelist(id_order, rook_neighbors_unique)
# edge_dt has ~1.37M rows (directed edges)

# ---- 3. Compute ALL neighbor features in one vectorised pass -----------------
compute_all_neighbor_features <- function(cell_dt, edge_dt,
                                          source_vars) {
  # Join edges with year to create (id, year, neighbor_id) triples,
  # then look up neighbor values and aggregate.

  # Step A: Get unique (id, year) and their row indices
  cell_dt[, .rowid := .I]

  # Step B: Create the join table — every (id, year) paired with its neighbors
  #   We need: for each row (id, year), find all neighbor_ids, then look up
  #   the neighbor's value for that same year.

  # Keyed join: edge_dt on cell_dt to get years for each id
  # But that would explode to 6.46M × avg_neighbors rows.
  # Instead, work with (id, year, neighbor_id) and join neighbor values.

  # Unique id-year combinations (same as cell_dt rows)
  id_year <- cell_dt[, .(id, year, .rowid)]

  # Merge with edge list: for each (id, year), get all neighbor_ids
  # This is the most memory-intensive step.
  # Estimated size: 6.46M rows × avg ~4 neighbors = ~25.8M rows
  setkey(edge_dt, id)
  setkey(id_year, id)

  cat("  Expanding id-year-neighbor triples...\n")
  # Use allow.cartesian because each id maps to multiple neighbors
  expanded <- edge_dt[id_year, on = "id", allow.cartesian = TRUE,
                      nomatch = NULL]
  # expanded has columns: id, neighbor_id, year, .rowid
  # .rowid refers to the original row in cell_dt

  # Step C: For each source variable, look up the neighbor's value
  #   by joining on (neighbor_id == id, year == year)
  # Prepare a lookup keyed on (id, year)
  cat("  Preparing value lookup...\n")
  value_lookup <- cell_dt[, c("id", "year", source_vars), with = FALSE]
  setnames(value_lookup, "id", "neighbor_id")
  setkeyv(value_lookup, c("neighbor_id", "year"))
  setkeyv(expanded, c("neighbor_id", "year"))

  cat("  Joining neighbor values...\n")
  expanded <- value_lookup[expanded, on = c("neighbor_id", "year"),
                           nomatch = NA]
  # Now expanded has: neighbor_id, year, <source_vars>, id, .rowid

  # Step D: Aggregate by .rowid (= original row) to get max, min, mean
  cat("  Aggregating neighbor statistics...\n")
  agg_exprs <- list()
  for (v in source_vars) {
    v_sym <- as.name(v)
    agg_exprs[[paste0("n_max_", v)]]  <- substitute(
      suppressWarnings(max(X, na.rm = TRUE)), list(X = v_sym))
    agg_exprs[[paste0("n_min_", v)]]  <- substitute(
      suppressWarnings(min(X, na.rm = TRUE)), list(X = v_sym))
    agg_exprs[[paste0("n_mean_", v)]] <- substitute(
      mean(X, na.rm = TRUE), list(X = v_sym))
  }

  agg_result <- expanded[, eval(as.call(c(as.name("list"),
                                           agg_exprs))),
                          by = .rowid]

  # Replace Inf/-Inf (from max/min of all-NA) with NA
  inf_cols <- grep("^n_max_|^n_min_", names(agg_result), value = TRUE)
  for (col in inf_cols) {
    set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
  }

  # Step E: Join aggregated features back to cell_dt by .rowid
  setkey(agg_result, .rowid)
  feat_cols <- setdiff(names(agg_result), ".rowid")

  cat("  Merging features back to main table...\n")
  cell_dt[agg_result, (feat_cols) := mget(feat_cols), on = ".rowid"]

  # Handle rows with NO neighbors (they won't appear in agg_result)
  # They already have NA from the join (data.table default), which is correct.

  # Clean up
  cell_dt[, .rowid := NULL]

  invisible(cell_dt)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features...\n")
system.time({
  compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})
# cell_data is modified in place — no copy

# Free intermediate memory
rm(edge_dt)
gc()

# ---- 4. Prediction (batch, memory-aware) -------------------------------------
predict_in_chunks <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  if (n <= chunk_size) {
    return(predict(model, newdata = newdata))
  }

  cat("Predicting in chunks of", chunk_size, "...\n")
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  for (i in seq_along(starts)) {
    s <- starts[i]
    e <- min(s + chunk_size - 1L, n)
    cat(sprintf("  Chunk %d/%d (rows %d-%d)\n", i, length(starts), s, e))
    preds[s:e] <- predict(model, newdata = newdata[s:e, ])
    if (i %% 5 == 0) gc()
  }
  preds
}

# Prepare the prediction data.frame (Random Forest predict methods often
# require a plain data.frame, not data.table)
predictor_cols <- setdiff(names(cell_data),
                          c("id", "year", "gdp", "response_variable"))
# Adjust the above exclusion list to match your actual non-predictor columns.

cat("Running Random Forest prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_in_chunks(
    rf_model,
    newdata = as.data.frame(.SD),
    chunk_size = 500000L
  ), .SDcols = predictor_cols]
})

cat("Done.\n")

# ---- 5. (Optional) Write results --------------------------------------------
# fwrite(cell_data[, .(id, year, predicted_gdp)], "predictions.csv")
```

---

## 4. SUMMARY OF KEY CHANGES AND WHY THEY WORK

| # | Change | Mechanism | Impact |
|---|---|---|---|
| 1 | Replace `build_neighbor_lookup` (6.46M-iteration `lapply` with string ops) with a vectorised `data.table` edge list built from the `nb` object using `rep.int` + `unlist` | Eliminates all per-row `paste`, `as.character`, and named-vector lookups | **~1000× faster** (seconds vs. hours) |
| 2 | Replace 5 × `compute_neighbor_stats` (each 6.46M-iteration `lapply` + `do.call(rbind, ...)`) with a single `data.table` grouped aggregation on the expanded edge table | One vectorised join + one grouped aggregation replaces 32.3M R-level iterations | **~500–1000× faster**; eliminates the `do.call(rbind, 6.46M-element list)` anti-pattern |
| 3 | Use `data.table` `:=` for in-place column addition | Avoids R's copy-on-modify; no full-table copies when adding 15 new columns | **Saves ~5 GB** of transient memory allocation |
| 4 | Load RF model once with `readRDS` | Avoids repeated disk I/O | Trivial but important if the original code reloaded per-chunk |
| 5 | Batch `predict()` with optional chunking | Single `predict` call is orders of magnitude faster than row-by-row; chunking at 500K keeps memory bounded | Prediction itself is typically fast for a pre-trained forest; chunking prevents OOM on 16 GB |
| 6 | Explicit `gc()` calls between stages | Returns freed memory to the OS promptly | Prevents swap thrashing on a 16 GB machine |

**The trained Random Forest model is never modified or retrained.** All numerical computations (`max`, `min`, `mean`) are identical to the originals, preserving the original numerical estimand.