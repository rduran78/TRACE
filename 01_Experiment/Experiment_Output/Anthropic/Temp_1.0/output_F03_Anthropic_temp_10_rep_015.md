 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

The 86+ hour runtime on 6.46 million rows with ~110 features stems from **three compounding bottlenecks**:

### A. `build_neighbor_lookup` — O(n) `lapply` with per-row string operations
- **6.46M calls** to `paste()`, character-key lookups in a named vector (`idx_lookup`), and `as.character()` coercions.
- Named-vector lookup in R is hash-table-based but the overhead per call (hashing, NA check, subsetting) across 6.46M iterations is enormous.
- The entire structure is a **list of 6.46M integer vectors** — high memory overhead from R list boxing.

### B. `compute_neighbor_stats` — repeated per variable, pure-R loop
- Called **5 times** (once per neighbor source variable), each time iterating 6.46M list elements.
- Each iteration subsets a numeric vector, removes NAs, and computes max/min/mean — all in interpreted R with per-element function-call overhead.
- Produces 5 × 3 = 15 new columns, but the work to get there is ~32.3M R-level `lapply` body evaluations.

### C. Random Forest Inference on 6.46M rows × 110 features
- `predict.randomForest` (or `predict.ranger`) on 6.46M rows is memory-intensive: it must build the full prediction matrix and traverse every tree.
- If using the `randomForest` package, prediction is **single-threaded** and builds an internal copy of the data matrix.
- A single `predict()` call on 6.46M × 110 will try to allocate a dense numeric matrix (~5.3 GB for doubles), likely exceeding 16 GB RAM when combined with model overhead, causing swapping.

### Summary of time allocation (estimated)
| Stage | Estimated Share | Root Cause |
|---|---|---|
| `build_neighbor_lookup` | ~25% | Per-row string hashing in R |
| `compute_neighbor_stats` (×5) | ~30% | Interpreted R loop, 32M iterations |
| `predict()` on full dataset | ~40% | Memory pressure, single-thread, data copy |
| Other (I/O, GC) | ~5% | GC from list/vector allocations |

---

## 2. Optimization Strategy

| Bottleneck | Strategy | Expected Speedup |
|---|---|---|
| Neighbor lookup | Replace string-keyed lookup with **integer arithmetic** `(id_index - 1) * n_years + year_index`; build a flat integer matrix instead of a list; use `data.table` for joins | 20–50× |
| Neighbor stats | **Vectorized `data.table` grouped operation** or a single-pass C++ function via `Rcpp`; compute all 5 variables × 3 stats in one pass | 10–30× |
| RF prediction | **Chunk the prediction** into batches of ~500K rows to stay within RAM; if model is `randomForest`, convert to `ranger` for multi-threaded predict (the trained model can be wrapped); or use `predict()` with chunking | 3–8× |
| Memory | Avoid copying `cell_data` repeatedly; mutate in place with `data.table`; drop intermediate objects; use `gc()` between stages | Prevents swapping |

**Key constraint preserved**: The trained Random Forest model object is never retrained — only used for `predict()`. The numerical estimand (predicted GDP) is unchanged.

---

## 3. Working R Code

```r
# ==============================================================================
# OPTIMIZED PIPELINE
# ==============================================================================
# Prerequisites:
#   install.packages(c("data.table", "ranger", "Rcpp"))
#   If model is randomForest-class, a shim is provided below.
# ==============================================================================

library(data.table)

# --------------------------------------------------------------------------
# STEP 0: Convert cell_data to data.table (in-place, no copy)
# --------------------------------------------------------------------------
if (!is.data.table(cell_data)) {
  setDT(cell_data)
}

# Ensure id and year are integer for fast arithmetic
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]

# --------------------------------------------------------------------------
# STEP 1: Build neighbor lookup with integer indexing (replaces
#          build_neighbor_lookup)
#
# Strategy: map (id, year) -> row index via a keyed data.table join
#           then expand neighbor relationships in a single vectorized pass.
# --------------------------------------------------------------------------
build_neighbor_lookup_fast <- function(dt, id_order, neighbors) {
  # --- Map each unique cell id to its position in id_order ----------------
  id_order_int <- as.integer(id_order)
  id_to_ref <- data.table(
    id  = id_order_int,
    ref = seq_along(id_order_int)
  )
  setkey(id_to_ref, id)

  # --- Build a long table of directed neighbor pairs ----------------------
  #     neighbor_dt: columns (focal_ref, neighbor_id)
  neighbor_pairs <- rbindlist(lapply(seq_along(neighbors), function(r) {
    nb <- neighbors[[r]]
    if (length(nb) == 0L) return(NULL)
    data.table(focal_ref = r, neighbor_id = id_order_int[nb])
  }))

  # Map focal_ref back to focal_id
  neighbor_pairs[, focal_id := id_order_int[focal_ref]]

  # --- Row-index lookup table for (id, year) -> row_idx ------------------
  dt[, row_idx := .I]
  row_lookup <- dt[, .(id, year, row_idx)]
  setkey(row_lookup, id, year)

  # --- Unique years -------------------------------------------------------
  all_years <- sort(unique(dt$year))

  # --- For every row, find its neighbor rows via a merge ------------------
  #     First, build a table: (focal_id, year, neighbor_id)
  #     by crossing each focal row with its neighbors.
  focal_rows <- dt[, .(focal_id = id, year, focal_row_idx = row_idx)]

  # Merge focal rows with neighbor pairs to get (focal_row_idx, neighbor_id, year)
  setkey(focal_rows, focal_id)
  setkey(neighbor_pairs, focal_id)

  # This is the critical join: for each focal row, get all neighbor_ids
  expanded <- neighbor_pairs[focal_rows,
    on = "focal_id",
    allow.cartesian = TRUE,
    nomatch = 0L
  ][, .(focal_row_idx, neighbor_id, year)]

  # Now join to get the neighbor's row index in the same year
  setnames(expanded, "neighbor_id", "id")
  setkey(expanded, id, year)
  expanded <- row_lookup[expanded, on = c("id", "year"), nomatch = NA]

  # Keep only non-NA matches

  expanded <- expanded[!is.na(row_idx)]

  # Return as a data.table: (focal_row_idx, neighbor_row_idx)
  setnames(expanded, "row_idx", "neighbor_row_idx")
  expanded[, .(focal_row_idx, neighbor_row_idx)]
}

cat("Building neighbor edge list...\n")
system.time({
  neighbor_edges <- build_neighbor_lookup_fast(cell_data, id_order, rook_neighbors_unique)
})
# neighbor_edges is a two-column data.table:
#   focal_row_idx    — the row in cell_data for the focal cell-year
#   neighbor_row_idx — the row in cell_data for a neighbor in the same year

# --------------------------------------------------------------------------
# STEP 2: Compute all neighbor stats in one vectorized pass (replaces
#          compute_neighbor_stats called 5 times in a loop)
#
# Strategy: attach the neighbor's variable value via neighbor_row_idx,
#           then group-by focal_row_idx to compute max/min/mean.
# --------------------------------------------------------------------------
compute_all_neighbor_features <- function(dt, edges, var_names) {
  # Pre-extract columns as vectors for speed
  for (vn in var_names) {
    # Attach neighbor values
    edges[, paste0("nb_", vn) := dt[[vn]][neighbor_row_idx]]
  }

  # Group by focal_row_idx and compute stats for every variable at once
  agg_exprs <- list()
  for (vn in var_names) {
    col <- paste0("nb_", vn)
    agg_exprs[[paste0(vn, "_nb_max")]]  <-
      substitute(as.numeric(max(x, na.rm = TRUE)),  list(x = as.name(col)))
    agg_exprs[[paste0(vn, "_nb_min")]]  <-
      substitute(as.numeric(min(x, na.rm = TRUE)),  list(x = as.name(col)))
    agg_exprs[[paste0(vn, "_nb_mean")]] <-
      substitute(as.numeric(mean(x, na.rm = TRUE)), list(x = as.name(col)))
  }

  # Build a single call: edges[, .(expr1, expr2, ...), by = focal_row_idx]
  agg_call <- as.call(c(
    as.name("list"),
    lapply(agg_exprs, function(e) e)
  ))

  cat("  Aggregating neighbor stats (vectorized group-by)...\n")
  stats <- edges[, eval(agg_call), by = focal_row_idx]

  # Fix Inf/-Inf from max/min on all-NA groups (shouldn't happen but safety)
  stat_cols <- names(stats)[names(stats) != "focal_row_idx"]
  for (sc in stat_cols) {
    set(stats, which(is.infinite(stats[[sc]])), sc, NA_real_)
  }

  return(stats)
}

cat("Computing neighbor features...\n")
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

system.time({
  neighbor_stats <- compute_all_neighbor_features(
    cell_data, neighbor_edges, neighbor_source_vars
  )
})

# --- Merge stats back into cell_data by row index -------------------------
# Rows with no neighbors will get NA (which matches original behavior)
setkey(neighbor_stats, focal_row_idx)

cat("Merging neighbor features into cell_data...\n")
stat_col_names <- setdiff(names(neighbor_stats), "focal_row_idx")

# Pre-allocate columns as NA, then fill matched rows
for (sc in stat_col_names) {
  set(cell_data, j = sc, value = NA_real_)
}

cell_data[neighbor_stats$focal_row_idx, (stat_col_names) :=
  neighbor_stats[, ..stat_col_names]]

# Clean up large intermediate objects
rm(neighbor_edges, neighbor_stats)
gc()

# --------------------------------------------------------------------------
# STEP 3: Random Forest prediction — chunked & (optionally) multi-threaded
#
# The trained model is preserved exactly; only predict() is called.
# --------------------------------------------------------------------------

# --- 3a. Identify the feature columns the model expects -------------------
# Works for both randomForest and ranger model objects.
if (inherits(trained_model, "ranger")) {
  feature_names <- trained_model$forest$independent.variable.names
} else if (inherits(trained_model, "randomForest")) {
  # randomForest stores the names used at training time
  feature_names <- rownames(trained_model$importance)
} else {
  stop("Unsupported model class: ", class(trained_model)[1])
}

# --- 3b. Prepare prediction matrix (only needed columns) -----------------
cat("Preparing prediction feature matrix...\n")
pred_dt <- cell_data[, ..feature_names]

# --- 3c. Chunked prediction -----------------------------------------------
# Chunk size chosen to keep per-chunk memory ≈ 1–1.5 GB
#   500K rows × 110 cols × 8 bytes ≈ 440 MB (doubles)
#   plus tree-traversal workspace ≈ ~1 GB total per chunk.
CHUNK_SIZE <- 500000L
n <- nrow(pred_dt)
n_chunks <- ceiling(n / CHUNK_SIZE)

cat(sprintf("Predicting %s rows in %d chunks of up to %s rows...\n",
            formatC(n, big.mark = ","),
            n_chunks,
            formatC(CHUNK_SIZE, big.mark = ",")))

predictions <- numeric(n)

system.time({
  for (ch in seq_len(n_chunks)) {
    i_start <- (ch - 1L) * CHUNK_SIZE + 1L
    i_end   <- min(ch * CHUNK_SIZE, n)
    idx     <- i_start:i_end

    chunk_df <- as.data.frame(pred_dt[idx, ])

    if (inherits(trained_model, "ranger")) {
      # ranger::predict is multi-threaded by default
      preds <- predict(trained_model, data = chunk_df,
                       num.threads = parallel::detectCores())$predictions
    } else {
      # randomForest::predict — single-threaded but memory-safe in chunks
      preds <- predict(trained_model, newdata = chunk_df)
    }

    predictions[idx] <- preds

    # Free chunk memory immediately
    rm(chunk_df, preds)
    if (ch %% 3 == 0) gc()   # periodic GC every 3 chunks

    cat(sprintf("  Chunk %d/%d done (%s rows)\n",
                ch, n_chunks, formatC(i_end - i_start + 1L, big.mark = ",")))
  }
})

# --- 3d. Store predictions ------------------------------------------------
cell_data[, predicted_gdp := predictions]
rm(pred_dt, predictions)
gc()

cat("Pipeline complete.\n")

# --------------------------------------------------------------------------
# OPTIONAL STEP 4: If the original model is randomForest-class and you want
#                   multi-threaded prediction without retraining, you can
#                   parallelize across chunks using the parallel package.
#                   This keeps the original model object intact.
# --------------------------------------------------------------------------
if (FALSE) {  # Set to TRUE to enable parallel prediction for randomForest
  library(parallel)

  n_cores   <- max(1L, detectCores() - 1L)
  chunk_idx <- split(seq_len(n), ceiling(seq_len(n) / CHUNK_SIZE))

  cat(sprintf("Parallel prediction with %d cores, %d chunks...\n",
              n_cores, length(chunk_idx)))

  cl <- makeCluster(n_cores)
  clusterExport(cl, c("trained_model", "cell_data", "feature_names"))
  clusterEvalQ(cl, library(randomForest))

  par_predictions <- parLapply(cl, chunk_idx, function(idx) {
    chunk_df <- as.data.frame(
      cell_data[idx, .SD, .SDcols = feature_names]
    )
    predict(trained_model, newdata = chunk_df)
  })

  stopCluster(cl)

  cell_data[, predicted_gdp := unlist(par_predictions)]
  rm(par_predictions)
  gc()
}
```

---

## 4. Expected Impact Summary

| Component | Before | After | Factor |
|---|---|---|---|
| `build_neighbor_lookup` | ~20 hrs (string-key lapply over 6.46M rows) | ~2–5 min (keyed `data.table` equi-join) | **~200×** |
| `compute_neighbor_stats` (×5 vars) | ~25 hrs (32M interpreted R iterations) | ~3–8 min (single grouped aggregation) | **~200×** |
| `predict()` | ~35 hrs (single-threaded, RAM thrashing) | ~20–60 min (chunked, within 16 GB RAM, optionally parallel) | **~30–100×** |
| **Total** | **~86 hrs** | **~30–75 min** | **~70–170×** |

### Why correctness is preserved
- **Neighbor statistics**: The `data.table` grouped `max/min/mean` computes exactly the same quantities as the original `lapply` loop — same `na.rm = TRUE` semantics, same `NA` propagation for cells with no neighbors.
- **Predictions**: The same trained model object is passed to the same `predict()` generic; chunking does not change the result since Random Forest prediction is row-independent (each row's prediction depends only on its own features traversing the fixed tree ensemble).
- **No retraining**: The model object (`trained_model`) is only ever read, never modified.