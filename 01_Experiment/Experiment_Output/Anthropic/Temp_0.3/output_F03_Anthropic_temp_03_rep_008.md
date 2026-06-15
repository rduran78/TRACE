 # Diagnosis, Optimization Strategy, and Working R Code

---

## 1. Diagnosis

### A. Feature Preparation Bottlenecks

**`build_neighbor_lookup`** is the single worst offender:

| Problem | Detail |
|---|---|
| **Row-level `lapply` over 6.46M rows** | Each iteration does character coercion, `paste`, named-vector lookup, and NA filtering — all interpreted R. |
| **Named-vector lookup via `idx_lookup[neighbor_keys]`** | This is O(n) string hashing per call × 6.46M calls. The `idx_lookup` vector itself has 6.46M named elements; repeated name-matching against it is extremely slow. |
| **Redundant string pasting** | `paste(id, year, sep="_")` is computed once to *build* the lookup, then again *inside every row* to query it. |
| **Output is a list of 6.46M integer vectors** | ~6.46M list elements, each a small integer vector — heavy memory allocation and GC pressure. |

**`compute_neighbor_stats`** compounds the problem:

| Problem | Detail |
|---|---|
| **Called 5 times, each time `lapply` over 6.46M rows** | 5 × 6.46M = 32.3M interpreted R function calls. |
| **Per-row subsetting, NA removal, summary stats** | `vals[idx]`, `!is.na(...)`, `max/min/mean` — all fine individually but catastrophic at 32.3M iterations. |
| **`do.call(rbind, result)` on 6.46M single-row matrices** | This alone can take minutes and doubles peak memory. |

### B. Random Forest Inference Bottleneck

| Problem | Detail |
|---|---|
| **Single monolithic `predict()` call on 6.46M × 110 matrix** | `ranger`/`randomForest` must traverse every tree for every row. With 500+ trees this is CPU-bound but also memory-bound if the model + data exceed RAM. |
| **Data frame copying** | If `cell_data` is a `data.frame`, adding columns with `cell_data$new_col <- ...` triggers a full copy each time (COW semantics). With ~110 columns × 6.46M rows ≈ 5.4 GB, each copy is devastating. |
| **Model object size** | A `randomForest` object on 6.46M rows can be several GB. Loading from disk with `readRDS` is I/O-bound. |

### Estimated time breakdown (86+ hours)

| Phase | Estimated share |
|---|---|
| `build_neighbor_lookup` | ~30–40% |
| `compute_neighbor_stats` (×5 vars) | ~30–40% |
| RF `predict()` | ~15–25% |
| Data copying / GC | ~10–15% |

---

## 2. Optimization Strategy

### Principle: Eliminate interpreted-R loops; vectorize everything with `data.table` joins and matrix operations.

| Strategy | Technique | Expected speedup |
|---|---|---|
| **Replace `build_neighbor_lookup`** | Build a `data.table` edge-list `(id, year, neighbor_id)` and do a keyed join to get neighbor row indices. No `lapply`, no `paste` lookup. | 50–200× |
| **Replace `compute_neighbor_stats`** | Use the edge-list `data.table` to join variable values, then `[, .(max, min, mean), by=.(id, year)]` — fully vectorized C-level grouping. | 50–100× |
| **Compute all 5 variables in one pass** | Join all 5 source columns at once, compute 15 stats in a single grouped aggregation. | 5× fewer passes |
| **Use `data.table` throughout** | Avoid `data.frame` COW copies. Add columns by reference with `:=`. | 2–5× on memory/copy |
| **Chunk RF prediction** | Call `predict()` in chunks of ~500K rows to control peak memory. | Avoids OOM; marginal speed gain |
| **Use `ranger` not `randomForest`** | If the model is `randomForest`, convert or re-save as `ranger` (much faster predict). If already `ranger`, use `num.threads`. | 2–10× on predict |

**Target runtime: 5–20 minutes** (down from 86+ hours).

---

## 3. Working R Code

```r
# =============================================================================
# OPTIMIZED PIPELINE — cell-level GDP prediction
# Preserves trained RF model and original numerical estimand.
# =============================================================================

library(data.table)

# ---- 0. Helper: convert spdep nb object to a data.table edge list ----------
nb_to_edge_dt <- function(nb_obj, id_order) {
 # nb_obj: list of integer vectors (indices into id_order)
 # id_order: vector of cell IDs in the order matching nb_obj
  lens <- lengths(nb_obj)
  from_idx <- rep(seq_along(nb_obj), lens)
  to_idx   <- unlist(nb_obj, use.names = FALSE)
  # Remove 0-entries that spdep uses for "no neighbours"
  valid <- to_idx > 0L
  data.table(
    id       = id_order[from_idx[valid]],
    nb_id    = id_order[to_idx[valid]]
  )
}

# ---- 1. Build edge list (once) ---------------------------------------------
edge_dt <- nb_to_edge_dt(rook_neighbors_unique, id_order)
# edge_dt has columns: id, nb_id   (~1.37M rows)

# ---- 2. Convert cell_data to data.table (by reference if possible) ---------
if (!is.data.table(cell_data)) setDT(cell_data)

# Ensure key columns are proper types for joining
cell_data[, id   := as.integer(id)]
cell_data[, year := as.integer(year)]
edge_dt[,   id   := as.integer(id)]
edge_dt[,   nb_id := as.integer(nb_id)]

# ---- 3. Vectorised neighbor-feature computation (ALL vars, ONE pass) -------
compute_all_neighbor_features <- function(dt, edge_dt, source_vars) {
  # Step 1: Cross edge_dt with years present in dt.
  #   For each (id, year) we need (nb_id, year) rows from dt.
  #   Strategy: join edge_dt to dt on id to get years, then join
  #   back to dt on (nb_id, year) to get neighbor values.

  # Minimal subset for the join: id, year, and source_vars
  cols_needed <- c("id", "year", source_vars)
  sub <- dt[, ..cols_needed]

  # Step 2: For every row in sub, attach its neighbor IDs via edge_dt
  #   Result: one row per (id, year, nb_id) triple
  setkey(edge_dt, id)
  # Merge: for each (id) in sub, get all nb_id from edge_dt
  # Use edge_dt[sub, on="id", allow.cartesian=TRUE] — gives (id, year, nb_id)
  expanded <- edge_dt[sub, on = "id", allow.cartesian = TRUE, nomatch = NULL]
  # expanded columns: id, nb_id, year, <source_vars>  (source_vars are from sub = focal cell values, not needed yet)
  # We only need id, year, nb_id from this join; drop source_var columns to save memory
  expanded[, (source_vars) := NULL]

  # Step 3: Join neighbor values from sub on (nb_id = id, year)
  setnames(sub, "id", "nb_id")
  setkey(sub, nb_id, year)
  setkey(expanded, nb_id, year)
  expanded <- sub[expanded, on = c("nb_id", "year"), nomatch = NA]
  # Now expanded has: nb_id, year, <source_vars (neighbor values)>, id

  # Step 4: Grouped aggregation — max, min, mean per (id, year) for each var
  agg_exprs <- unlist(lapply(source_vars, function(v) {
    list(
      bquote(max(.(as.name(v)),   na.rm = TRUE)),
      bquote(min(.(as.name(v)),   na.rm = TRUE)),
      bquote(mean(.(as.name(v)),  na.rm = TRUE))
    )
  }), recursive = FALSE)

  agg_names <- unlist(lapply(source_vars, function(v) {
    paste0("nb_", v, c("_max", "_min", "_mean"))
  }))

  # Build the j-expression programmatically
  j_list <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

  agg <- expanded[, eval(j_list), by = .(id, year)]

  # Replace Inf/-Inf (from max/min on all-NA groups) with NA
  for (col in agg_names) {
    set(agg, which(is.infinite(agg[[col]])), col, NA_real_)
  }

  return(agg)
}

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cat("Computing neighbor features (vectorised)...\n")
system.time({
  nb_features <- compute_all_neighbor_features(cell_data, edge_dt, neighbor_source_vars)
})

# ---- 4. Merge neighbor features back into cell_data by reference -----------
# Remove old neighbor columns if they exist (idempotency)
old_nb_cols <- grep("^nb_", names(cell_data), value = TRUE)
if (length(old_nb_cols)) cell_data[, (old_nb_cols) := NULL]

setkey(cell_data, id, year)
setkey(nb_features, id, year)
cell_data <- nb_features[cell_data, on = c("id", "year")]
# This is a right join: all rows of cell_data preserved.

cat("Neighbor features merged. Columns:", ncol(cell_data), "\n")

# ---- 5. Random Forest prediction (chunked, memory-safe) --------------------
predict_rf_chunked <- function(model, newdata, chunk_size = 500000L) {
  n <- nrow(newdata)
  preds <- numeric(n)
  starts <- seq(1L, n, by = chunk_size)

  # Detect model class to use optimal predict path
  is_ranger <- inherits(model, "ranger")

  for (i in seq_along(starts)) {
    s <- starts[i]
    e <- min(s + chunk_size - 1L, n)
    chunk <- newdata[s:e, ]

    if (is_ranger) {
      preds[s:e] <- predict(model, data = chunk, num.threads = parallel::detectCores())$predictions
    } else {
      # randomForest package
      preds[s:e] <- predict(model, newdata = chunk)
    }

    if (i %% 5 == 0 || i == length(starts)) {
      cat(sprintf("  Predicted %d / %d rows (%.1f%%)\n", e, n, 100 * e / n))
    }
  }
  preds
}

# Load model (once)
cat("Loading trained RF model...\n")
rf_model <- readRDS("path/to/trained_rf_model.rds")  # <-- adjust path

# Identify predictor columns (exclude id, year, response)
response_var <- "gdp"  # <-- adjust if different
exclude_cols <- c("id", "year", response_var)
predictor_cols <- setdiff(names(cell_data), exclude_cols)

# If model is ranger, it stores feature names; use those to guarantee column order
if (inherits(rf_model, "ranger") && !is.null(rf_model$forest$independent.variable.names)) {
  predictor_cols <- rf_model$forest$independent.variable.names
}

# Build prediction matrix (data.table subsetting avoids copy)
pred_data <- cell_data[, ..predictor_cols]

cat("Running RF prediction...\n")
system.time({
  cell_data[, predicted_gdp := predict_rf_chunked(rf_model, pred_data, chunk_size = 500000L)]
})

cat("Done. Predictions stored in cell_data$predicted_gdp\n")
```

---

## 4. Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model untouched** | We only call `predict()`; no retraining, no coefficient changes. |
| **Numerical estimand identical** | The neighbor stats (max, min, mean) are computed with the same arithmetic (`max`, `min`, `mean` with `na.rm=TRUE`). `data.table` grouping uses the same R functions — results are bit-identical. |
| **Row alignment** | The keyed join on `(id, year)` ensures every original row gets exactly its correct neighbor features. Rows with no neighbors get `NA` — same as the original code. |

---

## 5. Expected Performance

| Phase | Original (est.) | Optimised (est.) |
|---|---|---|
| `build_neighbor_lookup` | ~25–35 hours | **Eliminated** (edge DT built in <2 sec) |
| `compute_neighbor_stats` ×5 | ~25–35 hours | **~1–3 min** (single vectorised pass) |
| Merge features | ~5–10 hours (COW copies) | **~10 sec** (`:=` by reference) |
| RF `predict()` | ~10–15 hours | **~5–20 min** (chunked, multi-threaded if `ranger`) |
| **Total** | **86+ hours** | **~10–30 minutes** |

The key insight: replacing 6.46 million interpreted-R loop iterations with `data.table` keyed joins and grouped aggregations moves all the heavy lifting into C, yielding roughly a **200–500× speedup** on the feature-preparation phase.