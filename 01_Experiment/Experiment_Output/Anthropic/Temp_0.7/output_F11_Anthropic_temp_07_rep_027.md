 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence clearly points to a different bottleneck: **the spatial neighbor feature engineering pipeline**, specifically `build_neighbor_lookup()` and `compute_neighbor_stats()`.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, and another named-vector lookup (`idx_lookup[neighbor_keys]`). Named vector lookups in R are hash-based but still carry significant per-call overhead at this scale. With ~6.46M iterations, each doing string construction and named lookups, this alone is enormously expensive.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over ~6.46 million entries via `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, `mean`. That's ~32.3 million R-level function invocations with per-element overhead.

3. The `do.call(rbind, result)` at the end of `compute_neighbor_stats` binds ~6.46 million 3-element vectors into a matrix — another expensive operation done 5 times.

4. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with ~6.46M rows and ~110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and typically completes in seconds to a few minutes. Loading the model from disk is a one-time `readRDS()`. Writing predictions is a single vector write. There is no evidence in the code that RF inference is iterated, repeated, or implemented inefficiently.

**Conclusion:** The bottleneck is the O(N) R-level iteration with string operations and named lookups in `build_neighbor_lookup`, compounded by 5× O(N) iteration in `compute_neighbor_stats`. This is what drives the estimated 86+ hour runtime.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup` with a vectorized `data.table` merge.** Instead of building a per-row list of neighbor indices via string key lookups in an `lapply` over 6.46M rows, we:
   - Create a neighbor edge table (source_id → neighbor_id) from the `nb` object.
   - Cross-join it with years to get (source_id, year) → (neighbor_id, year) pairs.
   - Join against the data to resolve row indices in bulk using `data.table` binary search joins.

2. **Replace `compute_neighbor_stats` with a grouped `data.table` aggregation.** Instead of `lapply` over 6.46M list entries, we join the edge table to the data values and compute `max`, `min`, `mean` in a single grouped aggregation per variable.

3. **Preserve the trained Random Forest model** — no retraining. Preserve the original numerical estimand — the same neighbor features (max, min, mean of each variable across rook neighbors) are computed identically, just faster.

---

## Working R Code

```r
library(data.table)

# ─────────────────────────────────────────────────────────────
# 1. Build the directed neighbor edge list from the nb object
#    (done once, replaces build_neighbor_lookup entirely)
# ─────────────────────────────────────────────────────────────
build_neighbor_edges <- function(id_order, nb_obj) {
  # nb_obj is a list of length length(id_order); each element is

  # an integer vector of indices into id_order (0 = no neighbors).
  edges <- rbindlist(lapply(seq_along(nb_obj), function(i) {
    nbrs <- nb_obj[[i]]
    nbrs <- nbrs[nbrs != 0L]
    if (length(nbrs) == 0L) return(NULL)
    data.table(source_id = id_order[i], neighbor_id = id_order[nbrs])
  }))
  edges
}

# ─────────────────────────────────────────────────────────────
# 2. Compute neighbor stats for all variables in one pass
#    (replaces compute_neighbor_stats + the outer for-loop)
# ─────────────────────────────────────────────────────────────
compute_all_neighbor_features <- function(cell_data, id_order, nb_obj,
                                          neighbor_source_vars) {
  dt <- as.data.table(cell_data)

  # Step A: build edge list (source_id -> neighbor_id)
  # This is ~1.37M edges, trivially small.
  edges <- build_neighbor_edges(id_order, nb_obj)

  # Step B: for each year, the edge list is the same, so we cross-join

  # edges with all unique years to get the full (source_id, year, neighbor_id, year) table.
  years <- sort(unique(dt$year))

  # Expand edges × years: ~1.37M edges × 28 years ≈ 38.5M rows

  # This is the key table: for row (source_id, year), its neighbor is (neighbor_id, year).
  edge_year <- edges[, CJ(year = years), by = .(source_id, neighbor_id)]
  # Columns: source_id, neighbor_id, year

  # Step C: attach neighbor values by joining on (neighbor_id, year)
  # We only need the neighbor_source_vars columns from dt.
  # Create a keyed lookup table.
  val_cols <- neighbor_source_vars
  lookup_dt <- dt[, c("id", "year", val_cols), with = FALSE]
  setnames(lookup_dt, "id", "neighbor_id")
  setkeyv(lookup_dt, c("neighbor_id", "year"))
  setkeyv(edge_year, c("neighbor_id", "year"))

  # Merge: attach neighbor variable values to each edge-year row
  merged <- lookup_dt[edge_year, on = .(neighbor_id, year), nomatch = NA]
  # merged has columns: neighbor_id, year, <val_cols>, source_id

  # Step D: grouped aggregation — group by (source_id, year), compute stats
  # Build aggregation expressions dynamically
  agg_exprs <- unlist(lapply(val_cols, function(v) {
    list(
      bquote(max(.(as.name(v)), na.rm = TRUE)),
      bquote(min(.(as.name(v)), na.rm = TRUE)),
      bquote(mean(.(as.name(v)), na.rm = TRUE))
    )
  }))
  agg_names <- unlist(lapply(val_cols, function(v) {
    paste0("neighbor_", v, c("_max", "_min", "_mean"))
  }))
  names(agg_exprs) <- agg_names

  # Evaluate the aggregation
  stats <- merged[,
    lapply(agg_exprs, eval, envir = .SD),
    by = .(source_id, year),
    .SDcols = val_cols
  ]

  # Replace -Inf/Inf from max/min of all-NA groups with NA
  for (col_name in agg_names) {
    set(stats, which(is.infinite(stats[[col_name]])), col_name, NA_real_)
  }

  # Step E: merge stats back onto the main data.table
  setkeyv(stats, c("source_id", "year"))
  setnames(stats, "source_id", "id")
  setkeyv(dt, c("id", "year"))
  dt <- stats[dt, on = .(id, year)]

  # Return as data.frame if the downstream RF predict expects one

  as.data.frame(dt)
}

# ─────────────────────────────────────────────────────────────
# 3. Main pipeline call (replaces build_neighbor_lookup +
#    the for-loop over compute_and_add_neighbor_features)
# ─────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data,
  id_order,
  rook_neighbors_unique,
  neighbor_source_vars
)

# ─────────────────────────────────────────────────────────────
# 4. Random Forest prediction (unchanged — not the bottleneck)
# ─────────────────────────────────────────────────────────────
library(ranger)  # or randomForest — whichever was used to train

rf_model <- readRDS("trained_rf_model.rds")
cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
saveRDS(cell_data, "cell_data_with_predictions.rds")
```

---

## Why This Fixes the Bottleneck

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M R-level iterations with `paste()` + named vector lookups | Eliminated; replaced by a single `data.table` cross-join + keyed merge |
| `compute_neighbor_stats` | 5 × 6.46M `lapply` iterations with per-element subsetting | 1 grouped `data.table` aggregation over ~38.5M rows (C-level internally) |
| `do.call(rbind, ...)` | 5 × binding 6.46M 3-element vectors | Eliminated; result is already a `data.table` |
| String operations | ~6.46M `paste()` + named vector hash lookups | Zero; joins are on integer/numeric keys |
| **Expected runtime** | **86+ hours** | **Minutes** (dominated by the ~38.5M row merge + groupby, which `data.table` handles efficiently in memory on 16 GB RAM) |

The trained Random Forest model is untouched (loaded via `readRDS`, called via `predict`). The numerical estimand is preserved: the same `max`, `min`, and `mean` of each neighbor variable are computed for each (cell, year) combination, just via vectorized joins rather than row-level R loops.