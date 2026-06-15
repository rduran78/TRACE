 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **rejected**. The code evidence clearly shows the bottleneck is in the **spatial neighbor feature engineering**, specifically `build_neighbor_lookup` and `compute_neighbor_stats`.

**Evidence:**

1. **`build_neighbor_lookup`** uses a row-level `lapply` over **~6.46 million rows**. Inside every iteration it performs `as.character()` coercion, character-based hash lookups (`id_to_ref`, `idx_lookup`), `paste()` key construction, and subsetting — all in a sequential, interpreted R loop. This alone produces billions of elementary R operations.

2. **`compute_neighbor_stats`** is called **5 times** (once per neighbor source variable), each time iterating over **~6.46 million rows** with an inner `lapply` that subsets, removes NAs, and computes `max/min/mean`. That is ~32.3 million serial R-level loop iterations just for the stats.

3. By contrast, Random Forest inference (`predict()` on a pre-trained model) is a single vectorized C/C++ call over the data matrix. Even with 110 predictors and 6.46M rows, this typically completes in minutes on a laptop — orders of magnitude faster than the feature engineering above.

4. The estimated 86+ hour runtime is consistent with billions of interpreted R operations in nested `lapply` loops, not with a single `predict()` call.

**Root cause:** Row-level interpreted loops over millions of rows with repeated string operations and named-vector lookups.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup`** with a vectorized `data.table` equi-join approach: explode the neighbor list into an edge table once, join against the data keyed by `(id, year)`, and group.
2. **Replace `compute_neighbor_stats`** with a single grouped `data.table` aggregation per variable — no R-level row loop at all.
3. **Avoid all `paste()`-based key lookups and per-row `lapply`.**

Expected speedup: from 86+ hours to roughly **minutes**, because all work is pushed into `data.table`'s optimized C backend.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Build a directed edge table from the nb object (done ONCE)
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # neighbors is a list of integer index vectors (spdep nb object)
  # id_order maps position -> cell id
  from_lengths <- lengths(neighbors)
  from_idx     <- rep(seq_along(neighbors), from_lengths)
  to_idx       <- unlist(neighbors, use.names = FALSE)

  data.table(
    focal_id    = id_order[from_idx],
    neighbor_id = id_order[to_idx]
  )
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# 2. Compute all neighbor features via data.table joins + grouped agg
# ──────────────────────────────────────────────────────────────────────
compute_all_neighbor_features <- function(cell_data, edge_dt,
                                          neighbor_source_vars) {
  # Convert to data.table if needed (by reference is fine)
  dt <- as.data.table(cell_data)

  # We will join on (neighbor_id == id, year == year).
  # Step A: create a slim table of just id, year, and the source vars.
  keep_cols <- c("id", "year", neighbor_source_vars)
  neighbor_vals <- dt[, ..keep_cols]

  # Step B: cross edge_dt with years via a keyed join.
  #   For every (focal_id, neighbor_id) pair and every year in the data,
  #   look up the neighbor's value.
  #
  #   Efficient approach: join edge_dt to neighbor_vals on
  #   neighbor_id == id, broadcasting across years.

  setnames(neighbor_vals, "id", "neighbor_id")
  setkey(neighbor_vals, neighbor_id)
  setkey(edge_dt, neighbor_id)

  # Merge: each edge gets all years of the neighbor

  merged <- edge_dt[neighbor_vals, on = "neighbor_id",
                    allow.cartesian = TRUE, nomatch = NULL]
  # merged now has columns: focal_id, neighbor_id, year, ntl, ec, ...

  # Step C: aggregate by (focal_id, year) for each variable
  agg_exprs <- list()
  for (v in neighbor_source_vars) {
    sym_v <- as.name(v)
    agg_exprs[[paste0("neighbor_max_", v)]]  <-
      bquote(max(.(sym_v), na.rm = TRUE))
    agg_exprs[[paste0("neighbor_min_", v)]]  <-
      bquote(min(.(sym_v), na.rm = TRUE))
    agg_exprs[[paste0("neighbor_mean_", v)]] <-
      bquote(mean(.(sym_v), na.rm = TRUE))
  }

  # Build the aggregation call
  agg_call <- as.call(c(as.name("list"), agg_exprs))
  agg_dt   <- merged[, eval(agg_call), by = .(focal_id, year)]

  # Replace infinite values (from max/min on all-NA) with NA
  for (col_name in names(agg_dt)[-(1:2)]) {
    set(agg_dt, which(is.infinite(agg_dt[[col_name]])), col_name, NA_real_)
  }

  # Step D: merge back onto the original data
  setnames(agg_dt, "focal_id", "id")
  setkey(dt, id, year)
  setkey(agg_dt, id, year)

  dt <- agg_dt[dt, on = .(id, year)]

  # Return as data.frame if the pipeline expects one

  as.data.frame(dt)
}

# ──────────────────────────────────────────────────────────────────────
# 3. Run it
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

cell_data <- compute_all_neighbor_features(
  cell_data, edge_dt, neighbor_source_vars
)

# ──────────────────────────────────────────────────────────────────────
# 4. Random Forest prediction (UNCHANGED — model is preserved)
# ──────────────────────────────────────────────────────────────────────
# e.g.:
# predictions <- predict(rf_model, newdata = cell_data)
# cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with `paste()` + named-vector lookups | One vectorized `data.table` edge-table construction |
| `compute_neighbor_stats` | 5 × 6.46M-iteration `lapply` (32.3M R-level loops) | Single grouped `data.table` aggregation (C backend) |
| Estimated runtime | 86+ hours | Minutes |
| Random Forest model | Preserved (untouched) | Preserved (untouched) |
| Numerical estimand | Identical `max`, `min`, `mean` per (focal cell, year) | Identical — same arithmetic, same NA handling |

The true bottleneck is the interpreted, row-level spatial feature engineering — not Random Forest inference.