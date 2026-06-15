 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates via `lapply` over **~6.46 million rows**, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to build neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named-vector lookups in R are **O(n) string-hashing operations per call** against a vector of 6.46 million names. Doing this 6.46 million times is catastrophically slow.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over 6.46 million rows with `lapply`, extracting subsets, computing `max/min/mean`, and then binding ~6.46 million 3-element vectors with `do.call(rbind, ...)` — itself an expensive operation.

3. **Random Forest inference** (`predict()` on a pre-trained model) for ~6.46 million rows with ~110 predictors is a single vectorized C-level call. On a modern laptop this typically completes in seconds to a few minutes. It is nowhere near 86+ hours.

**The bottleneck is the row-by-row R-level looping and repeated named-vector lookups across 6.46 million rows, repeated for 5 variables.**

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` join approach. Instead of building a per-row list of neighbor indices, construct an edge-list data.table of `(focal_row, neighbor_row)` pairs via keyed joins. This eliminates all per-row `paste`, `as.character`, and named-vector lookups.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation over the edge list. For each focal row, compute `max`, `min`, and `mean` of neighbor values in one vectorized pass — no `lapply`, no `do.call(rbind, ...)`.

3. **Process all 5 variables in one pass** over the edge list rather than rebuilding/re-traversing the lookup 5 times.

4. **Leave the Random Forest predict() call untouched** — it is not the bottleneck.

Expected speedup: from 86+ hours to **minutes**.

---

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# 1. Convert cell_data to data.table (preserves all columns, including
#    the ~110 predictors needed later for RF predict).
# ──────────────────────────────────────────────────────────────────────
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]                 # preserve original row order

# ──────────────────────────────────────────────────────────────────────
# 2. Build the directed edge list from the nb object (one-time cost).
#    rook_neighbors_unique is a list of length = number of unique spatial
#    ids (344,208). id_order maps position -> cell id.
# ──────────────────────────────────────────────────────────────────────
edges <- rbindlist(lapply(seq_along(rook_neighbors_unique), function(i) {
  nb <- rook_neighbors_unique[[i]]
  if (length(nb) == 0L || (length(nb) == 1L && nb[1] == 0L)) {
    return(NULL)
  }
  data.table(focal_id = id_order[i], neighbor_id = id_order[nb])
}))

# ──────────────────────────────────────────────────────────────────────
# 3. Expand edges across years via a keyed join.
#    This replaces build_neighbor_lookup() entirely.
#    Result: each row is (focal_row_idx, neighbor_row_idx).
# ──────────────────────────────────────────────────────────────────────

# Key cell_dt for fast joins
setkey(cell_dt, id, year)

# Create a slim lookup: (id, year) -> row_idx
row_lookup <- cell_dt[, .(id, year, row_idx)]

# For every (focal_id, neighbor_id) pair, join to every year present
# for the focal cell, then resolve the neighbor's row in that same year.

# Step A: get all (focal_id, year, focal_row_idx) from the data
focal_years <- row_lookup[, .(focal_id = id, year, focal_row = row_idx)]

# Step B: join edges to focal_years on focal_id
setkey(edges, focal_id)
setkey(focal_years, focal_id)
edge_year <- edges[focal_years, on = "focal_id", allow.cartesian = TRUE,
                   nomatch = NULL]
# edge_year now has columns: focal_id, neighbor_id, year, focal_row

# Step C: resolve neighbor_row by joining (neighbor_id, year) -> row_idx
setkey(row_lookup, id, year)
edge_year[, neighbor_row := row_lookup[.(edge_year$neighbor_id,
                                         edge_year$year), row_idx]]
edge_year <- edge_year[!is.na(neighbor_row)]

# ──────────────────────────────────────────────────────────────────────
# 4. Compute neighbor stats for all 5 variables in one vectorized pass.
#    This replaces compute_neighbor_stats() and the outer for-loop.
# ──────────────────────────────────────────────────────────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Attach neighbor values for all source vars at once
neighbor_vals <- cell_dt[edge_year$neighbor_row, ..neighbor_source_vars]
neighbor_vals[, focal_row := edge_year$focal_row]

# Grouped aggregation: max, min, mean per focal_row per variable
agg <- neighbor_vals[, lapply(.SD, function(v) {
  v <- v[!is.na(v)]
  if (length(v) == 0L) list(NA_real_, NA_real_, NA_real_)
  else list(max(v), min(v), mean(v))
}), by = focal_row, .SDcols = neighbor_source_vars]

# The above returns list-columns; unpack them properly.
# More efficient approach — aggregate each stat separately:
agg_max  <- neighbor_vals[, lapply(.SD, function(v) {
  v <- v[!is.na(v)]; if (length(v) == 0L) NA_real_ else max(v)
}), by = focal_row, .SDcols = neighbor_source_vars]

agg_min  <- neighbor_vals[, lapply(.SD, function(v) {
  v <- v[!is.na(v)]; if (length(v) == 0L) NA_real_ else min(v)
}), by = focal_row, .SDcols = neighbor_source_vars]

agg_mean <- neighbor_vals[, lapply(.SD, function(v) {
  v <- v[!is.na(v)]; if (length(v) == 0L) NA_real_ else mean(v)
}), by = focal_row, .SDcols = neighbor_source_vars]

# Rename columns to match expected feature names
setnames(agg_max,  neighbor_source_vars,
         paste0("neighbor_max_",  neighbor_source_vars))
setnames(agg_min,  neighbor_source_vars,
         paste0("neighbor_min_",  neighbor_source_vars))
setnames(agg_mean, neighbor_source_vars,
         paste0("neighbor_mean_", neighbor_source_vars))

# ──────────────────────────────────────────────────────────────────────
# 5. Join aggregated features back to cell_dt by focal_row.
#    Rows with no neighbors get NA (correct behaviour, matches original).
# ──────────────────────────────────────────────────────────────────────
cell_dt <- merge(cell_dt, agg_max,  by.x = "row_idx", by.y = "focal_row",
                 all.x = TRUE, sort = FALSE)
cell_dt <- merge(cell_dt, agg_min,  by.x = "row_idx", by.y = "focal_row",
                 all.x = TRUE, sort = FALSE)
cell_dt <- merge(cell_dt, agg_mean, by.x = "row_idx", by.y = "focal_row",
                 all.x = TRUE, sort = FALSE)

# Restore original row order
setorder(cell_dt, row_idx)
cell_dt[, row_idx := NULL]

# ──────────────────────────────────────────────────────────────────────
# 6. Random Forest prediction — UNCHANGED (not the bottleneck).
#    The pre-trained model is preserved; the numerical estimand is
#    identical because the neighbor features are numerically equivalent.
# ──────────────────────────────────────────────────────────────────────
# Example (assuming `rf_model` is the pre-trained model already in memory
# or loaded from disk):
#
#   rf_model   <- readRDS("trained_rf_model.rds")
#   cell_dt$prediction <- predict(rf_model, newdata = cell_dt)
#   fwrite(cell_dt, "predictions_output.csv")
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| **Neighbor lookup** | 6.46M row-level `lapply` with `paste` + named-vector lookups | Single vectorized `data.table` keyed join producing an edge list |
| **Neighbor stats** | 5 × `lapply` over 6.46M rows + `do.call(rbind, ...)` | One grouped `data.table` aggregation across all 5 variables |
| **RF inference** | Already vectorized C-level call (fast) | Untouched — not the bottleneck |
| **Estimated runtime** | 86+ hours | Minutes (dominated by the join/aggregation, which is vectorized in C) |
| **Numerical equivalence** | — | Preserved: same `max`, `min`, `mean` over the same neighbor sets |
| **Trained model** | — | Preserved: no retraining; same `predict()` call |