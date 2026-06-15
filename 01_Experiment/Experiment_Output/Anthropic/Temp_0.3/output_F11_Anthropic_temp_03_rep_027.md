 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates tens of millions of string operations and named lookups.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's ~32.3 million R-level function calls across the 5 variables.

3. **Random Forest inference**, by contrast, is a single call to `predict()` on a pre-loaded model object. Even with ~6.46M rows and ~110 predictors, `predict.randomForest` (or `predict.ranger`) is implemented in C/C++ and runs in minutes, not hours. Loading the model from disk is a one-time `readRDS()`. Writing predictions is a single vectorized write. None of these are 86-hour operations.

**The bottleneck is the row-level R `lapply` loops over 6.46 million rows doing string construction and named-vector lookups, repeated across multiple variables.**

## Optimization Strategy

1. **Replace `build_neighbor_lookup()`** with a fully vectorized `data.table` merge/join approach. Instead of looping row-by-row, explode the neighbor list into an edge table (`cell_id → neighbor_id`), join it with the panel data on `(neighbor_id, year)` to get row indices, and group by the source row.

2. **Replace `compute_neighbor_stats()`** with a single grouped `data.table` aggregation per variable (or all variables at once), computing `max`, `min`, and `mean` in one vectorized pass.

3. **Eliminate per-row `lapply` entirely.** All string pasting, named lookups, and per-row subsetting are replaced by indexed joins and grouped aggregations — operations `data.table` executes in optimized C.

This should reduce the 86+ hour runtime to **minutes**.

## Working R Code

```r
library(data.table)

# ──────────────────────────────────────────────────────────────────────
# STEP 0: Convert panel data to data.table (if not already)
# ──────────────────────────────────────────────────────────────────────
setDT(cell_data)

# Ensure there is a row index we can reference
cell_data[, .row_id := .I]

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Build an edge table from the nb object (one-time, vectorized)
#
# rook_neighbors_unique is a list of length = number of spatial cells.
# id_order is the vector mapping list position → cell id.
# Edge table: each row is (cell_id, neighbor_id).
# ──────────────────────────────────────────────────────────────────────
build_edge_table <- function(id_order, neighbors) {
  # Number of neighbors per cell
  n_neighbors <- lengths(neighbors)
  
  # Source cell ids, repeated by number of neighbors
  from_id <- rep(id_order, times = n_neighbors)
  
  # Neighbor cell ids (unlist the nb list, index into id_order)
  to_idx  <- unlist(neighbors, use.names = FALSE)
  to_id   <- id_order[to_idx]
  
  data.table(cell_id = from_id, neighbor_id = to_id)
}

edge_dt <- build_edge_table(id_order, rook_neighbors_unique)

# ──────────────────────────────────────────────────────────────────────
# STEP 2: Vectorized neighbor feature computation
#
# For each (cell_id, year) row in cell_data, we need the max, min, mean
# of each neighbor source variable across its rook neighbors in the
# same year.
#
# Strategy:
#   a) Join edge_dt with cell_data to get (source_row, neighbor_row) pairs
#      matched on year.
#   b) Group by source_row and aggregate.
# ──────────────────────────────────────────────────────────────────────

neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Create a keyed lookup: (id, year) → .row_id + variable values
# We only need the neighbor source vars for the neighbor rows.
lookup_cols <- c("id", "year", neighbor_source_vars)
neighbor_vals_dt <- cell_data[, ..lookup_cols]
neighbor_vals_dt[, neighbor_id := id]  # alias for joining
setkey(neighbor_vals_dt, neighbor_id, year)

# Source side: (cell_id, year, .row_id) for joining with edge_dt
source_dt <- cell_data[, .(cell_id = id, year, .row_id)]
setkey(source_dt, cell_id)

# Join source rows with edge table to get (source .row_id, neighbor_id, year)
# This is the key step: for every row in cell_data, find its neighbor cell ids
# then we will look up the neighbor's values in the same year.
source_edges <- edge_dt[source_dt, on = .(cell_id), allow.cartesian = TRUE,
                        nomatch = 0L]
# source_edges now has columns: cell_id, neighbor_id, year, .row_id

# Join with neighbor values on (neighbor_id, year)
setkey(source_edges, neighbor_id, year)
joined <- neighbor_vals_dt[source_edges, on = .(neighbor_id, year),
                           nomatch = NA_integer_]
# joined has: neighbor_id, year, ntl, ec, pop_density, def, usd_est_n2,
#             cell_id, .row_id

# ──────────────────────────────────────────────────────────────────────
# STEP 3: Grouped aggregation — compute max, min, mean per source row
# ──────────────────────────────────────────────────────────────────────

# Build aggregation expressions dynamically
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

names(agg_exprs) <- agg_names

# Perform the grouped aggregation in one pass
agg_result <- joined[, eval(as.call(c(as.name("list"), agg_exprs))),
                     by = .row_id]

# Replace Inf/-Inf (from max/min on all-NA groups) with NA
inf_cols <- grep("_max$|_min$", names(agg_result), value = TRUE)
for (col in inf_cols) {
  set(agg_result, which(is.infinite(agg_result[[col]])), col, NA_real_)
}

# ──────────────────────────────────────────────────────────────────────
# STEP 4: Merge aggregated neighbor features back into cell_data
# ──────────────────────────────────────────────────────────────────────
setkey(agg_result, .row_id)
setkey(cell_data, .row_id)

cell_data <- agg_result[cell_data, on = .(.row_id)]

# Rows with no neighbors will have NA for all neighbor features (correct).
# Clean up helper column
cell_data[, .row_id := NULL]

# ──────────────────────────────────────────────────────────────────────
# STEP 5: Random Forest prediction (unchanged — not the bottleneck)
# ──────────────────────────────────────────────────────────────────────
# library(ranger)  # or randomForest
# rf_model <- readRDS("trained_rf_model.rds")
# cell_data$prediction <- predict(rf_model, data = cell_data)$predictions
# fwrite(cell_data, "predictions_output.csv")
```

## Summary

| Component | Colleague's claim | Actual evidence |
|---|---|---|
| `build_neighbor_lookup` | Not mentioned | **~6.46M row-level `lapply` with string ops and named lookups — major bottleneck** |
| `compute_neighbor_stats` | Not mentioned | **Called 5×, each time ~6.46M `lapply` iterations — major bottleneck** |
| RF `predict()` | "Main bottleneck" | Single vectorized C/C++ call — **not the bottleneck** |

The optimized code replaces all row-level R loops with vectorized `data.table` joins and grouped aggregations, reducing the estimated runtime from **86+ hours to minutes** while preserving the trained Random Forest model and the original numerical estimand (neighbor max, min, mean are computed identically).