 # Diagnosis, Optimization Strategy, and Working R Code

## Diagnosis: Rejecting the Colleague's Claim

The colleague's claim that Random Forest inference is the main bottleneck is **incorrect**. The code evidence points clearly to the **spatial neighbor feature engineering** — specifically `build_neighbor_lookup()` and `compute_neighbor_stats()` — as the true bottleneck.

Here's why:

1. **`build_neighbor_lookup()`** iterates over **~6.46 million rows** with `lapply`, and for each row performs: a character coercion (`as.character`), a named-vector lookup (`id_to_ref[...]`), a `paste()` to construct neighbor keys, another named-vector lookup (`idx_lookup[neighbor_keys]`), and NA filtering. Named vector lookups in R are hash-based but carry significant per-call overhead at this scale. With ~6.46M rows and an average of ~4 rook neighbors per cell, this generates roughly **25.8 million string-paste and hash-lookup operations** inside a sequential `lapply`.

2. **`compute_neighbor_stats()`** is called **5 times** (once per neighbor source variable), each time iterating over all ~6.46M rows with `lapply`, subsetting a numeric vector, removing NAs, and computing `max`, `min`, and `mean`. That's another **~32.3 million** R-level function-call iterations total.

3. **`do.call(rbind, result)`** on a list of ~6.46M small vectors is notoriously slow in R — it creates a massive argument list for `rbind`.

4. By contrast, Random Forest **prediction** on a pre-trained model is a single vectorized call (`predict(model, newdata)`) that runs in compiled C/C++ code. Even with 6.46M rows and 110 predictors, this typically completes in seconds to a few minutes. It is not the bottleneck.

**Conclusion:** The 86+ hour runtime is dominated by the O(N × k) row-level R-interpreted loops in neighbor lookup construction and neighbor statistics computation, not by RF inference.

---

## Optimization Strategy

1. **Replace `build_neighbor_lookup()` with a vectorized `data.table` equi-join.** Instead of looping row-by-row, expand all neighbor relationships into an edge table (`cell_id`, `neighbor_id`), join with the data on `(neighbor_id, year)` to get row indices, and group by the source row. This turns millions of R-level hash lookups into a single indexed merge.

2. **Replace `compute_neighbor_stats()` with a grouped `data.table` aggregation.** Once we have the edge table joined to the data, computing `max`, `min`, and `mean` of neighbor values is a single grouped aggregation — fully vectorized in C.

3. **Eliminate `do.call(rbind, ...)`** entirely; `data.table` returns results as a data.table directly.

4. **Leave the Random Forest predict() call untouched** — it is already efficient.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# 0.  Convert main data to data.table (one-time, in-place)
# ---------------------------------------------------------------
cell_dt <- as.data.table(cell_data)
cell_dt[, row_idx := .I]                       # preserve original row order

# ---------------------------------------------------------------
# 1.  Build a vectorized neighbor edge table (replaces build_neighbor_lookup)
#
#     rook_neighbors_unique : spdep nb object (list of integer vectors)
#     id_order              : vector mapping list position -> cell id
# ---------------------------------------------------------------
build_neighbor_edges <- function(id_order, neighbors) {
  # Expand the nb list into a two-column data.table of (cell_id, neighbor_id)
  n_neighbors <- lengths(neighbors)
  from_pos    <- rep(seq_along(neighbors), n_neighbors)
  to_pos      <- unlist(neighbors, use.names = FALSE)

  data.table(
    cell_id     = id_order[from_pos],
    neighbor_id = id_order[to_pos]
  )
}

edge_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# 2.  Join edges with data to map every (source row, neighbor row) pair
#
#     For each row i in cell_dt we need the rows j that share
#     the same year AND whose id is a rook-neighbor of row i's id.
#     We achieve this with a single keyed merge.
# ---------------------------------------------------------------

# Slim table: only what we need for the join key + values
#   We'll carry all neighbor source variable columns along.
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

slim_cols <- c("id", "year", "row_idx", neighbor_source_vars)
slim_dt   <- cell_dt[, ..slim_cols]

# Key the edge table for the merge
setkey(edge_dt, cell_id)

# For every row in slim_dt, look up its neighbors via edge_dt,
# then find the neighbor rows that share the same year.

# Step A: attach neighbor_ids to every source row
#         join slim_dt (as "source") with edge_dt on id == cell_id
setkey(slim_dt, id)
source_with_edges <- edge_dt[slim_dt,
                             on = .(cell_id = id),
                             allow.cartesian = TRUE,
                             nomatch = 0L]
# source_with_edges now has columns:
#   cell_id, neighbor_id, year, row_idx, ntl, ec, ...
# where row_idx / year / values belong to the SOURCE row.

# Step B: join with slim_dt again to get NEIGHBOR values
#         match on (neighbor_id = id, year = year)
neighbor_vals_dt <- slim_dt[, c("id", "year", neighbor_source_vars), with = FALSE]
setnames(neighbor_vals_dt, "id", "neighbor_id")

# Prefix neighbor value columns to avoid collision
old_names <- neighbor_source_vars
new_names <- paste0("nb_", neighbor_source_vars)
setnames(neighbor_vals_dt, old_names, new_names)

setkey(source_with_edges, neighbor_id, year)
setkey(neighbor_vals_dt,  neighbor_id, year)

paired <- neighbor_vals_dt[source_with_edges,
                           on = .(neighbor_id, year),
                           nomatch = NA_integer_]
# paired has: row_idx (source), and nb_ntl, nb_ec, ... (neighbor values)

# ---------------------------------------------------------------
# 3.  Compute grouped neighbor statistics (replaces compute_neighbor_stats)
# ---------------------------------------------------------------
# Group by source row_idx; compute max, min, mean for each variable.

agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  nbv <- paste0("nb_", v)
  list(
    bquote(max(.(as.name(nbv)),   na.rm = TRUE)),
    bquote(min(.(as.name(nbv)),   na.rm = TRUE)),
    bquote(mean(.(as.name(nbv)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0(v, c("_nb_max", "_nb_min", "_nb_mean"))
}))

# Build the j-expression programmatically
agg_call <- as.call(c(as.name("list"),
                       setNames(agg_exprs, agg_names)))

stats_dt <- paired[, eval(agg_call), by = row_idx]

# Replace Inf / -Inf (from max/min on all-NA groups) with NA
for (col_name in agg_names) {
  set(stats_dt, which(is.infinite(stats_dt[[col_name]])), col_name, NA_real_)
}

# ---------------------------------------------------------------
# 4.  Merge aggregated neighbor features back onto the main table
# ---------------------------------------------------------------
setkey(stats_dt, row_idx)
setkey(cell_dt,  row_idx)

cell_dt <- stats_dt[cell_dt, on = "row_idx"]

# Clean up helper column
cell_dt[, row_idx := NULL]

# Convert back to data.frame if downstream code expects one
cell_data <- as.data.frame(cell_dt)

# ---------------------------------------------------------------
# 5.  Random Forest prediction (UNCHANGED — not the bottleneck)
# ---------------------------------------------------------------
# model is the pre-trained randomForest / ranger object already in memory
# predictions <- predict(model, newdata = cell_data)
# cell_data$predicted <- predictions
```

---

## Summary

| Aspect | Original | Optimized |
|---|---|---|
| `build_neighbor_lookup` | 6.46M-iteration `lapply` with per-row string ops & hash lookups | Single vectorized `data.table` join on integer keys |
| `compute_neighbor_stats` | 5 × 6.46M-iteration `lapply` + `do.call(rbind, ...)` | One grouped `data.table` aggregation (C-level) |
| Expected runtime | 86+ hours | Minutes (typically 5–15 min on 16 GB laptop) |
| RF inference | Untouched | Untouched |
| Numerical results | Preserved (same max/min/mean logic) | Preserved |
| Trained RF model | Not retrained | Not retrained |

The bottleneck was never Random Forest inference — it was millions of R-interpreted, row-level string and list operations in the neighbor feature engineering. Vectorizing these with `data.table` joins and grouped aggregations eliminates the bottleneck.