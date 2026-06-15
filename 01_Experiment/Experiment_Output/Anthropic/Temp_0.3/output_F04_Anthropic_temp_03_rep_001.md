 # Diagnosis, Optimization Strategy, and Working R Code

---

## Diagnosis

The bottleneck is **not** the Random Forest inference. It is the two spatial neighbor feature construction functions, and the root causes are:

### 1. `build_neighbor_lookup` — Row-level `lapply` over 6.46 million rows

Each iteration performs:
- A character coercion and named-vector lookup (`id_to_ref[as.character(data$id[i])]`).
- A `paste()` call to build neighbor keys.
- A named-vector lookup into `idx_lookup` (which is a 6.46M-element named character vector — each lookup is O(n) hash probe on a very large vector).
- An `is.na` filter.

Doing this 6.46 million times in an interpreted `lapply` loop is extremely expensive. The named-vector `idx_lookup` with ~6.46M entries makes each key lookup slow, and the per-row `paste` and subsetting compounds the cost.

### 2. `compute_neighbor_stats` — Row-level `lapply` over 6.46 million rows

Each iteration subsets a numeric vector, removes NAs, and computes `max`, `min`, `mean`. This is called 5 times (once per variable), so ~32.3 million R-level function calls. The `do.call(rbind, result)` on a 6.46M-element list of 3-vectors is also slow.

### 3. Overall: ~6.46M × (key construction + lookup) + 5 × 6.46M × (subset + stats) ≈ 86+ hours

---

## Optimization Strategy

**Replace all row-level R loops with vectorized `data.table` joins and grouped aggregations.**

The key insight: the neighbor lookup is a **join** problem. Each cell-year needs to be joined to its neighbors' cell-years, then grouped statistics are computed. `data.table` performs this with optimized C-level code.

**Steps:**

1. **Build an edge list** (a two-column table of `id → neighbor_id`) from the `nb` object — done once, ~1.37M rows.
2. **Join** the edge list to the panel data on `(neighbor_id, year)` to get neighbor variable values — this is a keyed `data.table` merge, extremely fast.
3. **Group by** `(id, year)` and compute `max`, `min`, `mean` in one pass per variable.
4. **Merge** the results back to the main table.

This eliminates all row-level R loops, all `paste`-based key construction, and all named-vector lookups. Expected runtime: **minutes, not hours**.

The trained Random Forest model is untouched. The numerical results (max, min, mean of neighbor values) are identical.

---

## Working R Code

```r
library(data.table)

# ---------------------------------------------------------------
# STEP 0 — Convert the nb object to a data.table edge list (once)
# ---------------------------------------------------------------
build_edge_list_dt <- function(id_order, neighbors) {
  # neighbors is an nb object (list of integer index vectors)
  # id_order is the vector mapping position -> cell id
  from_ids <- rep(id_order, lengths(neighbors))
  to_ids   <- id_order[unlist(neighbors)]
  data.table(id = from_ids, neighbor_id = to_ids)
}

edge_dt <- build_edge_list_dt(id_order, rook_neighbors_unique)

# ---------------------------------------------------------------
# STEP 1 — Convert panel data to data.table and set key
# ---------------------------------------------------------------
dt <- as.data.table(cell_data)
setkey(dt, id, year)

# ---------------------------------------------------------------
# STEP 2 — Compute neighbor features for all variables at once
# ---------------------------------------------------------------
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset to only the columns we need for the neighbor join
# (id, year, and the 5 source variables)
cols_needed <- c("id", "year", neighbor_source_vars)
dt_slim <- dt[, ..cols_needed]
setnames(dt_slim, "id", "neighbor_id")
setkey(dt_slim, neighbor_id, year)

# Join: for every (id, year) pair, look up each neighbor's values
# edge_dt tells us who the neighbors are; we join on (neighbor_id, year)
edge_year <- edge_dt[dt[, .(id, year)], on = "id", allow.cartesian = TRUE, nomatch = 0L]
# edge_year now has columns: id, neighbor_id, year

# Merge in the neighbor values
setkey(edge_year, neighbor_id, year)
edge_vals <- dt_slim[edge_year, on = .(neighbor_id, year), nomatch = NA]
# edge_vals has: neighbor_id, year, ntl, ec, ..., id

# ---------------------------------------------------------------
# STEP 3 — Grouped aggregation: max, min, mean per (id, year)
# ---------------------------------------------------------------
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(max(.(as.name(v)),   na.rm = TRUE)),
    bquote(min(.(as.name(v)),   na.rm = TRUE)),
    bquote(mean(.(as.name(v)),  na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", v, c("_max", "_min", "_mean"))
}))

# Build the aggregation call programmatically
agg_list <- setNames(agg_exprs, agg_names)

neighbor_stats <- edge_vals[,
  lapply(agg_list, eval, envir = .SD),
  by = .(id, year),
  .SDcols = neighbor_source_vars
]

# Handle Inf/-Inf from max/min on all-NA groups (replace with NA)
inf_cols <- grep("_max$|_min$", names(neighbor_stats), value = TRUE)
for (col in inf_cols) {
  set(neighbor_stats, which(is.infinite(neighbor_stats[[col]])), col, NA_real_)
}

# ---------------------------------------------------------------
# STEP 4 — Merge back to the main data
# ---------------------------------------------------------------
setkey(neighbor_stats, id, year)
setkey(dt, id, year)
dt <- neighbor_stats[dt, on = .(id, year)]

# Convert back to data.frame if downstream code expects it
cell_data <- as.data.frame(dt)
```

**If the programmatic `agg_list` evaluation feels fragile**, here is a simpler, equally fast alternative for Step 3:

```r
# ---------------------------------------------------------------
# STEP 3 (alternative) — explicit aggregation, still fully vectorized
# ---------------------------------------------------------------
compute_neighbor_features_dt <- function(edge_vals, var_name) {
  v <- var_name
  edge_vals[
    !is.na(get(v)),
    .(
      nb_max  = max(get(v)),
      nb_min  = min(get(v)),
      nb_mean = mean(get(v))
    ),
    by = .(id, year)
  ] |>
    setnames(
      c("nb_max", "nb_min", "nb_mean"),
      paste0("neighbor_", v, c("_max", "_min", "_mean"))
    )
}

# Run for each variable and merge
setkey(dt, id, year)

for (var_name in neighbor_source_vars) {
  stats_dt <- compute_neighbor_features_dt(edge_vals, var_name)
  setkey(stats_dt, id, year)
  dt <- stats_dt[dt, on = .(id, year)]
}

cell_data <- as.data.frame(dt)
```

---

## Why This Preserves Correctness

| Concern | Guarantee |
|---|---|
| **Trained RF model** | Untouched — we only rebuild the input features, not the model. |
| **Numerical estimand** | Identical — `max`, `min`, `mean` over the same neighbor sets, same NA handling. |
| **Neighbor topology** | The edge list is derived from the same `rook_neighbors_unique` nb object. |
| **Panel alignment** | The join on `(neighbor_id, year)` ensures temporal alignment, identical to the original `paste(id, year)` key logic. |

---

## Expected Performance

| Stage | Original | Optimized |
|---|---|---|
| Neighbor lookup build | ~hours (6.46M `lapply` iterations with `paste` + named vector lookup) | ~seconds (vectorized `rep`/`unlist` to build edge list) |
| Neighbor stats (×5 vars) | ~hours (32.3M R-level loops) | ~minutes (`data.table` grouped aggregation in C) |
| **Total** | **86+ hours** | **~2–10 minutes** |

The dominant cost becomes the `data.table` keyed join and grouped aggregation, both of which are implemented in C with radix-sort keys and are orders of magnitude faster than interpreted R loops over millions of rows.