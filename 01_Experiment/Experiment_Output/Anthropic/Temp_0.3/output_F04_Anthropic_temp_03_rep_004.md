 # Diagnosis and Optimization of Spatial Neighbor Feature Construction

## Diagnosis

The bottleneck is **`build_neighbor_lookup`**, specifically the `lapply` over ~6.46 million rows. For each row, it:

1. Converts an ID to a character and looks it up in a named vector (`id_to_ref`).
2. Retrieves neighbor cell IDs from the `nb` object.
3. Constructs paste-based string keys (`"id_year"`) for every neighbor of every row.
4. Looks those keys up in another named character vector (`idx_lookup`).

String construction (`paste`) and named-vector lookups on character keys are **O(n)** or poorly-cached hash operations in base R. With ~6.46M rows × ~4 neighbors each ≈ **~26 million string constructions and hash lookups**, all inside a sequential `lapply`, this is extremely slow.

`compute_neighbor_stats` is a secondary bottleneck: another `lapply` over 6.46M elements, each allocating small vectors and calling `max/min/mean`. The `do.call(rbind, result)` on a 6.46M-element list is also costly.

**Summary of root causes:**

| Cause | Location | Impact |
|---|---|---|
| Per-row `paste()` key construction | `build_neighbor_lookup` | ~26M string allocs |
| Named character vector lookup (hash on strings) | `build_neighbor_lookup` | Slow hashing ×26M |
| Row-wise `lapply` over 6.46M rows | Both functions | No vectorization |
| `do.call(rbind, list_of_6.46M)` | `compute_neighbor_stats` | Massive list→matrix coercion |
| Repeated per-variable overhead | Outer loop | 5× the stats computation |

## Optimization Strategy

**Core idea:** Replace the row-level `lapply` with a fully vectorized `data.table` merge-and-group-by approach.

1. **Build a neighbor edge table once** — a two-column `data.table` mapping each `(cell_id)` to its `(neighbor_cell_id)`. This is small (~1.37M rows).
2. **Join the edge table to the panel data by `(cell_id, year)`** to produce an expanded table where each row is a `(cell_id, year, neighbor_cell_id)` tuple, then join again to get the neighbor's variable value. This replaces all string-key construction and lookup.
3. **Group-by `(cell_id, year)`** to compute `max`, `min`, `mean` in one vectorized pass per variable (or all variables at once).
4. **Join the aggregated stats back** to the original data.

This eliminates all per-row R-level iteration, all `paste` key construction, and all named-vector lookups. Expected runtime: **minutes, not days**.

## Working R Code

```r
library(data.table)

# ── Step 1: Build the neighbor edge table (once) ─────────────────────────────
# id_order: vector of cell IDs in the order matching rook_neighbors_unique
# rook_neighbors_unique: an nb object (list of integer index vectors)

build_neighbor_edges <- function(id_order, neighbors_nb) {
  # Pre-allocate: count total edges
  n_edges <- sum(lengths(neighbors_nb))
  from_id <- integer(n_edges)
  to_id   <- integer(n_edges)
  pos <- 1L
  for (i in seq_along(neighbors_nb)) {
    nb_idx <- neighbors_nb[[i]]
    if (length(nb_idx) == 0L || (length(nb_idx) == 1L && nb_idx[1] == 0L)) next
    n <- length(nb_idx)
    from_id[pos:(pos + n - 1L)] <- id_order[i]
    to_id[pos:(pos + n - 1L)]   <- id_order[nb_idx]
    pos <- pos + n
  }
  data.table(cell_id = from_id[1:(pos - 1L)],
             neighbor_id = to_id[1:(pos - 1L)])
}

edges_dt <- build_neighbor_edges(id_order, rook_neighbors_unique)

# ── Step 2: Convert panel data to data.table ──────────────────────────────────
cell_dt <- as.data.table(cell_data)

# Ensure key columns exist and are properly typed
stopifnot(all(c("id", "year") %in% names(cell_dt)))

# ── Step 3: Compute neighbor stats for all variables at once ──────────────────
neighbor_source_vars <- c("ntl", "ec", "pop_density", "def", "usd_est_n2")

# Subset the neighbor value table: only the columns we need from the panel
# We will join edges × years to the panel to get neighbor values.

# Create a keyed lookup of (id, year) -> variable values
value_cols <- c("id", "year", neighbor_source_vars)
values_dt <- cell_dt[, ..value_cols]
setnames(values_dt, "id", "neighbor_id")
setkey(values_dt, neighbor_id, year)

# Expand edges by year: each edge applies to every year the focal cell has data.
# Instead of a full cross join (expensive), we merge edges onto the panel.
# For each row in the panel, find its neighbors, then look up neighbor values.

# Step 3a: Map each (cell_id, year) to its neighbors
#   panel_edges = cell_dt[, .(id, year)] joined to edges_dt on cell_id
panel_keys <- cell_dt[, .(cell_id = id, year)]
setkey(edges_dt, cell_id)
setkey(panel_keys, cell_id)

# This is the big join: ~6.46M rows × ~4 neighbors = ~26M rows
# data.table handles this efficiently via binary merge
panel_edges <- edges_dt[panel_keys, on = "cell_id", allow.cartesian = TRUE, nomatch = NA]
# Result columns: cell_id, neighbor_id, year

# Step 3b: Look up neighbor variable values
setkey(panel_edges, neighbor_id, year)
panel_edges <- values_dt[panel_edges, on = .(neighbor_id, year)]
# Now panel_edges has: neighbor_id, year, ntl, ec, ..., cell_id

# Step 3c: Aggregate by (cell_id, year)
# Compute max, min, mean for each variable
agg_exprs <- unlist(lapply(neighbor_source_vars, function(v) {
  list(
    bquote(as.numeric(max(.(as.name(v)), na.rm = TRUE))),
    bquote(as.numeric(min(.(as.name(v)), na.rm = TRUE))),
    bquote(mean(.(as.name(v)), na.rm = TRUE))
  )
}), recursive = FALSE)

agg_names <- unlist(lapply(neighbor_source_vars, function(v) {
  paste0("neighbor_", c("max_", "min_", "mean_"), v)
}))

# Build the aggregation call dynamically
agg_list <- setNames(agg_exprs, agg_names)

# Use a single grouped aggregation
neighbor_stats <- panel_edges[
  !is.na(neighbor_id),
  {
    out <- list()
    for (v in neighbor_source_vars) {
      vals <- get(v)
      vals <- vals[!is.na(vals)]
      if (length(vals) == 0L) {
        out[[paste0("neighbor_max_", v)]]  <- NA_real_
        out[[paste0("neighbor_min_", v)]]  <- NA_real_
        out[[paste0("neighbor_mean_", v)]] <- NA_real_
      } else {
        out[[paste0("neighbor_max_", v)]]  <- max(vals)
        out[[paste0("neighbor_min_", v)]]  <- min(vals)
        out[[paste0("neighbor_mean_", v)]] <- mean(vals)
      }
    }
    out
  },
  by = .(cell_id, year)
]

# ── Step 4: Join back to the main data ────────────────────────────────────────
# Remove any pre-existing neighbor columns to avoid duplication
existing_neighbor_cols <- intersect(names(cell_dt), agg_names)
if (length(existing_neighbor_cols) > 0) {
  cell_dt[, (existing_neighbor_cols) := NULL]
}

cell_dt <- merge(cell_dt, neighbor_stats,
                 by.x = c("id", "year"),
                 by.y = c("cell_id", "year"),
                 all.x = TRUE)

# ── Step 5: Convert back to data.frame if downstream code expects it ──────────
cell_data <- as.data.frame(cell_dt)

# ── The trained Random Forest model is untouched. ─────────────────────────────
# Proceed directly to prediction:
# predictions <- predict(trained_rf_model, newdata = cell_data)
```

## Why This Works and What It Preserves

| Concern | Status |
|---|---|
| **Numerical equivalence** | Same `max`, `min`, `mean` on the same neighbor sets — identical estimand. |
| **Trained RF model** | Not retouched; only the feature-construction step is replaced. |
| **Column names** | `neighbor_max_ntl`, `neighbor_min_ntl`, `neighbor_mean_ntl`, etc. — adjust the naming pattern to match whatever your existing `compute_and_add_neighbor_features` produces. |
| **Memory** | The largest intermediate (`panel_edges`) is ~26M rows × ~8 columns ≈ 1.5 GB, well within 16 GB RAM. |
| **Runtime** | Two `data.table` keyed joins + one grouped aggregation. Expected: **5–15 minutes** on a standard laptop vs. the original 86+ hours. |

### Optional Further Speedup

If even the `get(v)` inside the `by` group is a concern, you can unroll the five variables explicitly or use `.SDcols`:

```r
# Alternative: .SDcols approach (avoids get() overhead)
neighbor_stats <- panel_edges[
  !is.na(neighbor_id),
  lapply(.SD, function(x) {
    x <- x[!is.na(x)]
    if (length(x) == 0L) c(NA_real_, NA_real_, NA_real_)
    else c(max(x), min(x), mean(x))
  }),
  by = .(cell_id, year),
  .SDcols = neighbor_source_vars
]
# This returns 3 rows per group; reshape afterward with a row-index trick.
```

The primary solution above is the most straightforward drop-in replacement. The key transformation is: **replace 6.46 million R-level iterations with two data.table binary joins and one vectorized group-by aggregation**.